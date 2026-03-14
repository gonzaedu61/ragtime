import os
import json
import hashlib
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from Utilities import Simple_Progress_Bar


class Hierarchy_Embedder:
    """
    Unified pipeline:

    1. Load hierarchy JSON.
    2. Optionally restrict processing to a branch_id (subtree).
    3. Detect language from multilang blocks (if not overridden).
    4. For each cluster in the selected subtree:
       - Build synthetic query from label/summary/keywords.
       - Perform dense + sparse hybrid retrieval.
       - Re-rank with keyword overlap.
       - Deduplicate chunks.
       - Keep top-N retrieved chunks (cluster_chunks).
    5. Compute drift metrics (overlap_distance, semantic_shift) on the hierarchy.
    6. For each cluster:
       - Load additional metadata from <base_dir>/<cluster_id>/<cluster_id>_category.json.
       - Add is_leaf and depth.
       - Add drift metrics into metadata.drift.
       - Compute semantic_lineage_path (labels of ancestor chain).
       - Compute semantic_heading_path from retrieved chunk heading_path metadata.
       - Build cluster record (schema below).
       - (Commented) delete + upsert record in VDB (with embedding).
       - Optionally save <cluster_id>_enrichment.json (record without embedding).
    7. Progress bar covers all per-cluster steps.

    Cluster record schema (for VDB; enrichment JSON omits `embedding`):

    {
      "cluster_id": "string",
      "label": "string",
      "summary": "string",
      "keywords": ["string", "..."],
      "embedding": [...],   # VDB only

      "metadata": {
        "children_count": 6,
        "size": 15,
        "is_document_cluster": true,
        "is_leaf": true,
        "depth": 2,

        "semantic_lineage_path": [
          "Printing",
          "Thermal Printing",
          "Direct Thermal",
          "Label Printers"
        ],

        "semantic_heading_path": [
          "Intro > Thermal Basics",
          "Components > Printhead",
          "Applications > Labels"
        ],

        "source_documents": [...],
        "text_class": [...],

        "drift": {
          "overlap_distance": 0.42,
          "semantic_shift": 0.13
        }
      },

      "cluster_chunks": ["chunk_id2", "chunk_id4"],
      "leaf_chunks": ["chunk_id1", "chunk_id2"]
    }
    """

    def __init__(
        self,
        vector_db,
        clusters_vdb,
        embedder,
        category_base_dir: str,
        candidate_k: int = 40,
        final_k: int = 10,
        verbose: bool = False,
        progress_bar: bool = False,
        language: Optional[str] = None,
        hybrid_alpha: float = 0.5,
        keyword_weight: float = 0.2,
        save_cluster_files: bool = True,
        max_heading_paths: int = 5,
    ):
        """
        vector_db: object with:
            - all_chunks() -> list of chunks
            - search(embedding, top_n) -> list of chunks with fields:
                { "chunk_id", "text", "metadata", "score" }
            - get_embedding(chunk_id) -> np.array-like (for drift semantic shift)
            - (later, commented) delete_cluster(cluster_id), upsert_cluster(record)
        embedder: object with:
            - embed(text, progress_bar=False) -> embedding vector
        category_base_dir: base directory where per-cluster folders live.
        """
        self.vector_db = vector_db
        self.clusters_vdb = clusters_vdb
        self.embedder = embedder
        self.category_base_dir = category_base_dir

        self.candidate_k = candidate_k
        self.final_k = final_k

        self.verbose = verbose and not progress_bar
        self.progress_bar_enabled = progress_bar

        self.language_override = language
        self.hybrid_alpha = hybrid_alpha
        self.keyword_weight = keyword_weight

        self.save_cluster_files = save_cluster_files
        self.max_heading_paths = max_heading_paths

        self.progress = None

        # For semantic shift, reuse vector_db as vdb_client
        self.vdb = vector_db

        self._init_sparse_index()

    # ---------------------------------------------------------
    # Logging helper
    # ---------------------------------------------------------
    def log(self, msg: str):
        if self.verbose:
            print(msg)

    # ---------------------------------------------------------
    # Build BM25 sparse index
    # ---------------------------------------------------------
    def _init_sparse_index(self):
        try:
            self.all_chunks = self.vector_db.all_chunks()
        except Exception:
            self.all_chunks = None

        if not self.all_chunks:
            self.log("WARNING: vector_db.all_chunks() returned no data. Using dense-only retrieval.")
            self.bm25 = None
            self.chunk_by_id = {}
            return

        tokenized = [c["text"].split() for c in self.all_chunks]
        self.bm25 = BM25Okapi(tokenized)

        self.chunk_by_id = {c["chunk_id"]: c for c in self.all_chunks}

        self.log(f"Initialized BM25 index with {len(self.all_chunks)} chunks.")

    # ---------------------------------------------------------
    # Load input
    # ---------------------------------------------------------
    def load_input(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------------------------------------------------------
    # Detect language from hierarchy
    # ---------------------------------------------------------
    def detect_language(self, hierarchy: Dict[str, Any]) -> str:
        if self.language_override:
            self.log(f"Using user-specified language: {self.language_override}")
            return self.language_override

        def find_first_multilang(node):
            clusters = node.get("clusters", [])
            for c in clusters:
                if "multilang" in c:
                    return c["multilang"]
                children = c.get("children")
                if children is not None:
                    result = find_first_multilang(children)
                    if result:
                        return result
            return None

        multilang = find_first_multilang(hierarchy)
        if not multilang:
            raise ValueError("No multilingual data found in hierarchy.")

        languages = list(multilang.keys())
        lang = languages[0]
        self.log(f"Detected languages {languages}. Using: {lang}")
        return lang

    # ---------------------------------------------------------
    # Branch selection
    # ---------------------------------------------------------
    def _find_branch_root(self, hierarchy: Dict[str, Any], branch_id: Optional[str]):
        """
        If branch_id is None, return the original hierarchy root.
        Otherwise, find the node with cluster_id == branch_id and return that node
        as the root of the processing subtree.
        """
        if branch_id is None:
            return hierarchy

        def recurse(node):
            if node.get("cluster_id") == branch_id:
                return node
            children = node.get("children")
            if children and "clusters" in children:
                for child in children["clusters"]:
                    found = recurse(child)
                    if found is not None:
                        return found
            return None

        if "cluster_id" in hierarchy:
            root_candidate = recurse(hierarchy)
        else:
            root_candidate = None
            for c in hierarchy.get("clusters", []):
                root_candidate = recurse(c)
                if root_candidate is not None:
                    break

        if root_candidate is None:
            raise ValueError(f"branch_id '{branch_id}' not found in hierarchy.")
        return root_candidate

    # ---------------------------------------------------------
    # Build synthetic query
    # ---------------------------------------------------------
    def build_query(self, label: str, summary: str, keywords: List[str]) -> str:
        keywords_str = ", ".join(keywords)
        return (
            f"Topic label: {label}\n"
            f"Summary: {summary}\n"
            f"Keywords: {keywords_str}\n"
            f"Retrieve text relevant to this topic."
        )

    # ---------------------------------------------------------
    # Keyword score
    # ---------------------------------------------------------
    def keyword_score(self, query: str, text: str) -> int:
        q_words = set(query.lower().split())
        t_words = set(text.lower().split())
        return len(q_words & t_words)

    # ---------------------------------------------------------
    # Deduplication helper
    # ---------------------------------------------------------
    def dedupe_chunks(self, results):
        seen = set()
        deduped = []
        for r in results:
            text = r["chunk"]["text"].strip()
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                deduped.append(r)
        return deduped

    # ---------------------------------------------------------
    # Efficient hybrid retrieval
    # ---------------------------------------------------------
    def hybrid_retrieve(self, query_embedding, query_text):
        k = self.candidate_k

        dense_results = self.vector_db.search(query_embedding, top_n=k)
        dense_map = {c["chunk_id"]: float(c.get("score", 0.0)) for c in dense_results}

        if not getattr(self, "all_chunks", None) or not self.bm25:
            return [
                {
                    "chunk": c,
                    "dense_score": dense_map[c["chunk_id"]],
                    "sparse_score": 0.0,
                    "hybrid_score": dense_map[c["chunk_id"]],
                }
                for c in dense_results
            ]

        sparse_scores = self.bm25.get_scores(query_text.split())
        if len(sparse_scores) <= k:
            top_sparse_idx = list(range(len(sparse_scores)))
        else:
            top_sparse_idx = np.argpartition(sparse_scores, -k)[-k:]

        sparse_map = {
            self.all_chunks[idx]["chunk_id"]: float(sparse_scores[idx])
            for idx in top_sparse_idx
        }

        candidate_ids = set(dense_map.keys()) | set(sparse_map.keys())

        if sparse_map:
            norm = np.linalg.norm(list(sparse_map.values())) + 1e-6
        else:
            norm = 1.0

        hybrid = []
        for cid in candidate_ids:
            dense = dense_map.get(cid, 0.0)
            sparse = sparse_map.get(cid, 0.0) / norm

            chunk = next((c for c in dense_results if c["chunk_id"] == cid), None)
            if chunk is None:
                chunk = self.chunk_by_id[cid]

            hybrid_score = self.hybrid_alpha * dense + (1 - self.hybrid_alpha) * sparse

            hybrid.append({
                "chunk": chunk,
                "dense_score": dense,
                "sparse_score": sparse,
                "hybrid_score": hybrid_score,
            })

        hybrid_sorted = sorted(hybrid, key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_sorted[:k]

    # ---------------------------------------------------------
    # Keyword re-ranking
    # ---------------------------------------------------------
    def rerank_keywords(self, results, query_text):
        reranked = []
        for r in results:
            kw = self.keyword_score(query_text, r["chunk"]["text"])
            final = r["hybrid_score"] + self.keyword_weight * kw
            r["keyword_score"] = kw
            r["final_score"] = float(final)
            reranked.append(r)

        return sorted(reranked, key=lambda x: x["final_score"], reverse=True)

    # ---------------------------------------------------------
    # Hierarchy traversal helpers
    # ---------------------------------------------------------
    def _iter_clusters(self, root_node: Dict[str, Any]):
        """
        Yield all cluster nodes in the subtree rooted at root_node.
        """
        if "cluster_id" in root_node:
            yield root_node
            children = root_node.get("children")
            if children and "clusters" in children:
                for child in children["clusters"]:
                    yield from self._iter_clusters(child)
        else:
            for c in root_node.get("clusters", []):
                yield from self._iter_clusters(c)

    def _is_leaf(self, node: Dict[str, Any]) -> bool:
        """
        A leaf is defined as node with children == None and having "chunk_ids".
        """
        return node.get("children") is None and bool(node.get("chunk_ids"))

    # ---------------------------------------------------------
    # Drift helpers
    # ---------------------------------------------------------
    def _collect_leaf_chunks(self, node):
        """
        Recursively collect all chunk_ids from descendant leaves.
        """
        if node.get("children") is None:
            return node.get("chunk_ids", []) or []

        collected = []
        for child in node["children"].get("clusters", []):
            collected.extend(self._collect_leaf_chunks(child))
        return collected

    def _compute_overlap_distance(self, original, matching):
        if not original and not matching:
            return None
        A, B = set(original), set(matching)
        if not A and not B:
            return None
        return 1 - len(A & B) / len(A | B)

    def _compute_semantic_shift(self, original, matching):
        if not self.vdb or not original or not matching:
            return None

        orig_embs = [self.vdb.get_embedding(cid) for cid in original]
        match_embs = [self.vdb.get_embedding(cid) for cid in matching]

        orig_embs = [e for e in orig_embs if e is not None]
        match_embs = [e for e in match_embs if e is not None]

        if not orig_embs or not match_embs:
            return None

        c_orig = np.mean(np.array(orig_embs), axis=0)
        c_match = np.mean(np.array(match_embs), axis=0)

        denom = (np.linalg.norm(c_orig) * np.linalg.norm(c_match))
        if denom == 0:
            return None

        cos_sim = float(np.dot(c_orig, c_match) / denom)
        return 1 - cos_sim

    def _enrich_node_with_drift(self, node, retrieved_index):
        cid = node.get("cluster_id")

        matching = retrieved_index.get(cid, []) or []
        node["matching_chunks"] = matching

        original = self._collect_leaf_chunks(node)

        node["overlap_distance"] = self._compute_overlap_distance(original, matching)
        node["semantic_shift"] = self._compute_semantic_shift(original, matching)

        children = node.get("children")
        if children and "clusters" in children:
            for child in children["clusters"]:
                self._enrich_node_with_drift(child, retrieved_index)

    def compute_aggregate_drift(self, hierarchy_root):
        overlap_values = []
        semantic_values = []
        weighted_overlap = []
        weighted_semantic = []
        total_size = 0

        def traverse(node):
            nonlocal total_size

            od = node.get("overlap_distance")
            ss = node.get("semantic_shift")
            size = len(self._collect_leaf_chunks(node))

            if size > 0:
                total_size += size

                if od is not None:
                    overlap_values.append(od)
                    weighted_overlap.append(od * size)

                if ss is not None:
                    semantic_values.append(ss)
                    weighted_semantic.append(ss * size)

            children = node.get("children")
            if children and "clusters" in children:
                for child in children["clusters"]:
                    traverse(child)

        traverse(hierarchy_root)

        if total_size == 0 and not overlap_values and not semantic_values:
            return {
                "mean_overlap": None,
                "mean_semantic_shift": None,
                "weighted_mean_overlap": None,
                "weighted_mean_semantic_shift": None,
                "max_overlap": None,
                "max_semantic_shift": None,
                "percentiles_overlap": {},
                "percentiles_semantic_shift": {},
            }

        def safe_mean(values):
            return float(np.mean(values)) if values else None

        def safe_max(values):
            return float(np.max(values)) if values else None

        def safe_percentiles(values):
            if not values:
                return {}
            return {
                "25": float(np.percentile(values, 25)),
                "50": float(np.percentile(values, 50)),
                "75": float(np.percentile(values, 75)),
                "95": float(np.percentile(values, 95)),
            }

        weighted_mean_overlap = (
            float(sum(weighted_overlap) / total_size)
            if weighted_overlap and total_size > 0
            else None
        )
        weighted_mean_semantic = (
            float(sum(weighted_semantic) / total_size)
            if weighted_semantic and total_size > 0
            else None
        )

        return {
            "mean_overlap": safe_mean(overlap_values),
            "mean_semantic_shift": safe_mean(semantic_values),
            "weighted_mean_overlap": weighted_mean_overlap,
            "weighted_mean_semantic_shift": weighted_mean_semantic,
            "max_overlap": safe_max(overlap_values),
            "max_semantic_shift": safe_max(semantic_values),
            "percentiles_overlap": safe_percentiles(overlap_values),
            "percentiles_semantic_shift": safe_percentiles(semantic_values),
        }

    def compute_drift_by_depth(self, hierarchy_root):
        drift_by_depth = {}

        def traverse(node, depth):
            od = node.get("overlap_distance")
            ss = node.get("semantic_shift")

            if depth not in drift_by_depth:
                drift_by_depth[depth] = {"overlap": [], "semantic": []}

            if od is not None:
                drift_by_depth[depth]["overlap"].append(od)
            if ss is not None:
                drift_by_depth[depth]["semantic"].append(ss)

            children = node.get("children")
            if children and "clusters" in children:
                for child in children["clusters"]:
                    traverse(child, depth + 1)

        traverse(hierarchy_root, 0)

        summary = {}
        for depth, values in drift_by_depth.items():
            overlaps = values["overlap"]
            semantics = values["semantic"]

            mean_overlap = float(np.mean(overlaps)) if overlaps else None
            mean_semantic = float(np.mean(semantics)) if semantics else None

            summary[depth] = {
                "mean_overlap": mean_overlap,
                "mean_semantic_shift": mean_semantic,
                "cluster_count": len(overlaps) + len(semantics) - min(len(overlaps), len(semantics)),
            }

        return summary

    # ---------------------------------------------------------
    # Extra metadata loading and enrichment files
    # ---------------------------------------------------------
    def _load_cluster_category_metadata(self, cluster_id: str) -> Dict[str, Any]:
        folder = os.path.join(self.category_base_dir, cluster_id)
        path = os.path.join(folder, f"{cluster_id}_category.json")
        if not os.path.isfile(path):
            self.log(f"[{cluster_id}] No category file found at {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.log(f"[{cluster_id}] Error loading category file: {e}")
            return {}

    def _save_cluster_enrichment_file(self, cluster_id: str, record: Dict[str, Any]):
        folder = os.path.join(self.category_base_dir, cluster_id)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{cluster_id}_enrichment.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        self.log(f"[{cluster_id}] Saved enrichment file at {path}")

    # ---------------------------------------------------------
    # Semantic paths
    # ---------------------------------------------------------
    def _build_cluster_index(self, root_container: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Build a mapping cluster_id -> node for the entire hierarchy.
        """
        index = {}

        def recurse(node):
            cid = node.get("cluster_id")
            if cid is not None:
                index[cid] = node
            children = node.get("children")
            if children and "clusters" in children:
                for child in children["clusters"]:
                    recurse(child)

        if "cluster_id" in root_container:
            recurse(root_container)
        else:
            for c in root_container.get("clusters", []):
                recurse(c)
        return index

    def _semantic_lineage_path(self, cluster_id: str, cluster_index: Dict[str, Dict[str, Any]], lang: str) -> List[str]:
        """
        Build semantic lineage path from ancestor labels, excluding root if it has no meaningful label.
        Uses cluster_id segments to walk up the tree.
        """
        if not cluster_id:
            return []

        segments = cluster_id.split(".")
        lineage_ids = []
        for i in range(1, len(segments) + 1):
            lineage_ids.append(".".join(segments[:i]))

        labels = []
        for cid in lineage_ids:
            node = cluster_index.get(cid)
            if not node:
                continue

            label = node.get("label")
            multilang = node.get("multilang")
            if multilang and lang in multilang:
                label = multilang[lang].get("label", label)

            if label:
                labels.append(label)

        # Optionally drop a generic root label if needed; for now we keep all non-empty labels.
        return labels

    def _semantic_heading_path_from_chunks(self, retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Build semantic heading path list from chunk metadata.heading_path.
        Returns top-N most frequent canonical heading strings.
        """
        freq: Dict[str, int] = {}

        for r in retrieved_chunks:
            meta = r.get("metadata", {})
            heading_path = meta.get("heading_path")
            if not heading_path or not isinstance(heading_path, list):
                continue
            parts = [str(p).strip() for p in heading_path if str(p).strip()]
            if not parts:
                continue
            key = " > ".join(parts)
            freq[key] = freq.get(key, 0) + 1

        if not freq:
            return []

        sorted_paths = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        top = sorted_paths[: self.max_heading_paths]
        return [p[0] for p in top]

    # ---------------------------------------------------------
    # Main public API
    # ---------------------------------------------------------
    def process(
        self,
        hierarchy_path: str,
        branch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
        - Load hierarchy.
        - Optionally restrict to branch_id.
        - Detect language.
        - For each cluster in subtree:
            * build query
            * hybrid retrieval
            * keyword rerank
            * dedupe
            * keep top-N chunks (cluster_chunks)
        - Compute drift metrics.
        - For each cluster:
            * load extra metadata
            * mark is_leaf + depth
            * add drift into metadata.drift
            * compute semantic_lineage_path
            * compute semantic_heading_path
            * build cluster record
            * (commented) delete + upsert in VDB
            * optionally save <cluster_id>_enrichment.json
        Returns the enriched hierarchy (in-memory only).
        """
        self.log(f"Loading hierarchy from {hierarchy_path}")
        hierarchy = self.load_input(hierarchy_path)

        if isinstance(hierarchy, dict) and "clusters" in hierarchy and isinstance(hierarchy["clusters"], list):
            if len(hierarchy["clusters"]) == 0:
                raise ValueError("Hierarchy has an empty 'clusters' list.")
            logical_root = hierarchy["clusters"][0]
            root_container = hierarchy
        else:
            logical_root = hierarchy
            root_container = hierarchy

        branch_root = self._find_branch_root(root_container, branch_id)
        lang = self.detect_language(root_container)

        clusters_to_process = list(self._iter_clusters(branch_root))
        clusters_to_process = sorted(
            clusters_to_process,
            key=lambda n: n.get("cluster_id", "")
        )

        total_steps = max(1, len(clusters_to_process) * 2)
        self.progress = Simple_Progress_Bar(total_steps, enabled=self.progress_bar_enabled)

        retrieved_index: Dict[str, List[str]] = {}

        # Retrieval per cluster
        for node in clusters_to_process:
            cid = node.get("cluster_id")
            if cid is None:
                continue

            label = node.get("label")
            summary = node.get("summary")
            keywords = node.get("keywords", [])

            multilang = node.get("multilang")
            if multilang and lang in multilang:
                lang_block = multilang[lang]
                label = lang_block.get("label", label)
                summary = lang_block.get("summary", summary)
                keywords = lang_block.get("keywords", keywords)

            cluster_keywords = sorted(set(keywords or []))
            label = label or f"cluster_{cid}"
            summary = summary or ""

            node["label"] = label
            node["summary"] = summary
            node["keywords"] = cluster_keywords

            query_text = self.build_query(label, summary, cluster_keywords)
            query_embedding = self.embedder.embed(query_text, progress_bar=False)

            hybrid = self.hybrid_retrieve(query_embedding, query_text)
            #self.progress.update(label=f"[{cid}] Retrieval")

            reranked = self.rerank_keywords(hybrid, query_text)
            #self.progress.update(label=f"[{cid}] Re-ranking")

            deduped = self.dedupe_chunks(reranked)
            final = deduped[: self.final_k]
            self.progress.update(label=f"[{cid}] Retrieval")

            chunk_ids = [r["chunk"]["chunk_id"] for r in final]
            retrieved_index[cid] = chunk_ids

            node["retrieved_chunks"] = [
                {
                    "chunk_id": r["chunk"]["chunk_id"],
                    "text": r["chunk"]["text"],
                    "metadata": r["chunk"].get("metadata", {}),
                    "final_score": r["final_score"],
                }
                for r in final
            ]

        # Drift enrichment on full logical root
        self._enrich_node_with_drift(logical_root, retrieved_index)

        global_stats = self.compute_aggregate_drift(logical_root)
        drift_by_depth = self.compute_drift_by_depth(logical_root)

        logical_root["_global_drift_statistics"] = global_stats
        logical_root["_drift_by_depth"] = drift_by_depth

        # Build cluster index for semantic lineage
        cluster_index = self._build_cluster_index(root_container)

        # Per-cluster metadata + record building + optional file save
        for node in clusters_to_process:
            cid = node.get("cluster_id")
            if cid is None:
                continue

            extra_meta = self._load_cluster_category_metadata(cid)

            node_metadata = node.get("metadata") or {}
            node["metadata"] = node_metadata

            node_metadata.update(extra_meta)

            is_leaf = self._is_leaf(node)
            node_metadata["is_leaf"] = is_leaf

            depth = cid.count(".") if isinstance(cid, str) else None
            node_metadata["depth"] = depth

            # Drift into metadata.drift
            drift_block = {
                "overlap_distance": node.get("overlap_distance"),
                "semantic_shift": node.get("semantic_shift"),
            }
            node_metadata["drift"] = drift_block

            # Semantic lineage path
            semantic_lineage = self._semantic_lineage_path(cid, cluster_index, lang)
            node_metadata["semantic_lineage_path"] = semantic_lineage

            # Semantic heading path from retrieved chunks
            retrieved_chunks = node.get("retrieved_chunks", [])
            semantic_heading = self._semantic_heading_path_from_chunks(retrieved_chunks)
            node_metadata["semantic_heading_path"] = semantic_heading

            # Leaf chunks
            leaf_chunks = node.get("chunk_ids", []) if is_leaf else []

            # Cluster record (without embedding)
            record = {
                "cluster_id": cid,
                "label": node.get("label"),
                "summary": node.get("summary"),
                "keywords": node.get("keywords", []),
                "metadata": node_metadata,
                "cluster_chunks": retrieved_index.get(cid, []),
                "leaf_chunks": leaf_chunks,
            }

            # Cluster-level embedding text
            text_parts = [
                str(record.get("label") or ""),
                str(record.get("summary") or ""),
                ", ".join(record.get("keywords") or []),
                " | ".join(node_metadata.get("semantic_lineage_path", [])),
                " | ".join(node_metadata.get("semantic_heading_path", [])),
            ]
            cluster_text_for_embedding = "\n".join([t for t in text_parts if t])
            cluster_embedding = self.embedder.embed(cluster_text_for_embedding, progress_bar=False)

            record_with_embedding = dict(record)
            record_with_embedding["embedding"] = cluster_embedding


            # --------------------------------------------
            # VDB WRITE
            # --------------------------------------------
            try:
                # Delete any existing cluster record
                self.clusters_vdb.delete(where={"cluster_id": cid})
            except Exception as e:
                self.log(f"[{cid}] delete failed: {e}")

            try:
                # Insert new cluster record

                record_json = json.dumps(record, ensure_ascii=False)


                self.clusters_vdb.add(
                    ids=[cid],
                    texts=[cluster_text_for_embedding],   # or "" if you prefer no document text
                    embeddings=[cluster_embedding],
                    metadata=[{
                        "cluster_id": cid,
                        "record_json": record_json
                    }]
                )

            except Exception as e:
                self.log(f"[{cid}] add failed: {e}")


            if self.save_cluster_files:
                self._save_cluster_enrichment_file(cid, record)

            self.progress.update(label=f"[{cid}] Save")

        return root_container
