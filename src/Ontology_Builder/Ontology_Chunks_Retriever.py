import json
from typing import List, Dict, Any
from Utilities import Simple_Progress_Bar
from rank_bm25 import BM25Okapi
import numpy as np
import hashlib


class Ontology_Chunks_Retriever:
    """
    Retrieves top-N relevant raw chunks for each cluster.
    Now includes:
    - efficient hybrid retrieval (dense + sparse)
    - keyword re-ranking
    - chunk deduplication
    - minimal diagnostics
    - configurable candidate_k and final_k
    - deduped cluster keywords
    - per-cluster progress visibility
    """

    def __init__(
        self,
        vector_db,
        embedder,
        candidate_k=40,     # NEW
        final_k=10,         # NEW
        verbose=False,
        progress_bar=False,
        language: str = None,
        hybrid_alpha=0.5,
        keyword_weight=0.2,
    ):
        self.vector_db = vector_db
        self.embedder = embedder

        self.candidate_k = candidate_k
        self.final_k = final_k

        self.verbose = verbose and not progress_bar
        self.progress_bar_enabled = progress_bar

        self.language_override = language
        self.hybrid_alpha = hybrid_alpha
        self.keyword_weight = keyword_weight

        self.progress = None

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
    # Save JSON
    # ---------------------------------------------------------
    def save_json(self, data, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------
    # Detect hierarchy
    # ---------------------------------------------------------
    def is_hierarchy(self, data: Any) -> bool:
        return isinstance(data, dict) and "clusters" in data

    # ---------------------------------------------------------
    # Determine which language to use
    # ---------------------------------------------------------
    def detect_language(self, hierarchy: Dict[str, Any]) -> str:
        if self.language_override:
            self.log(f"Using user-specified language: {self.language_override}")
            return self.language_override

        def find_first_multilang(node):
            for c in node["clusters"]:
                if "multilang" in c:
                    return c["multilang"]
                if c["children"] is not None:
                    result = find_first_multilang(c["children"])
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
    # Flatten hierarchy
    # ---------------------------------------------------------
    def flatten_clusters(self, hierarchy: Dict[str, Any], lang="EN") -> List[Dict[str, Any]]:
        flat = []

        def recurse(node):
            for c in node["clusters"]:
                if "multilang" in c and lang in c["multilang"]:
                    flat.append({
                        "cluster_id": c["cluster_id"],
                        "label": c["multilang"][lang]["label"],
                        "summary": c["multilang"][lang]["summary"],
                        "keywords": c["multilang"][lang]["keywords"],
                    })

                if c["children"] is not None:
                    recurse(c["children"])

        recurse(hierarchy)
        return flat

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

        # Dense top-k
        dense_results = self.vector_db.search(query_embedding, top_n=k)
        dense_map = {c["chunk_id"]: float(c.get("score", 0.0)) for c in dense_results}

        # Dense-only fallback
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

        # Sparse top-k
        sparse_scores = self.bm25.get_scores(query_text.split())
        if len(sparse_scores) <= k:
            top_sparse_idx = list(range(len(sparse_scores)))
        else:
            top_sparse_idx = np.argpartition(sparse_scores, -k)[-k:]

        sparse_map = {
            self.all_chunks[idx]["chunk_id"]: float(sparse_scores[idx])
            for idx in top_sparse_idx
        }

        # Merge candidates
        candidate_ids = set(dense_map.keys()) | set(sparse_map.keys())

        # Normalize sparse
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
    # Minimal diagnostics
    # ---------------------------------------------------------
    def format_diagnostics(self, results):
        return [
            {
                "chunk_id": r["chunk"]["chunk_id"],
                "final_score": r["final_score"],
            }
            for r in results
        ]

    # ---------------------------------------------------------
    # Main retrieval method
    # ---------------------------------------------------------
    def retrieve(
        self,
        input_json_path: str,
        output_json_path: str,
        flattened_debug_path: str = "flattened_clusters.json"
    ):
        self.log(f"Loading input from {input_json_path}")
        data = self.load_input(input_json_path)

        # Step 1: Flatten if needed
        if self.is_hierarchy(data):
            lang = self.detect_language(data)
            clusters = self.flatten_clusters(data, lang=lang)
            self.save_json(clusters, flattened_debug_path)
        else:
            clusters = data

        # Progress bar
        total_steps = len(clusters) * 4
        self.progress = Simple_Progress_Bar(total_steps, enabled=self.progress_bar_enabled)

        results = []

        # Step 3: Retrieval
        for cluster in clusters:
            cid = cluster["cluster_id"]

            # Deduped cluster keywords
            cluster_keywords = sorted(set(cluster.get("keywords", [])))

            query_text = self.build_query(
                cluster["label"],
                cluster["summary"],
                cluster_keywords
            )

            query_embedding = self.embedder.embed(query_text, progress_bar=False)

            # A — Hybrid retrieval
            hybrid = self.hybrid_retrieve(query_embedding, query_text)
            self.progress.update(label="Hybrid retrieval")

            # B — Keyword re-ranking
            reranked = self.rerank_keywords(hybrid, query_text)
            self.progress.update(label="Keyword re-ranking")

            # C — Deduplication
            deduped = self.dedupe_chunks(reranked)
            self.progress.update(label="Deduplication")

            # D — Final selection + diagnostics
            final = deduped[:self.final_k]

            retrieved_chunks = []
            for r in final:
                retrieved_chunks.append({
                    "chunk_id": r["chunk"]["chunk_id"],
                    "text": r["chunk"]["text"],
                    "metadata": r["chunk"].get("metadata", {}),
                    "final_score": r["final_score"],
                })

            diagnostics = self.format_diagnostics(final)
            self.progress.update(label="Diagnostics")

            results.append({
                "cluster_id": cid,
                "cluster_label": cluster["label"],
                "cluster_keywords": cluster_keywords,  # NEW
                "retrieved_count": len(retrieved_chunks),
                "retrieved_chunks": retrieved_chunks,
                "diagnostics": diagnostics,
            })

        self.save_json(results, output_json_path)
        return results
