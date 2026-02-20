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
    - hybrid retrieval (dense + sparse)
    - keyword re-ranking
    - chunk deduplication
    - scoring diagnostics
    - per-cluster progress visibility
    - automatic hierarchy flattening
    - automatic/configurable language selection
    - saving flattened clusters for debugging
    """

    def __init__(
        self,
        vector_db,
        embedder,
        top_n=30,
        verbose=False,
        progress_bar=False,
        language: str = None,
        hybrid_alpha=0.5,          # NEW
        keyword_weight=0.2,        # NEW
    ):
        self.vector_db = vector_db
        self.embedder = embedder
        self.top_n = top_n

        self.progress_bar_enabled = progress_bar
        self.verbose = verbose and not progress_bar

        self.language_override = language
        self.hybrid_alpha = hybrid_alpha
        self.keyword_weight = keyword_weight

        self.progress = None

        # -----------------------------------------------------
        # NEW: Build sparse index (BM25)
        # -----------------------------------------------------
        self._init_sparse_index()

    # ---------------------------------------------------------
    # Logging helper
    # ---------------------------------------------------------
    def log(self, msg: str):
        if self.verbose:
            print(msg)

    # ---------------------------------------------------------
    # NEW: Build BM25 sparse index
    # ---------------------------------------------------------
    def _init_sparse_index(self):
        try:
            self.all_chunks = self.vector_db.all_chunks()
        except Exception:
            self.all_chunks = []

        if not self.all_chunks:
            self.log("WARNING: vector_db.all_chunks() returned no data. Sparse retrieval disabled.")
            self.bm25 = None
            return

        tokenized = [c["text"].split() for c in self.all_chunks]
        self.bm25 = BM25Okapi(tokenized)
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
        if len(languages) == 1:
            lang = languages[0]
            self.log(f"Detected single language: {lang}")
            return lang

        lang = languages[0]
        self.log(f"Detected multiple languages {languages}. Using first: {lang}")
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
    # NEW: Keyword score
    # ---------------------------------------------------------
    def keyword_score(self, query: str, text: str) -> int:
        q_words = set(query.lower().split())
        t_words = set(text.lower().split())
        return len(q_words & t_words)

    # ---------------------------------------------------------
    # NEW: Deduplication helper
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
    # NEW: Hybrid retrieval (dense + sparse)
    # ---------------------------------------------------------
    def hybrid_retrieve(self, query_embedding, query_text, k=30):
        dense_results = self.vector_db.search(query_embedding, top_n=k)
        dense_map = {c["chunk_id"]: c["score"] for c in dense_results}

        if self.bm25:
            sparse_scores = self.bm25.get_scores(query_text.split())
        else:
            sparse_scores = np.zeros(len(self.all_chunks))

        sparse_norm = sparse_scores / (np.linalg.norm(sparse_scores) + 1e-6)

        hybrid = []
        for idx, chunk in enumerate(self.all_chunks):
            cid = chunk["chunk_id"]
            dense = dense_map.get(cid, 0.0)
            sparse = sparse_norm[idx]

            hybrid_score = (
                self.hybrid_alpha * dense +
                (1 - self.hybrid_alpha) * sparse
            )

            hybrid.append({
                "chunk": chunk,
                "dense_score": float(dense),
                "sparse_score": float(sparse),
                "hybrid_score": float(hybrid_score),
            })

        hybrid_sorted = sorted(hybrid, key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_sorted[:k]

    # ---------------------------------------------------------
    # NEW: Keyword re-ranking
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
    # NEW: Diagnostics formatter
    # ---------------------------------------------------------
    def format_diagnostics(self, results):
        diagnostics = []
        for r in results:
            diagnostics.append({
                "chunk_id": r["chunk"]["chunk_id"],
                "text_preview": r["chunk"]["text"][:120] + "...",
                "final_score": r["final_score"],
                "hybrid_score": r["hybrid_score"],
                "dense_score": r["dense_score"],
                "sparse_score": r["sparse_score"],
                "keyword_score": r["keyword_score"],
            })
        return diagnostics

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
            self.log("Detected hierarchy. Determining language...")
            lang = self.detect_language(data)

            self.log(f"Flattening clusters using language: {lang}")
            clusters = self.flatten_clusters(data, lang=lang)

            self.log(f"Flattened {len(clusters)} clusters. Saving to {flattened_debug_path}")
            self.save_json(clusters, flattened_debug_path)
        else:
            self.log("Detected flat cluster list.")
            clusters = data

        # -----------------------------------------------------
        # NEW: Progress bar with 4 sub-steps per cluster
        # -----------------------------------------------------
        total_steps = len(clusters) * 4
        self.progress = Simple_Progress_Bar(total_steps, enabled=self.progress_bar_enabled)

        results = []

        # Step 3: Retrieval
        for cluster in clusters:
            cid = cluster["cluster_id"]
            self.log(f"Processing cluster {cid}")

            query_text = self.build_query(
                cluster["label"],
                cluster["summary"],
                cluster.get("keywords", [])
            )

            query_embedding = self.embedder.embed(query_text, progress_bar=False)

            # -------------------------------------------------
            # NEW: Step A — Hybrid retrieval
            # -------------------------------------------------
            hybrid = self.hybrid_retrieve(query_embedding, query_text, k=self.top_n)
            self.progress.update(label="Hybrid retrieval")

            # -------------------------------------------------
            # NEW: Step B — Keyword re-ranking
            # -------------------------------------------------
            reranked = self.rerank_keywords(hybrid, query_text)
            self.progress.update(label="Keyword re-ranking")

            # -------------------------------------------------
            # NEW: Step C — Deduplication
            # -------------------------------------------------
            deduped = self.dedupe_chunks(reranked)
            self.progress.update(label="Deduplication")

            # -------------------------------------------------
            # NEW: Step D — Diagnostics
            # -------------------------------------------------
            final = deduped[:self.top_n]
            diagnostics = self.format_diagnostics(final)
            self.progress.update(label="Diagnostics")

            results.append({
                "cluster_id": cid,
                "cluster_label": cluster["label"],
                "retrieved_count": len(final),
                "retrieved_chunks": [r["chunk"] for r in final],
                "diagnostics": diagnostics,
            })

        # Step 4: Save output
        self.log(f"Saving retrieved chunks to {output_json_path}")
        self.save_json(results, output_json_path)

        self.log("Retrieval completed.")
        return results
