import json
from typing import List, Dict, Any
from Utilities import Simple_Progress_Bar


class Ontology_Chunks_Retriever:
    """
    Retrieves top-N relevant raw chunks for each cluster.
    Includes:
    - automatic hierarchy flattening
    - automatic/configurable language selection
    - saving flattened clusters for debugging
    - shared progress bar
    """

    def __init__(
        self,
        vector_db,
        embedder,
        top_n=30,
        verbose=False,
        progress_bar=False,
        language: str = None,   # NEW: optional language override
    ):
        self.vector_db = vector_db
        self.embedder = embedder
        self.top_n = top_n

        self.progress_bar_enabled = progress_bar
        self.verbose = verbose and not progress_bar

        self.language_override = language  # NEW

        self.progress = None

    # ---------------------------------------------------------
    # Logging helper
    # ---------------------------------------------------------
    def log(self, msg: str):
        if self.verbose:
            print(msg)

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
        """
        Rules:
        1. If user passed a language → use it.
        2. Else detect available languages from the first cluster.
        3. If only one language → use it.
        4. If multiple → use the first one.
        """
        if self.language_override:
            self.log(f"Using user-specified language: {self.language_override}")
            return self.language_override

        # Find first cluster with multilang
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

        # -----------------------------------------------------
        # Step 1: Flatten if needed
        # -----------------------------------------------------
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
        # Step 2: Progress bar
        # -----------------------------------------------------
        total = len(clusters)
        self.progress = Simple_Progress_Bar(total, enabled=self.progress_bar_enabled)

        results = []

        # -----------------------------------------------------
        # Step 3: Retrieval
        # -----------------------------------------------------
        for cluster in clusters:
            cid = cluster["cluster_id"]
            self.log(f"Processing cluster {cid}")

            query = self.build_query(
                cluster["label"],
                cluster["summary"],
                cluster.get("keywords", [])
            )

            embedding = self.embedder.embed(query, progress_bar=False)
            retrieved = self.vector_db.search(embedding, top_n=self.top_n)

            results.append({
                "cluster_id": cid,
                "cluster_label": cluster["label"],
                "retrieved_count": len(retrieved),
                "retrieved_chunks": retrieved
            })

            self.progress.update()

        # -----------------------------------------------------
        # Step 4: Save output
        # -----------------------------------------------------
        self.log(f"Saving retrieved chunks to {output_json_path}")
        self.save_json(results, output_json_path)

        self.log("Retrieval completed.")
        return results
