import os
import json
from typing import Dict, Any, List

from Utilities import Simple_Progress_Bar


class Clusters_Baseline_Maker:
    """
    Step 1 baseline extractor:
    - Label
    - High-level summary
    - Keywords
    - Candidate entities
    - Candidate processes

    Produces one JSON file per cluster:
        base_<cluster_id>_knowledge.json
    """

    def __init__(
        self,
        input_json_path: str,
        output_folder: str,
        llm: any,
        vdb: any,
        top_n_chunks: int = 10,
    ):
        """
        input_json_path: path to Enriched_Clusters.json
        output_folder: folder where baseline JSONs will be written
        llm: an object implementing LLMBackend
        vdb: an object implementing VectorDBBackend
        top_n_chunks: number of matching chunks to send to LLM
        """
        self.input_json_path = input_json_path
        self.output_folder = output_folder
        self.llm = llm
        self.vdb = vdb
        self.top_n_chunks = top_n_chunks

        os.makedirs(self.output_folder, exist_ok=True)

        with open(self.input_json_path, "r", encoding="utf-8") as f:
            self.root = json.load(f)

    # ------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------
    def traverse(self):
        """
        Depth-first traversal of the cluster tree.
        Yields each cluster node.
        """
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node

            children = node.get("children", {})
            if children and "clusters" in children:
                for child in children["clusters"]:
                    stack.append(child)

    # ------------------------------------------------------------
    # Fetch top-N matching chunks from VDB
    # ------------------------------------------------------------
    def get_top_matching_chunks(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts:
            { "chunk_id": ..., "document": ..., "metadata": ... }
        """
        matching_ids = node.get("matching_chunks", [])[: self.top_n_chunks]

        results = []
        for cid in matching_ids:
            entry = self.vdb.get_by_id(cid)
            if entry:
                results.append(entry)
        return results

    # ------------------------------------------------------------
    # Build LLM prompt
    # ------------------------------------------------------------
    def build_prompt(self, cluster_id: str, chunks: List[Dict[str, Any]]) -> str:
        chunk_texts = [c["document"] for c in chunks]

        return f"""
    You are extracting baseline ontology elements for cluster {cluster_id}.

    IMPORTANT:
    - Detect the dominant language of the input text.
    - Produce ALL output fields in that same language.
    - Do NOT translate the content.
    - Preserve terminology exactly as used in the input chunks.

    Input chunks (semantic core of the cluster):
    {json.dumps(chunk_texts, indent=2, ensure_ascii=False)}

    Extract the following fields:

    1. Label (short, 1–4 words)
    2. High-level summary (3–6 sentences)
    3. Keywords (10–20)
    4. Candidate entities (list)
    5. Candidate processes (list)

    Return ONLY valid JSON with these keys:
    label, summary, keywords, entities, processes.
    """




    # ------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        response = self.llm.complete(prompt)

        try:
            return json.loads(response)
        except Exception:
            return {
                "label": "ERROR",
                "summary": "LLM returned invalid JSON.",
                "keywords": [],
                "entities": [],
                "processes": [],
                "raw_response": response,
            }

    # ------------------------------------------------------------
    # Write output JSON (UTF‑8 with German characters preserved)
    # ------------------------------------------------------------
    def write_output(self, cluster_id: str, data: Dict[str, Any]):
        filename = f"base_{cluster_id}_knowledge.json"
        path = os.path.join(self.output_folder, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------
    def run(self):
        """
        Runs baseline extraction for all clusters.
        """
        all_nodes = list(self.traverse())
        progress = Simple_Progress_Bar(total=len(all_nodes), enabled=True)

        for node in all_nodes:
            cid = node.get("cluster_id", "unknown")

            # 1. Fetch top matching chunks from VDB
            chunks = self.get_top_matching_chunks(node)

            # 2. Build prompt
            prompt = self.build_prompt(cid, chunks)

            # 3. Call LLM
            baseline = self.call_llm(prompt)

            # 4. Write output
            self.write_output(cid, baseline)

            # 5. Update progress bar
            progress.update(label=f"Cluster {cid}")

        print("\nBaseline extraction completed.")
