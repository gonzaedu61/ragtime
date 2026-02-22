import os
import json
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from Utilities import Simple_Progress_Bar


class Clusters_Baseline_Maker:
    """
    Baseline extractor with optional multi-threaded LLM calls.
    """

    def __init__(
        self,
        input_json_path: str,
        output_folder: str,
        llm: any,
        vdb: any,
        top_n_chunks: int = 10,
        num_threads: int = 1,          # NEW: 1 = single-thread mode
        max_retries: int = 3,          # NEW: retry mechanism
        retry_delay: float = 2.0,      # NEW: seconds between retries
    ):
        self.input_json_path = input_json_path
        self.output_folder = output_folder
        self.llm = llm
        self.vdb = vdb
        self.top_n_chunks = top_n_chunks
        self.num_threads = max(1, num_threads)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        os.makedirs(self.output_folder, exist_ok=True)

        with open(self.input_json_path, "r", encoding="utf-8") as f:
            self.root = json.load(f)

    # ------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------
    def traverse(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node

            children = node.get("children", {})
            if children and "clusters" in children:
                for child in children["clusters"]:
                    stack.append(child)

    # ------------------------------------------------------------
    # Fetch top-N matching chunks
    # ------------------------------------------------------------
    def get_top_matching_chunks(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    # LLM call with retry mechanism
    # ------------------------------------------------------------
    def call_llm_with_retry(self, prompt: str) -> Dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm.complete(prompt)
                return json.loads(response)
            except Exception:
                if attempt == self.max_retries:
                    return {
                        "label": "ERROR",
                        "summary": f"LLM failed after {self.max_retries} attempts.",
                        "keywords": [],
                        "entities": [],
                        "processes": [],
                    }
                time.sleep(self.retry_delay)

    # ------------------------------------------------------------
    # Write output JSON
    # ------------------------------------------------------------
    def write_output(self, cluster_id: str, data: Dict[str, Any]):
        filename = f"base_{cluster_id}_knowledge.json"
        path = os.path.join(self.output_folder, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------
    # Process a single cluster (thread-safe)
    # ------------------------------------------------------------
    def process_single_cluster(self, node: Dict[str, Any]) -> str:
        cid = node.get("cluster_id", "unknown")
        output_path = os.path.join(self.output_folder, f"base_{cid}_knowledge.json")

        # Skip if already exists
        if os.path.exists(output_path):
            return f"{cid} (skipped)"

        chunks = self.get_top_matching_chunks(node)
        prompt = self.build_prompt(cid, chunks)
        baseline = self.call_llm_with_retry(prompt)
        self.write_output(cid, baseline)

        return f"{cid} (done)"

    # ------------------------------------------------------------
    # Main execution (single-thread or multi-thread)
    # ------------------------------------------------------------
    def run(self):
        all_nodes = list(self.traverse())
        total_clusters = len(all_nodes)

        # ------------------------------------------------------------
        # PRE-RUN SCAN OF OUTPUT FOLDER
        # ------------------------------------------------------------
        existing_ids = []
        missing_ids = []

        for node in all_nodes:
            cid = node.get("cluster_id", "unknown")
            output_path = os.path.join(self.output_folder, f"base_{cid}_knowledge.json")
            if os.path.exists(output_path):
                existing_ids.append(cid)
            else:
                missing_ids.append(cid)

        print("\n=== Baseline Maker Status Check ===")
        print(f"Total clusters ..: {total_clusters}")
        print(f"Existing clusters: {len(existing_ids)}")
        print(f"Pending clusters.: {len(missing_ids)}")

        if len(existing_ids) == total_clusters:
            print("\nAll cluster baseline files already exist. Nothing to do.")
            return

        '''
        if len(existing_ids) > 0:
            print("\nSome clusters are already processed. Only missing ones will be computed.")
            print(f"Missing cluster IDs: {missing_ids}")
        '''

        print("\nStarting LLM processing...\n")

        # ------------------------------------------------------------
        # PROGRESS BAR
        # ------------------------------------------------------------
        progress = Simple_Progress_Bar(total=total_clusters, enabled=True)

        # ------------------------------------------------------------
        # SINGLE-THREAD MODE
        # ------------------------------------------------------------
        if self.num_threads == 1:
            for node in all_nodes:
                cid = node.get("cluster_id", "unknown")

                # Skip existing
                if cid in existing_ids:
                    progress.update(label=f"{cid} (skipped)")
                    continue

                result = self.process_single_cluster(node)
                progress.update(label=result)

            print("\nBaseline extraction completed.")
        else:
            # ------------------------------------------------------------
            # MULTI-THREAD MODE
            # ------------------------------------------------------------
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self.process_single_cluster, node): node
                    for node in all_nodes
                }

                for future in as_completed(futures):
                    node = futures[future]
                    cid = node.get("cluster_id", "unknown")

                    # Skip existing
                    if cid in existing_ids:
                        progress.update(label=f"{cid} (skipped)")
                        continue

                    try:
                        result = future.result()
                    except Exception:
                        result = f"{cid} (error)"

                    progress.update(label=result)

            print("\nParallel baseline extraction completed.")

        # ------------------------------------------------------------
        # POST-RUN SCAN
        # ------------------------------------------------------------
        final_existing = []
        final_missing = []

        for node in all_nodes:
            cid = node.get("cluster_id", "unknown")
            output_path = os.path.join(self.output_folder, f"base_{cid}_knowledge.json")
            if os.path.exists(output_path):
                final_existing.append(cid)
            else:
                final_missing.append(cid)

        print("\n=== Final Output Folder Status ===")
        print(f"Total clusters:          {total_clusters}")
        print(f"Files generated:         {len(final_existing)}")
        print(f"Files still missing:     {len(final_missing)}")

        # Always generate missing_clusters.txt if anything is missing
        if final_missing:
            print("\nWARNING: Some clusters are still missing.")
            print(f"Missing cluster IDs: {final_missing}")

            missing_file = os.path.join(self.output_folder, "missing_clusters.txt")
            with open(missing_file, "w", encoding="utf-8") as f:
                for cid in final_missing:
                    f.write(cid + "\n")

            print(f"Missing cluster list written to: {missing_file}")
        else:
            print("\nAll cluster baseline files are present. Complete set generated.")
