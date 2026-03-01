import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional, List
from Utilities import Simple_Progress_Bar


class Cluster_Info_Extractor:
    """
    Extracts information for each cluster using an LLM and Vector DB.

    - Two prompt templates:
        * leaf_prompt_template (must contain {text})
        * internal_prompt_template (must contain {json_list}) or None
    - If internal_prompt_template is None → only leaf clusters are processed.
    - Only clusters under branch_id are processed (if provided).
    - Each cluster output is written inside: output_folder / cluster_id / info_type.json
    """

    def __init__(
        self,
        llm: Any,
        vectordb: Any,
        leaf_prompt_template: str,
        internal_prompt_template: Optional[str],
        info_type: str,
        output_folder: str,
        store_cache: bool = True,
        cache_path: str = "cluster_info_cache.json",
        retry_attempts: int = 3,
        verbose: bool = False,
        show_progress_bar: bool = False,
        max_concurrent_llm_calls: int = 10,
        log_prompts: bool = False,
        branch_id: Optional[str] = None,   # NEW
    ):
        self.llm = llm
        self.vectordb = vectordb
        self.leaf_prompt_template = leaf_prompt_template
        self.internal_prompt_template = internal_prompt_template
        self.info_type = info_type
        self.output_folder = output_folder

        self.store_cache = store_cache
        self.cache_path = cache_path
        self.retry_attempts = retry_attempts
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.log_prompts = log_prompts
        self.branch_id = branch_id  # NEW

        self._semaphore = asyncio.Semaphore(max_concurrent_llm_calls)

        if store_cache and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        os.makedirs(output_folder, exist_ok=True)

        self.progress = None
        self._branch_root = None  # NEW

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    def _log_prompt(self, cid: str, prompt: str):
        if not self.log_prompts:
            return
        folder = os.path.join(self.output_folder, cid)
        os.makedirs(folder, exist_ok=True)
        log_path = os.path.join(folder, "PROMPT.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(prompt)

    # -------------------------------------------------------------------------
    # Count clusters (only those under branch_id if provided)
    # -------------------------------------------------------------------------
    def _count_clusters(self, node: Dict[str, Any]) -> int:
        if self.branch_id and not self._branch_root:
            # Find branch root first
            self._branch_root = self._find_branch(node, self.branch_id)
            if not self._branch_root:
                raise ValueError(f"branch_id '{self.branch_id}' not found in hierarchy.")
            return self._count_clusters(self._branch_root)

        count = len(node["clusters"])
        for c in node["clusters"]:
            if c["children"] is not None:
                count += self._count_clusters(c["children"])
        return count

    # -------------------------------------------------------------------------
    # Find branch root
    # -------------------------------------------------------------------------
    def _find_branch(self, node: Dict[str, Any], target: str) -> Optional[Dict]:
        for c in node["clusters"]:
            if c["cluster_id"] == target:
                return c  # return the REAL node, not a wrapper
            if c["children"] is not None:
                found = self._find_branch(c["children"], target)
                if found:
                    return found
        return None

    # -------------------------------------------------------------------------
    # Clean JSON
    # -------------------------------------------------------------------------
    def _clean_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].strip()
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0].strip()
        return text

    # -------------------------------------------------------------------------
    # Extract first JSON object
    # -------------------------------------------------------------------------
    def _extract_first_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue

            if ch == "\\":
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]

        return None

    # -------------------------------------------------------------------------
    # Vector DB
    # -------------------------------------------------------------------------
    def _get_record_by_id(self, chunk_id: str):
        return self.vectordb.get_by_id(chunk_id)

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------
    def process_hierarchy_file(self, input_json_path: str):
        try:
            return asyncio.run(self._aprocess_hierarchy_file(input_json_path))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._aprocess_hierarchy_file(input_json_path))

    # -------------------------------------------------------------------------
    # Async main
    # -------------------------------------------------------------------------
    async def _aprocess_hierarchy_file(self, input_json_path: str):
        with open(input_json_path, "r", encoding="utf-8") as f:
            hierarchy = json.load(f)

        # If branch_id is set, restrict hierarchy BEFORE counting
        if self.branch_id:
            self._branch_root = self._find_branch(hierarchy, self.branch_id)
            if not self._branch_root:
                raise ValueError(f"branch_id '{self.branch_id}' not found.")
            hierarchy = {"clusters": [self._branch_root]}

        # Now count only the selected branch
        total = self._count_clusters(hierarchy)
        self.progress = Simple_Progress_Bar(total, enabled=self.show_progress_bar)

        await self._aprocess_node(hierarchy)

        if self.store_cache:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)

        return hierarchy

    # -------------------------------------------------------------------------
    # Sequential recursive traversal
    # -------------------------------------------------------------------------
    async def _aprocess_node(self, node: Dict[str, Any]):
        for cluster in node["clusters"]:
            if cluster["children"] is not None:
                await self._aprocess_node(cluster["children"])

        for cluster in node["clusters"]:
            await self._aprocess_cluster(cluster)

    # -------------------------------------------------------------------------
    # Process a single cluster
    # -------------------------------------------------------------------------
    async def _aprocess_cluster(self, cluster: Dict[str, Any]):
        cid = cluster["cluster_id"]

        # Skip internal clusters if no internal prompt
        if cluster["children"] is not None and self.internal_prompt_template is None:
            self.progress.update(label=f"Skipping internal {cid}")
            return

        # Cache skip
        if self.store_cache and cid in self.cache:
            self.progress.update(label=f"Skipping {cid}")
            return

        # ---------------------------------------------------------------------
        # LEAF CLUSTER
        # ---------------------------------------------------------------------
        if cluster["children"] is None:
            chunk_ids = cluster.get("ids") or cluster.get("chunk_ids") or []
            texts = []
            for chunk_id in chunk_ids:
                rec = self._get_record_by_id(chunk_id)
                texts.append(rec.get("document", "") if rec else "")
            combined_text = "\n\n".join(texts)

            prompt = self.leaf_prompt_template.replace("{text}", combined_text)

        # ---------------------------------------------------------------------
        # INTERNAL CLUSTER
        # ---------------------------------------------------------------------
        else:
            child_jsons = []
            for child in cluster["children"]["clusters"]:
                child_id = child["cluster_id"]
                path = os.path.join(self.output_folder, child_id, f"{self.info_type}.json")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        child_jsons.append(json.load(f))

            json_list_str = json.dumps(child_jsons, indent=2, ensure_ascii=False)
            prompt = self.internal_prompt_template.replace("{json_list}", json_list_str)

        # ---------------------------------------------------------------------
        # LOG PROMPT IF ENABLED
        # ---------------------------------------------------------------------
        self._log_prompt(cid, prompt)

        # ---------------------------------------------------------------------
        # LLM CALL WITH RETRIES
        # ---------------------------------------------------------------------
        data = None
        for attempt in range(self.retry_attempts):
            try:
                async with self._semaphore:
                    response = await self.llm.acomplete(prompt)

                clean = self._clean_json(response)
                json_candidate = self._extract_first_json_object(clean)
                if not json_candidate:
                    raise ValueError("No JSON object extracted")

                data = json.loads(json_candidate)
                break

            except Exception as e:
                print(f"ERROR in cluster {cid}: {e}")
                if attempt == self.retry_attempts - 1:
                    self.progress.update(label=f"FAILED {cid}")
                    return
                await asyncio.sleep(2 ** attempt)

        # ---------------------------------------------------------------------
        # SAVE JSON FILE (inside cluster_id folder)
        # ---------------------------------------------------------------------
        cluster_folder = os.path.join(self.output_folder, cid)
        os.makedirs(cluster_folder, exist_ok=True)

        out_path = os.path.join(cluster_folder, f"{self.info_type}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if self.store_cache:
            self.cache[cid] = True

        # ---------------------------------------------------------------------
        # UPDATE PROGRESS BAR AFTER SUCCESS
        # ---------------------------------------------------------------------
        self.progress.update(label=f"Done {cid}")
