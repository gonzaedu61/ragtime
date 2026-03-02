import asyncio
import json
import os
import re
from typing import Any, Dict, Optional, List
from Utilities import Simple_Progress_Bar


class Cluster_Info_Extender:
    """
    Extends cluster information using an LLM, Vector DB, and an embedder.

    Features:
    - Reads input JSON per cluster (info_type_input)
    - Leaf clusters: prompt = original chunks + input JSON (+ optional semantic chunks)
    - Internal clusters: prompt = input JSON (+ optional semantic chunks)
    - Optional semantic retrieval using embedder + vector DB
    - Skips leaf clusters if leaf_prompt_template is None
    - Skips internal clusters if internal_prompt_template is None
    - Repairs malformed JSON before json.loads()
    """

    def __init__(
        self,
        llm: Any,
        vectordb: Any,
        embedder: Any,   # <-- NEW
        leaf_prompt_template: Optional[str],
        internal_prompt_template: Optional[str],
        info_type: str,
        info_type_input: str,
        output_folder: str,
        retrieve_semantic_chunks: bool = False,
        top_number_of_chunks: int = 10,
        retry_attempts: int = 3,
        verbose: bool = False,
        show_progress_bar: bool = False,
        max_concurrent_llm_calls: int = 10,
        log_prompts: bool = False,
        branch_id: Optional[str] = None,
    ):
        self.llm = llm
        self.vectordb = vectordb
        self.embedder = embedder  # <-- NEW

        self.leaf_prompt_template = leaf_prompt_template
        self.internal_prompt_template = internal_prompt_template
        self.info_type = info_type
        self.info_type_input = info_type_input
        self.output_folder = output_folder

        self.retrieve_semantic_chunks = retrieve_semantic_chunks
        self.top_number_of_chunks = top_number_of_chunks

        self.retry_attempts = retry_attempts
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.log_prompts = log_prompts
        self.branch_id = branch_id

        self._semaphore = asyncio.Semaphore(max_concurrent_llm_calls)
        os.makedirs(output_folder, exist_ok=True)

        self.progress = None
        self._branch_root = None

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    def _log_prompt(self, cid: str, prompt: str):
        if not self.log_prompts:
            return
        folder = os.path.join(self.output_folder, cid)
        os.makedirs(folder, exist_ok=True)
        log_path = os.path.join(folder, f"{cid}_{self.info_type}_PROMPT.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(prompt)

    # -------------------------------------------------------------------------
    # JSON cleaning helpers
    # -------------------------------------------------------------------------
    def _clean_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].strip()
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0].strip()
        return text

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
    # JSON repair
    # -------------------------------------------------------------------------
    def _repair_json(self, text: str) -> str:
        text = re.sub(r",\s*([}\]])", r"\1", text)
        text = text.replace("None", "null").replace("True", "true").replace("False", "false")
        text = re.sub(r"//.*?\n", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

        if '"' not in text:
            text = text.replace("'", '"')

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

        return text

    # -------------------------------------------------------------------------
    # Vector DB helpers
    # -------------------------------------------------------------------------
    def _get_record_by_id(self, chunk_id: str):
        return self.vectordb.get_by_id(chunk_id)

    def _semantic_search(self, text: str, k: int) -> List[str]:
        if k <= 0:
            return []

        embedding_list = self.embedder.embed([text])
        embedding = embedding_list[0]

        results = self.vectordb.search(embedding, top_n=k)

        ids = []
        for r in results:
            cid = r.get("chunk_id") or r.get("id")
            if cid:
                ids.append(cid)

        return ids

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

        if self.branch_id:
            self._branch_root = self._find_branch(hierarchy, self.branch_id)
            if not self._branch_root:
                raise ValueError(f"branch_id '{self.branch_id}' not found.")
            hierarchy = {"clusters": [self._branch_root]}

        total = self._count_clusters(hierarchy)
        self.progress = Simple_Progress_Bar(total, enabled=self.show_progress_bar)

        await self._aprocess_node(hierarchy)
        return hierarchy

    # -------------------------------------------------------------------------
    # Count clusters
    # -------------------------------------------------------------------------
    def _count_clusters(self, node: Dict[str, Any]) -> int:
        count = len(node["clusters"])
        for c in node["clusters"]:
            if c["children"] is not None:
                count += self._count_clusters(c["children"])
        return count

    # -------------------------------------------------------------------------
    # Find branch
    # -------------------------------------------------------------------------
    def _find_branch(self, node: Dict[str, Any], target: str) -> Optional[Dict]:
        for c in node["clusters"]:
            if c["cluster_id"] == target:
                return c
            if c["children"] is not None:
                found = self._find_branch(c["children"], target)
                if found:
                    return found
        return None

    # -------------------------------------------------------------------------
    # Recursive traversal
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

        cluster_folder = os.path.join(self.output_folder, cid)
        out_path = os.path.join(cluster_folder, f"{cid}_{self.info_type}.json")

        if os.path.exists(out_path):
            self.progress.update(label=f"Skipping {cid} (exists)")
            return

        if cluster["children"] is None and self.leaf_prompt_template is None:
            #self.progress.update(label=f"Skipping leaf {cid} (no leaf prompt)")
            return

        if cluster["children"] is not None and self.internal_prompt_template is None:
            return

        input_json_path = os.path.join(cluster_folder, f"{cid}_{self.info_type_input}.json")
        if not os.path.exists(input_json_path):
            self.progress.update(label=f"Missing input JSON for {cid}")
            return

        with open(input_json_path, "r", encoding="utf-8") as f:
            input_json_data = json.load(f)

        input_json_str = json.dumps(input_json_data, indent=2, ensure_ascii=False)

        # LEAF CLUSTER
        if cluster["children"] is None:
            chunk_ids = cluster.get("ids") or cluster.get("chunk_ids") or []
            texts = []

            for chunk_id in chunk_ids:
                rec = self._get_record_by_id(chunk_id)
                texts.append(rec.get("document", "") if rec else "")

            combined_text = "\n\n".join(texts)

            extra_texts = []
            if self.retrieve_semantic_chunks:
                missing = max(0, self.top_number_of_chunks - len(chunk_ids))
                if missing > 0:
                    extra_ids = self._semantic_search(input_json_str, missing)
                    for eid in extra_ids:
                        rec = self._get_record_by_id(eid)
                        extra_texts.append(rec.get("document", "") if rec else "")

            full_text = combined_text + "\n\n" + "\n\n".join(extra_texts)

            prompt = (
                self.leaf_prompt_template
                .replace("{text}", full_text)
                .replace("{input_json}", input_json_str)
            )

        # INTERNAL CLUSTER
        else:
            extra_texts = []
            if self.retrieve_semantic_chunks:
                extra_ids = self._semantic_search(input_json_str, self.top_number_of_chunks)
                for eid in extra_ids:
                    rec = self._get_record_by_id(eid)
                    extra_texts.append(rec.get("document", "") if rec else "")

            extra_text_block = "\n\n".join(extra_texts)

            prompt = (
                self.internal_prompt_template
                .replace("{input_json}", input_json_str)
                .replace("{extra_chunks}", extra_text_block)
            )

        self._log_prompt(cid, prompt)

        # LLM CALL
        data = None
        for attempt in range(self.retry_attempts):
            try:
                async with self._semaphore:
                    response = await self.llm.acomplete(prompt)

                clean = self._clean_json(response)
                extracted = self._extract_first_json_object(clean)

                if not extracted:
                    raise ValueError("No JSON object extracted")

                repaired = self._repair_json(extracted)

                data = json.loads(repaired)
                break

            except Exception as e:
                print(f"ERROR in cluster {cid}: {e}")
                if attempt == self.retry_attempts - 1:
                    self.progress.update(label=f"FAILED {cid}")
                    return
                await asyncio.sleep(2 ** attempt)

        os.makedirs(cluster_folder, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.progress.update(label=f"Done {cid}")
