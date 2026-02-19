import asyncio
import json
import os
from typing import Any, Dict, Optional


class Topic_Hierarchy_Labeler:
    """
    Bottom-up LLM-based labeling and summarization for a topic hierarchy.

    llm: object with .acomplete(prompt: str) -> str (async)
         optional .complete(prompt: str) -> str (sync wrapper)
    vectordb: object with .get_by_id(id: str) -> {"document": str, "metadata": dict}
    store_summaries: if True, summaries are stored in the hierarchy JSON
    cache_path: JSON file used to store/retrieve cached summaries and labels
    verbose: if True, prints progress information

    This version is async-first and parallelizes LLM calls per node using asyncio.gather.
    """

    DEFAULT_COMBINED_PROMPT = (
        "You are analyzing a set of text chunks that belong to the same topic.\n"
        "Your tasks:\n"
        "1. Write a concise summary (4–6 sentences) capturing the shared theme.\n"
        "2. Propose a short, human-readable label (max 6 words) that captures the essence.\n"
        "3. Extract 5–10 high-value keywords that represent the topic.\n\n"
        "TEXTS:\n{text}\n\n"
        "LANGUAGE REQUIREMENTS:\n"
        "- Detect the dominant language of the provided TEXTS.\n"
        "- Write the summary, label, and keywords in that same language.\n"
        "- Only use English if the input language cannot be determined.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Respond ONLY with valid JSON.\n"
        "- Do NOT include any Markdown formatting.\n"
        "- Do NOT include code fences (no ```json or ```).\n"
        "- Do NOT include explanations, comments, or natural language outside JSON.\n"
        "- The FIRST character of your response must be '{{'.\n"
        "- The LAST character of your response must be '}}'.\n\n"
        "Return JSON in exactly this structure:\n"
        "{{\n"
        "  \"summary\": \"...\",\n"
        "  \"label\": \"...\",\n"
        "  \"keywords\": [\"...\", \"...\"]\n"
        "}}\n"
    )

    def __init__(
        self,
        llm: Any,
        vectordb: Any,
        store_summaries: bool = True,
        cache_path: str = "hierarchy_label_cache.json",
        combined_prompt: Optional[str] = None,
        verbose: bool = False,
        max_concurrent_llm_calls: int = 10,
    ):
        self.llm = llm
        self.vectordb = vectordb
        self.store_summaries = store_summaries
        self.cache_path = cache_path
        self.combined_prompt = combined_prompt or self.DEFAULT_COMBINED_PROMPT
        self.verbose = verbose

        # Progress tracking
        self._total_clusters: int = 0
        self._processed_clusters: int = 0

        # For incremental flushing
        self._current_hierarchy: Optional[Dict[str, Any]] = None
        self._current_output_path: Optional[str] = None

        # Concurrency controls
        self._semaphore = asyncio.Semaphore(max_concurrent_llm_calls)
        self._flush_lock = asyncio.Lock()

        # Load cache if exists
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.cache: Dict[str, Dict[str, Any]] = json.load(f)
        else:
            self.cache = {}

    # ---------------------------------------------------------
    # Helper: verbose logger
    # ---------------------------------------------------------
    def _log(self, message: str):
        if self.verbose:
            print(message)

    # ---------------------------------------------------------
    # Helper: count clusters for progress
    # ---------------------------------------------------------
    def _count_clusters(self, node: Dict[str, Any]) -> int:
        count = len(node["clusters"])
        for cluster in node["clusters"]:
            if cluster["children"] is not None:
                count += self._count_clusters(cluster["children"])
        return count

    # ---------------------------------------------------------
    # Helper: clean JSON (remove code fences)
    # ---------------------------------------------------------
    def _clean_json(self, text: str) -> str:
        """
        Removes Markdown code fences and trims whitespace so json.loads() can parse it.
        """
        text = text.strip()

        if text.startswith("```"):
            parts = text.split("\n", 1)
            if len(parts) > 1:
                text = parts[1].strip()
            else:
                text = ""

        if text.endswith("```"):
            parts = text.rsplit("\n", 1)
            if len(parts) > 1:
                text = parts[0].strip()
            else:
                text = ""

        return text.strip()

    # ---------------------------------------------------------
    # Helper: flush progress (cache + hierarchy)
    # ---------------------------------------------------------
    async def _aflush_progress(self):
        if self._current_hierarchy is None or self._current_output_path is None:
            return

        async with self._flush_lock:
            # Save cache
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)

            # Save hierarchy
            with open(self._current_output_path, "w", encoding="utf-8") as f:
                json.dump(self._current_hierarchy, f, indent=2, ensure_ascii=False)

    def _flush_progress(self):
        """
        Sync wrapper for flushing, used only in sync contexts if needed.
        """
        import asyncio
        try:
            asyncio.run(self._aflush_progress())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._aflush_progress())

    # ---------------------------------------------------------
    # Public API (sync wrapper)
    # ---------------------------------------------------------
    def label_hierarchy_file(self, input_json_path: str, output_json_path: str):
        """
        Synchronous wrapper around the async labeling method.
        """
        import asyncio
        try:
            return asyncio.run(self.alabel_hierarchy_file(input_json_path, output_json_path))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.alabel_hierarchy_file(input_json_path, output_json_path))

    # ---------------------------------------------------------
    # Public API (async)
    # ---------------------------------------------------------
    async def alabel_hierarchy_file(self, input_json_path: str, output_json_path: str):
        """
        Loads the hierarchy from JSON, labels it asynchronously, and writes the result to a new JSON file.
        Progress is flushed to disk after each cluster (with concurrency-safe locking).
        """
        self._log(f"Loading hierarchy from: {input_json_path}")

        with open(input_json_path, "r", encoding="utf-8") as f:
            hierarchy = json.load(f)

        # Set context for incremental flushing
        self._current_hierarchy = hierarchy
        self._current_output_path = output_json_path

        # Initialize progress counters
        self._total_clusters = self._count_clusters(hierarchy)
        self._processed_clusters = 0
        self._log(f"Total clusters to process: {self._total_clusters}")

        self._log("Starting bottom-up labeling (async)...")
        await self._aprocess_node(hierarchy)

        self._log("Final flush to disk...")
        await self._aflush_progress()

        self._log("Labeling complete.")
        return hierarchy

    # ---------------------------------------------------------
    # Recursive bottom-up processing (async)
    # ---------------------------------------------------------
    async def _aprocess_node(self, node: Dict[str, Any]) -> None:
        """
        Post-order traversal:
        - process children first
        - then summarize and label this node's clusters
        LLM calls for clusters at the same node level are parallelized.
        """
        tasks = []

        for cluster in node["clusters"]:
            if cluster["children"] is not None:
                await self._aprocess_node(cluster["children"])

            tasks.append(self._alabel_cluster(cluster))

        if tasks:
            await asyncio.gather(*tasks)

    # ---------------------------------------------------------
    # Cluster labeling logic (single async LLM call)
    # ---------------------------------------------------------
    async def _alabel_cluster(self, cluster: Dict[str, Any]) -> None:
        """
        Summarizes, labels, and extracts keywords for a cluster using ONE async LLM call.
        Uses caching: if summary/label/keywords already exist in cache, reuse them.
        Fetches raw chunk texts from the Vector DB using cluster['ids'].
        """

        self._processed_clusters += 1
        cluster_id = cluster["cluster_id"]
        self._log(
            f"\nProcessing cluster {cluster_id} "
            f"({self._processed_clusters}/{self._total_clusters})..."
        )

        # 1. Cache check
        if cluster_id in self.cache:
            self._log(f"  Cache hit for cluster {cluster_id}")
            cached = self.cache[cluster_id]
            cluster["label"] = cached.get("label", "")
            cluster["keywords"] = cached.get("keywords", [])
            if self.store_summaries:
                cluster["summary"] = cached.get("summary", "")
            await self._aflush_progress()
            return

        # 2. Build input text
        if cluster["children"] is None:
            self._log(f"  Leaf cluster: fetching {len(cluster['ids'])} chunks from VectorDB")
            texts = []
            for chunk_id in cluster["ids"]:
                record = self._get_record_by_id(chunk_id)
                texts.append(record["document"] if record else "")
            combined_text = "\n\n".join(texts)

        else:
            self._log(f"  Internal cluster: summarizing {len(cluster['children']['clusters'])} child summaries")
            child_summaries = []
            for child in cluster["children"]["clusters"]:
                if "summary" in child:
                    child_summaries.append(child["summary"])
                else:
                    fallback_texts = []
                    for chunk_id in child["ids"]:
                        record = self._get_record_by_id(chunk_id)
                        if record:
                            fallback_texts.append(record["document"])
                    child_summaries.append("\n\n".join(fallback_texts))

            combined_text = "\n\n".join(child_summaries)

        # 3. Single async LLM call (summary + label + keywords)
        self._log(f"  Calling LLM for summary + label + keywords for cluster {cluster_id}")
        prompt = self.combined_prompt.format(text=combined_text)

        async with self._semaphore:
            response = await self.llm.acomplete(prompt)

        response = response.strip()
        clean = self._clean_json(response)

        try:
            data = json.loads(clean)
        except Exception:
            self._log(f"  ERROR: LLM did not return valid JSON for cluster {cluster_id}")
            raise ValueError(f"Invalid LLM JSON response: {response}")

        summary = data.get("summary", "").strip()
        label = data.get("label", "").strip()
        keywords = data.get("keywords", [])

        # 4. Store in cluster
        cluster["label"] = label
        cluster["keywords"] = keywords
        if self.store_summaries:
            cluster["summary"] = summary

        # 5. Store in cache
        self.cache[cluster_id] = {
            "summary": summary,
            "label": label,
            "keywords": keywords,
        }

        # 6. Flush progress to disk
        await self._aflush_progress()

        self._log(f"  Stored summary + label + keywords for cluster {cluster_id}")

    # ---------------------------------------------------------
    # Helper: fetch a record from Vector DB
    # ---------------------------------------------------------
    def _get_record_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Wrapper around vectordb.get_by_id(chunk_id).
        Expects a dict with at least 'document' and 'metadata' keys.
        """
        return self.vectordb.get_by_id(chunk_id)
