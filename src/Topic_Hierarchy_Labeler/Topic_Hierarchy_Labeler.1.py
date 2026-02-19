import asyncio
import json
import os
from typing import Any, Dict, Optional, List


class Topic_Hierarchy_Labeler:
    """
    Bottom-up LLM-based labeling and summarization for a topic hierarchy.

    NEW FEATURES:
    - One LLM call per cluster.
    - LLM returns multilingual results dynamically based on `languages`.
    - Strong JSON schema enforcement.
    - Automatic repair of concatenated JSON objects.
    - Per-language output files: e.g. topics_EN.json, topics_ES.json, topics_DE.json.
    - Cache stores multilingual results.
    """

    # -------------------------------------------------------------------------
    # STRICT multilingual prompt template
    # -------------------------------------------------------------------------
    DEFAULT_COMBINED_PROMPT = (
        "You are analyzing a set of text chunks that belong to the same topic.\n"
        "For EACH of the following languages: {languages}, produce:\n"
        "- A concise summary (4–6 sentences)\n"
        "- A short label (max 6 words)\n"
        "- 5–10 high-value keywords\n\n"
        "TEXTS:\n{text}\n\n"
        "CRITICAL FORMAT RULES:\n"
        "- Respond ONLY with valid JSON.\n"
        "- The JSON MUST contain EXACTLY these top-level keys: {languages}.\n"
        "- Each top-level key MUST map to an object containing ONLY: summary, label, keywords.\n"
        "- Do NOT repeat 'label' or 'keywords' outside the language objects.\n"
        "- Do NOT output multiple JSON objects.\n"
        "- Do NOT output text before or after the JSON.\n"
        "- The FIRST character must be '{{'. The LAST must be '}}'.\n\n"
        "Return JSON in exactly this structure:\n"
        "{json_schema}\n"
    )

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        llm: Any,
        vectordb: Any,
        languages: Optional[List[str]] = None,
        store_summaries: bool = True,
        cache_path: str = "hierarchy_label_cache.json",
        combined_prompt: Optional[str] = None,
        verbose: bool = False,
        max_concurrent_llm_calls: int = 10,
    ):
        self.llm = llm
        self.vectordb = vectordb
        self.languages = list(languages) if languages else ["EN"]
        self.store_summaries = store_summaries
        self.cache_path = cache_path
        self.combined_prompt = combined_prompt or self.DEFAULT_COMBINED_PROMPT
        self.verbose = verbose

        # Progress tracking
        self._total_clusters = 0
        self._processed_clusters = 0

        # For incremental flushing
        self._current_hierarchy = None
        self._current_output_path = None

        # Concurrency
        self._semaphore = asyncio.Semaphore(max_concurrent_llm_calls)
        self._flush_lock = asyncio.Lock()

        # Load cache
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    # -------------------------------------------------------------------------
    # Logging helper
    # -------------------------------------------------------------------------
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

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
    # Clean JSON (remove code fences)
    # -------------------------------------------------------------------------
    def _clean_json(self, text: str) -> str:
        text = text.strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[-1].strip()

        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0].strip()

        return text

    # -------------------------------------------------------------------------
    # Build dynamic JSON schema for prompt
    # -------------------------------------------------------------------------
    def _build_multilang_json_schema(self) -> str:
        parts = []
        for lang in self.languages:
            parts.append(
                f'  "{lang}": {{"summary": "...", "label": "...", "keywords": ["..."]}}'
            )
        return "{\n" + ",\n".join(parts) + "\n}"

    # -------------------------------------------------------------------------
    # Flush progress (cache + base hierarchy)
    # -------------------------------------------------------------------------
    async def _aflush_progress(self):
        if not self._current_hierarchy or not self._current_output_path:
            return

        async with self._flush_lock:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)

            with open(self._current_output_path, "w", encoding="utf-8") as f:
                json.dump(self._current_hierarchy, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Sync wrapper
    # -------------------------------------------------------------------------
    def label_hierarchy_file(self, input_json_path: str, output_json_path: str):
        try:
            return asyncio.run(self.alabel_hierarchy_file(input_json_path, output_json_path))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.alabel_hierarchy_file(input_json_path, output_json_path))

    # -------------------------------------------------------------------------
    # Main async entry point
    # -------------------------------------------------------------------------
    async def alabel_hierarchy_file(self, input_json_path: str, output_json_path: str):
        self._log(f"Loading hierarchy from: {input_json_path}")

        with open(input_json_path, "r", encoding="utf-8") as f:
            hierarchy = json.load(f)

        self._current_hierarchy = hierarchy
        self._current_output_path = output_json_path

        self._total_clusters = self._count_clusters(hierarchy)
        self._processed_clusters = 0

        self._log(f"Total clusters to process: {self._total_clusters}")
        self._log("Starting bottom-up labeling (async, multilingual)...")

        await self._aprocess_node(hierarchy)

        self._log("Final flush...")
        await self._aflush_progress()

        # Write per-language files
        for lang in self.languages:
            lang_hierarchy = self._extract_language_view(hierarchy, lang)
            out_path = self._append_lang_suffix(output_json_path, lang)
            self._log(f"Writing {out_path}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(lang_hierarchy, f, indent=2, ensure_ascii=False)

        self._log("Labeling complete.")
        return hierarchy

    # -------------------------------------------------------------------------
    # Append language suffix
    # -------------------------------------------------------------------------
    def _append_lang_suffix(self, path: str, lang: str) -> str:
        base, ext = os.path.splitext(path)
        return f"{base}_{lang}{ext or '.json'}"

    # -------------------------------------------------------------------------
    # Recursive processing
    # -------------------------------------------------------------------------
    async def _aprocess_node(self, node: Dict[str, Any]):
        tasks = []

        for cluster in node["clusters"]:
            if cluster["children"] is not None:
                await self._aprocess_node(cluster["children"])

            tasks.append(self._alabel_cluster(cluster))

        if tasks:
            await asyncio.gather(*tasks)

    # -------------------------------------------------------------------------
    # JSON repair helper
    # -------------------------------------------------------------------------
    def _repair_json_if_needed(self, text: str) -> str:
        """
        If the LLM returned multiple concatenated JSON objects,
        keep only the first complete one.
        """
        # Try direct parse first
        try:
            json.loads(text)
            return text
        except Exception:
            pass

        # Attempt to extract first valid object
        last_brace = text.rfind("}")
        if last_brace != -1:
            candidate = text[: last_brace + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        return text  # fallback

    # -------------------------------------------------------------------------
    # Cluster labeling
    # -------------------------------------------------------------------------
    async def _alabel_cluster(self, cluster: Dict[str, Any]):
        self._processed_clusters += 1
        cid = cluster["cluster_id"]

        self._log(f"\nProcessing cluster {cid} ({self._processed_clusters}/{self._total_clusters})...")

        # Cache check
        if cid in self.cache:
            self._log(f"  Cache hit for cluster {cid}")
            cluster["multilang"] = self.cache[cid]
            await self._aflush_progress()
            return

        # Build input text
        if cluster["children"] is None:
            self._log(f"  Leaf cluster: fetching {len(cluster['ids'])} chunks")
            texts = []
            for chunk_id in cluster["ids"]:
                rec = self._get_record_by_id(chunk_id)
                texts.append(rec["document"] if rec else "")
            combined_text = "\n\n".join(texts)

        else:
            self._log(f"  Internal cluster: summarizing children")
            child_summaries = []
            for child in cluster["children"]["clusters"]:
                if "multilang" in child:
                    lang0 = self.languages[0]
                    if lang0 in child["multilang"]:
                        child_summaries.append(child["multilang"][lang0]["summary"])
                        continue

                # fallback to raw docs
                fallback = []
                for chunk_id in child["ids"]:
                    rec = self._get_record_by_id(chunk_id)
                    if rec:
                        fallback.append(rec["document"])
                child_summaries.append("\n\n".join(fallback))

            combined_text = "\n\n".join(child_summaries)

        # Build prompt
        json_schema = self._build_multilang_json_schema()
        prompt = self.combined_prompt.format(
            text=combined_text,
            languages=", ".join(self.languages),
            json_schema=json_schema,
        )

        # Call LLM
        async with self._semaphore:
            response = await self.llm.acomplete(prompt)

        clean = self._clean_json(response)
        clean = self._repair_json_if_needed(clean)

        # Parse JSON
        try:
            data = json.loads(clean)
        except Exception:
            raise ValueError(f"Invalid LLM JSON response: {response}")

        # Validate schema
        for lang in self.languages:
            if lang not in data:
                raise ValueError(f"LLM output missing language '{lang}'")
            if not isinstance(data[lang], dict):
                raise ValueError(f"Invalid structure for language '{lang}'")

        # Store multilingual data
        cluster["multilang"] = {}
        for lang in self.languages:
            entry = data[lang]
            cluster["multilang"][lang] = {
                "summary": entry.get("summary", "").strip(),
                "label": entry.get("label", "").strip(),
                "keywords": entry.get("keywords", []),
            }

        # Cache
        self.cache[cid] = cluster["multilang"]

        # Flush
        await self._aflush_progress()

        self._log(f"  Stored multilingual results for cluster {cid}")

    # -------------------------------------------------------------------------
    # Vector DB fetch
    # -------------------------------------------------------------------------
    def _get_record_by_id(self, chunk_id: str):
        return self.vectordb.get_by_id(chunk_id)

    # -------------------------------------------------------------------------
    # Extract per-language hierarchy
    # -------------------------------------------------------------------------
    def _extract_language_view(self, root: Dict[str, Any], lang: str) -> Dict[str, Any]:
        hierarchy = json.loads(json.dumps(root))  # deep copy

        def recurse(node):
            for c in node["clusters"]:
                if "multilang" in c and lang in c["multilang"]:
                    c["summary"] = c["multilang"][lang]["summary"]
                    c["label"] = c["multilang"][lang]["label"]
                    c["keywords"] = c["multilang"][lang]["keywords"]
                if c["children"] is not None:
                    recurse(c["children"])

        recurse(hierarchy)
        return hierarchy
