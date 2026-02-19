import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional, List


class Topic_Hierarchy_Labeler:
    """
    Bottom-up LLM-based labeling and summarization for a topic hierarchy.

    Features:
    - One LLM call per cluster.
    - Multilingual output for a configurable list of languages.
    - Robust JSON extraction from LLM responses (brace-balanced).
    - Retry logic with exponential backoff.
    - Incomplete cluster detection and placeholder summaries.
    - Internal nodes NEVER fall back to raw chunks.
    - Each cluster has a `source_language`.
    - Writes one hierarchy JSON per language.
    - Removes `multilang` and `source_language` from per-language output files.
    - Optional progress bar with time estimate.
    - Repair mode: only process incomplete clusters + ancestors.
    - Final summary of successful vs incomplete clusters.
    """

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

    def __init__(
        self,
        llm: Any,
        vectordb: Any,
        languages: Optional[List[str]] = None,
        store_summaries: bool = True,
        cache_path: str = "hierarchy_label_cache.json",
        combined_prompt: Optional[str] = None,
        verbose: bool = False,
        show_progress_bar: bool = False,
        retry_attempts: int = 3,
        repair_mode: bool = False,
        max_concurrent_llm_calls: int = 10,
    ):
        self.llm = llm
        self.vectordb = vectordb
        self.languages = list(languages) if languages else ["EN"]
        self.store_summaries = store_summaries
        self.cache_path = cache_path
        self.combined_prompt = combined_prompt or self.DEFAULT_COMBINED_PROMPT

        # Retry configuration
        self.retry_attempts = retry_attempts

        # Repair mode
        self.repair_mode = repair_mode

        # Progress bar logic
        self.show_progress_bar = show_progress_bar
        self.verbose = False if show_progress_bar else verbose
        self._start_time = None

        # Counters
        self._total_clusters = 0
        self._processed_clusters = 0
        self._successful_clusters = 0
        self._incomplete_clusters = 0

        self._current_hierarchy = None
        self._current_output_path = None

        self._semaphore = asyncio.Semaphore(max_concurrent_llm_calls)
        self._flush_lock = asyncio.Lock()

        # Load cache
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # -------------------------------------------------------------------------
    # Time formatting helper
    # -------------------------------------------------------------------------
    def _fmt_time(self, seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # -------------------------------------------------------------------------
    # Progress bar with time estimate
    # -------------------------------------------------------------------------
    def _update_progress_bar(self):
        if not self.show_progress_bar:
            return

        pct = (self._processed_clusters / self._total_clusters) * 100
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        elapsed = time.time() - self._start_time
        avg_per_cluster = elapsed / max(1, self._processed_clusters)
        est_total = avg_per_cluster * self._total_clusters
        remaining = est_total - elapsed

        sys.stdout.write(
            f"\r[{bar}] {pct:5.1f}%  "
            f"({self._processed_clusters}/{self._total_clusters})  "
            f"Elapsed: {self._fmt_time(elapsed)} | "
            f"ETA: {self._fmt_time(remaining)} | "
            f"Total est.: {self._fmt_time(est_total)}"
        )
        sys.stdout.flush()

        if self._processed_clusters == self._total_clusters:
            print()

    # -------------------------------------------------------------------------
    # Count clusters
    # -------------------------------------------------------------------------
    def _count_clusters(self, node: Dict[str, Any]) -> int:
        count = len(node["clusters"])
        for c in node["clusters"]:
            if c["children"] is not None:
                count += self._count_clusters(c["children"])
        return count

    def _count_incomplete(self, node: Dict[str, Any]) -> int:
        count = 0
        for c in node["clusters"]:
            if c.get("incomplete"):
                count += 1
            if c["children"] is not None:
                count += self._count_incomplete(c["children"])
        return count

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
    # Extract first JSON object (brace-balanced)
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
    # JSON schema for prompt
    # -------------------------------------------------------------------------
    def _build_multilang_json_schema(self) -> str:
        parts = [
            f'  "{lang}": {{"summary": "...", "label": "...", "keywords": ["..."]}}'
            for lang in self.languages
        ]
        return "{\n" + ",\n".join(parts) + "\n}"

    # -------------------------------------------------------------------------
    # Flush progress
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

        # Count clusters depending on mode
        if self.repair_mode:
            self._total_clusters = self._count_incomplete(hierarchy)
        else:
            self._total_clusters = self._count_clusters(hierarchy)

        self._processed_clusters = 0
        self._successful_clusters = 0
        self._incomplete_clusters = 0
        self._start_time = time.time()

        if not self.show_progress_bar:
            mode = "Repair mode" if self.repair_mode else "Full mode"
            self._log(f"{mode}: {self._total_clusters} clusters to process")
            self._log("Starting...")

        await self._aprocess_node(hierarchy)

        await self._aflush_progress()

        # Write per-language files
        for lang in self.languages:
            lang_hierarchy = self._extract_language_view(hierarchy, lang)
            out_path = self._append_lang_suffix(output_json_path, lang)
            self._log(f"Writing {out_path}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(lang_hierarchy, f, indent=2, ensure_ascii=False)

        # Final summary
        if self.repair_mode:
            print("\nRepair mode summary:")
        else:
            print("\nSummary:")

        print(f"  Successful clusters: {self._successful_clusters}")
        print(f"  Incomplete clusters: {self._incomplete_clusters}")

        return hierarchy

    # -------------------------------------------------------------------------
    # Filename helper
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
    # Mark ancestors incomplete
    # -------------------------------------------------------------------------
    def _mark_ancestors_incomplete(self, cluster):
        parent = cluster.get("parent")
        while parent:
            parent["incomplete"] = True
            parent = parent.get("parent")

    # -------------------------------------------------------------------------
    # Cluster labeling
    # -------------------------------------------------------------------------
    async def _alabel_cluster(self, cluster: Dict[str, Any]):
        self._processed_clusters += 1
        self._update_progress_bar()

        cid = cluster["cluster_id"]

        # Skip healthy clusters in repair mode
        if self.repair_mode and not cluster.get("incomplete"):
            self._successful_clusters += 1
            return

        # Cache check (only successful clusters are cached)
        if not self.repair_mode and cid in self.cache:
            cached = self.cache[cid]
            cluster["multilang"] = cached["multilang"]
            cluster["source_language"] = cached["source_language"]
            self._successful_clusters += 1
            return

        # Build input text
        if cluster["children"] is None:
            # Leaf → raw chunks
            texts = []
            for chunk_id in cluster["ids"]:
                rec = self._get_record_by_id(chunk_id)
                texts.append(rec["document"] if rec else "")
            combined_text = "\n\n".join(texts)
        else:
            # Internal → child summaries ONLY
            child_summaries = []
            for child in cluster["children"]["clusters"]:
                if child.get("incomplete"):
                    child_summaries.append("[INCOMPLETE]")
                    continue

                child_lang = child.get("source_language")
                if "multilang" in child and child_lang in child["multilang"]:
                    child_summaries.append(child["multilang"][child_lang]["summary"])
                else:
                    child_summaries.append("[INCOMPLETE]")

            combined_text = "\n\n".join(child_summaries)

        # Build prompt
        json_schema = self._build_multilang_json_schema()
        prompt = self.combined_prompt.format(
            text=combined_text,
            languages=", ".join(self.languages),
            json_schema=json_schema,
        )

        # Retry logic with exponential backoff
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

                # Validate schema
                for lang in self.languages:
                    if lang not in data or not isinstance(data[lang], dict):
                        raise ValueError(f"Missing or invalid language '{lang}'")

                break  # success

            except Exception:
                if attempt == self.retry_attempts - 1:
                    # All retries failed → mark incomplete
                    cluster["incomplete"] = True
                    self._incomplete_clusters += 1
                    self._mark_ancestors_incomplete(cluster)
                    return
                else:
                    await asyncio.sleep(2 ** attempt)

        # Store multilingual data
        cluster["multilang"] = {
            lang: {
                "summary": data[lang].get("summary", "").strip(),
                "label": data[lang].get("label", "").strip(),
                "keywords": data[lang].get("keywords", []),
            }
            for lang in self.languages
        }

        # Determine source_language
        if cluster["children"] is None:
            cluster["source_language"] = max(
                self.languages,
                key=lambda l: len(cluster["multilang"][l]["summary"])
            )
        else:
            scores: Dict[str, int] = {}
            for child in cluster["children"]["clusters"]:
                lang = child.get("source_language")
                if lang and "multilang" in child and lang in child["multilang"]:
                    scores[lang] = scores.get(lang, 0) + len(child["multilang"][lang]["summary"])
            cluster["source_language"] = max(scores, key=scores.get) if scores else self.languages[0]

        # Remove incomplete flag if repaired
        if "incomplete" in cluster:
            del cluster["incomplete"]

        # Cache only successful clusters
        self.cache[cid] = {
            "multilang": cluster["multilang"],
            "source_language": cluster["source_language"],
        }

        self._successful_clusters += 1

    # -------------------------------------------------------------------------
    # Vector DB
    # -------------------------------------------------------------------------
    def _get_record_by_id(self, chunk_id: str):
        return self.vectordb.get_by_id(chunk_id)

    # -------------------------------------------------------------------------
    # Per-language hierarchy view (removes multilang and source_language)
    # -------------------------------------------------------------------------
    def _extract_language_view(self, root: Dict[str, Any], lang: str) -> Dict[str, Any]:
        hierarchy = json.loads(json.dumps(root))

        def recurse(node: Dict[str, Any]):
            for c in node["clusters"]:
                if "multilang" in c and lang in c["multilang"]:
                    c["summary"] = c["multilang"][lang]["summary"]
                    c["label"] = c["multilang"][lang]["label"]
                    c["keywords"] = c["multilang"][lang]["keywords"]

                if "multilang" in c:
                    del c["multilang"]

                if "source_language" in c:
                    del c["source_language"]

                if c["children"] is not None:
                    recurse(c["children"])

        recurse(hierarchy)
        return hierarchy
