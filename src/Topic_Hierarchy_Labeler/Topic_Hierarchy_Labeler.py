import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional, List


class Topic_Hierarchy_Labeler:
    """
    Bottom-up LLM-based labeling and summarization for a topic hierarchy.
    Now includes:
    - Guaranteed labels (prompt + fallback)
    - Missing-label detection
    - Repair mode regenerates labels only for missing-label clusters
    """

    DEFAULT_COMBINED_PROMPT = (
        "You are analyzing a set of text chunks that belong to the same topic.\n"
        "For EACH of the following languages: {languages}, produce:\n"
        "- A concise summary (4–6 sentences)\n"
        "- A short label (max 6 words). YOU MUST ALWAYS PROVIDE A LABEL.\n"
        "- 5–10 high-value keywords\n\n"
        "If the topic is unclear, too small, or ambiguous, you MUST STILL provide the best possible descriptive label.\n"
        "Never return an empty label, 'None', 'N/A', or similar placeholders.\n\n"
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
        self.languages = list(languages) if languages else None
        self.store_summaries = store_summaries
        self.cache_path = cache_path
        self.combined_prompt = combined_prompt or self.DEFAULT_COMBINED_PROMPT

        self.retry_attempts = retry_attempts
        self.repair_mode = repair_mode

        self.show_progress_bar = show_progress_bar
        self.verbose = False if show_progress_bar else verbose
        self._start_time = None

        self._total_clusters = 0
        self._processed_clusters = 0
        self._successful_clusters = 0
        self._incomplete_clusters = 0

        self._current_hierarchy = None
        self._current_output_path = None

        self._semaphore = asyncio.Semaphore(max_concurrent_llm_calls)
        self._flush_lock = asyncio.Lock()

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
    # Time formatting
    # -------------------------------------------------------------------------
    def _fmt_time(self, seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # -------------------------------------------------------------------------
    # Progress bar
    # -------------------------------------------------------------------------
    def _update_progress_bar(self):
        if not self.show_progress_bar:
            return

        pct = (self._processed_clusters / self._total_clusters) * 100
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        elapsed = time.time() - self._start_time
        avg = elapsed / max(1, self._processed_clusters)
        est_total = avg * self._total_clusters
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

    # -------------------------------------------------------------------------
    # Missing-label detection
    # -------------------------------------------------------------------------
    def _cluster_missing_label(self, cluster):
        if "multilang" not in cluster:
            return True
        for lang in self.languages:
            label = cluster["multilang"].get(lang, {}).get("label", "")
            if not label or not label.strip():
                return True
        return False

    # -------------------------------------------------------------------------
    # Detect Language
    # -------------------------------------------------------------------------
    async def _detect_language(self, text: str) -> str:
        """
        Uses the LLM to detect the dominant language of the input text.
        Returns a 2-letter ISO code (e.g., EN, ES, DE, FR).
        Defaults to EN if detection fails.
        """
        prompt = (
            "Detect the primary language of the following text. "
            "Respond ONLY with a 2-letter ISO language code (e.g., EN, ES, DE, FR). "
            "Text:\n\n"
            f"{text}"
        )

        try:
            response = await self.llm.acomplete(prompt)
            code = response.strip().upper()
            # Basic validation
            if len(code) == 2 and code.isalpha():
                return code
        except Exception:
            pass

        return "EN"

    # -------------------------------------------------------------------------
    # Count incomplete clusters (including missing labels)
    # -------------------------------------------------------------------------
    def _count_incomplete(self, node: Dict[str, Any]) -> int:
        count = 0
        for c in node["clusters"]:
            if c.get("incomplete") or self._cluster_missing_label(c):
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
    # JSON schema for prompt
    # -------------------------------------------------------------------------
    def _build_multilang_json_schema(self) -> str:
        parts = [
            f'  "{lang}": {{"summary": "...", "label": "...", "keywords": ["..."]}}'
            for lang in self.languages
        ]
        return "{\n" + ",\n".join(parts) + "\n}"

    # -------------------------------------------------------------------------
    # Safe flush
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

        for lang in self.languages:
            lang_hierarchy = self._extract_language_view(hierarchy, lang)
            out_path = self._append_lang_suffix(output_json_path, lang)
            self._log(f"Writing {out_path}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(lang_hierarchy, f, indent=2, ensure_ascii=False)

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

        # NEW: treat missing labels as incomplete
        if not cluster.get("incomplete") and self._cluster_missing_label(cluster):
            cluster["incomplete"] = True

        # Skip healthy clusters in repair mode
        if self.repair_mode and not cluster.get("incomplete"):
            self._successful_clusters += 1
            await self._aflush_progress()
            return

        # Cache check
        if not self.repair_mode and cid in self.cache:
            cached = self.cache[cid]
            cluster["multilang"] = cached["multilang"]
            cluster["source_language"] = cached["source_language"]
            self._successful_clusters += 1
            await self._aflush_progress()
            return

        # Build input text
        if cluster["children"] is None:
            texts = []
            for chunk_id in cluster["ids"]:
                rec = self._get_record_by_id(chunk_id)
                texts.append(rec["document"] if rec else "")
            combined_text = "\n\n".join(texts)
        else:
            child_summaries = []
            for child in cluster["children"]["clusters"]:
                if child.get("incomplete"):
                    child_summaries.append("[INCOMPLETE]")
                    continue

                lang = child.get("source_language")
                if "multilang" in child and lang in child["multilang"]:
                    child_summaries.append(child["multilang"][lang]["summary"])
                else:
                    child_summaries.append("[INCOMPLETE]")

            combined_text = "\n\n".join(child_summaries)

        # Auto-detect language if not provided
        if self.languages is None:
            detected = await self._detect_language(combined_text)
            self.languages = [detected]
            self._log(f"Auto-detected language: {detected}")

        # Build prompt
        json_schema = self._build_multilang_json_schema()
        prompt = self.combined_prompt.format(
            text=combined_text,
            languages=", ".join(self.languages),
            json_schema=json_schema,
        )

        # Retry logic
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

                for lang in self.languages:
                    if lang not in data or not isinstance(data[lang], dict):
                        raise ValueError(f"Missing or invalid language '{lang}'")

                break

            except Exception:
                if attempt == self.retry_attempts - 1:
                    cluster["incomplete"] = True
                    self._incomplete_clusters += 1
                    self._mark_ancestors_incomplete(cluster)
                    await self._aflush_progress()
                    return
                else:
                    await asyncio.sleep(2 ** attempt)

        # Store multilingual data with HARD fallback
        cluster["multilang"] = {}

        for lang in self.languages:
            summary = data[lang].get("summary", "").strip()
            label = data[lang].get("label", "").strip()
            keywords = data[lang].get("keywords", [])

            # HARD GUARANTEE: enforce non-empty label
            if not label:
                # Leaf fallback
                if cluster["children"] is None and cluster.get("metadatas"):
                    label = cluster["metadatas"][0].get("document_name", "Unnamed Cluster")

                # Internal fallback
                elif cluster["children"] is not None:
                    for child in cluster["children"]["clusters"]:
                        child_label = child.get("multilang", {}).get(lang, {}).get("label")
                        if child_label:
                            label = child_label
                            break

                # Final fallback
                if not label:
                    label = f"Cluster {cluster['cluster_id']}"

            cluster["multilang"][lang] = {
                "summary": summary,
                "label": label,
                "keywords": keywords,
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

        if "incomplete" in cluster:
            del cluster["incomplete"]

        # Cache successful clusters
        self.cache[cid] = {
            "multilang": cluster["multilang"],
            "source_language": cluster["source_language"],
        }

        self._successful_clusters += 1
        await self._aflush_progress()

    # -------------------------------------------------------------------------
    # Vector DB
    # -------------------------------------------------------------------------
    def _get_record_by_id(self, chunk_id: str):
        return self.vectordb.get_by_id(chunk_id)

    # -------------------------------------------------------------------------
    # Per-language view
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
