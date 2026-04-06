import json
from pathlib import Path
from Utilities import Simple_Progress_Bar
from docxtpl import DocxTemplate

import re

INVALID_XML_CHARS = r"[\x00-\x08\x0B\x0C\x0E-\x1F]"

def sanitize(text):
    if not isinstance(text, str):
        return ""

    # Remove illegal XML chars
    text = re.sub(INVALID_XML_CHARS, "", text)

    # Remove soft hyphens (invisible but break Word)
    text = text.replace("\u00AD", "")

    # Replace non-breaking spaces with normal spaces
    text = text.replace("\u00A0", " ")

    # Remove zero-width characters
    text = text.replace("\u200B", "")
    text = text.replace("\u200C", "")
    text = text.replace("\u200D", "")

    # Normalize line/paragraph separators
    text = text.replace("\u2028", "\n")
    text = text.replace("\u2029", "\n")

    # Replace smart quotes with ASCII equivalents
    replacements = {
        "„": "\"", "“": "\"", "”": "\"",
        "’": "'", "‘": "'",
        "–": "-", "—": "-"
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Escape XML-breaking characters
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")        

    return text


class WordDocBuilder:
    def __init__(
        self,
        working_folder: str,
        tree_pathname: str,
        branch_id: str,
        verbose: bool = False,
        show_progress_bar: bool = False,
        log_json: bool = False,
        enable_word_generation: bool = False,
        word_template_path: str = None,
        use_existing_json: bool = False
    ):
        self.working_folder = Path(working_folder)
        self.tree_pathname = Path(tree_pathname)
        self.branch_id = str(branch_id)

        # Mutually exclusive output modes
        self.show_progress_bar = show_progress_bar
        self.verbose = verbose if not show_progress_bar else False

        self.log_json = log_json
        self.enable_word_generation = enable_word_generation
        self.word_template_path = Path(word_template_path) if word_template_path else None
        self.use_existing_json = use_existing_json

        self.tree = self._load_json(self.tree_pathname)

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------
    def _load_json(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_info_file(self, cluster_id: str, info_type: str):
        file_path = self.working_folder / cluster_id / f"{cluster_id}_{info_type}.json"
        if not file_path.exists():
            return None
        return self._load_json(file_path)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _load_existing_json(self, cluster_id: str):
        json_path = self.working_folder / cluster_id / f"{cluster_id}_word_doc.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Existing JSON not found at {json_path}")
        return self._load_json(json_path)

    # ---------------------------------------------------------
    # Tree traversal helpers
    # ---------------------------------------------------------
    def _extract_children(self, node):
        if not isinstance(node, dict):
            return []

        if "clusters" in node and isinstance(node["clusters"], list):
            return node["clusters"]

        if "children" in node and isinstance(node["children"], dict):
            return node["children"].get("clusters", [])

        return []

    def _find_node(self, node, cluster_id):
        if isinstance(node, dict) and node.get("cluster_id") == cluster_id:
            return node

        for child in self._extract_children(node):
            found = self._find_node(child, cluster_id)
            if found:
                return found

        return None

    def _is_leaf(self, node):
        return len(self._extract_children(node)) == 0

    def _collect_parent_clusters(self, start_node):
        parents = []

        def recurse(n):
            children = self._extract_children(n)
            if not children:
                return

            all_leaf = all(self._is_leaf(c) for c in children)
            if all_leaf:
                parents.append(n)
            else:
                for c in children:
                    recurse(c)

        recurse(start_node)
        return parents

    # ---------------------------------------------------------
    # JSON building
    # ---------------------------------------------------------
    def _build_word_json(self, parent_node):
        parent_id = parent_node["cluster_id"]

        # Load parent or leaf header files
        b_context = self._load_info_file(parent_id, "B_Context")
        enrichment = self._load_info_file(parent_id, "enrichment")

        result = {
            "internal_topic_name": sanitize(enrichment.get("label", "")) if enrichment else "",
            "internal_topic_summary": sanitize(enrichment.get("summary", "")) if enrichment else "",
            "internal_B_Context": sanitize(b_context.get("business_context", "")) if b_context else "",
            "data_elements": {
                "BOs_title": "Business Objects",   # optional, avoids missing key
                "BOs": []
            },
            "leaf_entries": []
        }

        children = self._extract_children(parent_node)

        # ---------------------------------------------------------
        # CASE 1: parent_node is a LEAF → treat it as its own leaf process
        # ---------------------------------------------------------
        if not children:
            leaf = parent_node
            leaf_id = leaf["cluster_id"]

            bo_file = self._load_info_file(leaf_id, "BO")
            process_b = self._load_info_file(leaf_id, "process_b")
            steps = self._load_info_file(leaf_id, "steps")
            what = self._load_info_file(leaf_id, "WHAT")
            why = self._load_info_file(leaf_id, "WHY")

            # BOs
            if bo_file:
                for bo in bo_file.get("business_objects", []):
                    result["data_elements"]["BOs"].append({
                        "bo_name": sanitize(bo.get("bo_name", "")),
                        "bo_description": sanitize(bo.get("bo_description", ""))
                    })

            # Leaf entry
            leaf_entry = {
                "leaf_process_name": sanitize(process_b.get("process_name", "")) if process_b else "",
                "leaf_process_description": sanitize(process_b.get("process_description", "")) if process_b else "",
                "tasks": {"task": []},
                "qa": {"elements": []}
            }

            # Steps
            if steps:
                for step in steps.get("process_steps", []):
                    d = step.get("details", {})
                    leaf_entry["tasks"]["task"].append({
                        "step_name": sanitize(step.get("name", "")),
                        "step_context": sanitize(d.get("context", "")),
                        "step_objective": sanitize(d.get("objective", "")),
                        "step_explanation": sanitize(d.get("explanation", "")),
                        "step_precondition": sanitize(d.get("pre-condition", "")),
                        "step_postcondition": sanitize(d.get("post-condition", "")),
                        "step_exceptions": sanitize(d.get("exceptions", "")),
                        "step_warnings": sanitize(d.get("warnings", ""))
                    })

            # Q&A
            qa_list = []
            if what:
                qa_list.extend(what.get("WHAT_Answers", []))
            if why:
                qa_list.extend(why.get("WHY_Answers", []))

            for qa in qa_list:
                leaf_entry["qa"]["elements"].append({
                    "question": sanitize(qa.get("question", "")),
                    "answer": sanitize(qa.get("answer", ""))
                })

            result["leaf_entries"].append(leaf_entry)
            return result

        # ---------------------------------------------------------
        # CASE 2: parent_node has children → original behavior
        # ---------------------------------------------------------
        for leaf in children:
            leaf_id = leaf["cluster_id"]

            leaf_b_context = self._load_info_file(leaf_id, "B_Context")
            leaf_enrichment = self._load_info_file(leaf_id, "enrichment")
            bo_file = self._load_info_file(leaf_id, "BO")
            concept = self._load_info_file(leaf_id, "concept")
            process_b = self._load_info_file(leaf_id, "process_b")
            steps = self._load_info_file(leaf_id, "steps")
            what = self._load_info_file(leaf_id, "WHAT")
            why = self._load_info_file(leaf_id, "WHY")

            # BOs
            if bo_file:
                for bo in bo_file.get("business_objects", []):
                    result["data_elements"]["BOs"].append({
                        "bo_name": sanitize(bo.get("bo_name", "")),
                        "bo_description": sanitize(bo.get("bo_description", ""))
                    })

            # Leaf entry
            leaf_entry = {
                "leaf_topic_name": sanitize(leaf_enrichment.get("label", "")) if leaf_enrichment else "",
                "leaf_topic_summary": sanitize(leaf_enrichment.get("summary", "")) if leaf_enrichment else "",
                "leaf_B_Context": sanitize(leaf_b_context.get("business_context", "")) if leaf_b_context else "",
                "leaf_concept": sanitize(concept.get("concept_description", "")) if concept else "",
                "leaf_process_name": sanitize(process_b.get("process_name", "")) if process_b else "",
                "leaf_process_description": sanitize(process_b.get("process_description", "")) if process_b else "",
                "concept_elements": [],
                "tasks": {"task": []},
                "qa": {"elements": []}
            }


            # Concept Elements
            if concept:
                for e in concept.get("concept_structure", []):
                    leaf_entry["concept_elements"].append({
                        "element_name": sanitize(e.get("concept_element_name", "")),
                        "element_description": sanitize(e.get("description", "")),
                    })

            # Steps
            if steps:
                for step in steps.get("process_steps", []):
                    d = step.get("details", {})
                    leaf_entry["tasks"]["task"].append({
                        "step_name": sanitize(step.get("name", "")),
                        "step_context": sanitize(d.get("context", "")),
                        "step_objective": sanitize(d.get("objective", "")),
                        "step_explanation": sanitize(d.get("explanation", "")),
                        "step_precondition": sanitize(d.get("pre-condition", "")),
                        "step_postcondition": sanitize(d.get("post-condition", "")),
                        "step_exceptions": sanitize(d.get("exceptions", "")),
                        "step_warnings": sanitize(d.get("warnings", ""))
                    })

            # Q&A
            qa_list = []
            if what:
                qa_list.extend(what.get("WHAT_Answers", []))
            if why:
                qa_list.extend(why.get("WHY_Answers", []))

            for qa in qa_list:
                leaf_entry["qa"]["elements"].append({
                    "question": sanitize(qa.get("question", "")),
                    "answer": sanitize(qa.get("answer", ""))
                })

            result["leaf_entries"].append(leaf_entry)

        return result

    # ---------------------------------------------------------
    # Word generation (disabled by default)
    # ---------------------------------------------------------
    def _render_word_document(self, cluster_id: str, json_data: dict):
        if not self.enable_word_generation:
            return

        if not self.word_template_path:
            raise ValueError("Word template path must be provided when Word generation is enabled.")

        # Output folder = <working_folder>/<cluster_id>/
        output_folder = self.working_folder / cluster_id
        output_folder.mkdir(parents=True, exist_ok=True)

        # --- ENABLE WORD GENERATION ---
        doc = DocxTemplate(self.word_template_path)
        doc.render(json_data)

        output_path = output_folder / f"{cluster_id}.docx"

        # --- PRINT OUTPUT PATH ---
        print(f"[WordDocBuilder] Saving Word document to: {output_path}")

        doc.save(output_path)

    # ---------------------------------------------------------
    # Main method
    # ---------------------------------------------------------
    def generate_word(self):
        root = self.tree
        start_node = self._find_node(root, self.branch_id)
        if not start_node:
            raise ValueError(f"Cluster {self.branch_id} not found in tree.")

        parent_clusters = self._collect_parent_clusters(start_node)

        # FIX: only treat start_node as a leaf if it is actually a leaf
        if not parent_clusters and self._is_leaf(start_node):
            parent_clusters = [start_node]

        outputs = {}
        bar = Simple_Progress_Bar(total=len(parent_clusters), enabled=True) if self.show_progress_bar else None

        for parent in parent_clusters:
            parent_id = parent["cluster_id"]
            self._log(f"Processing parent cluster {parent_id}")

            if self.use_existing_json:
                self._log(f"Loading existing JSON for cluster {parent_id}")
                word_json = self._load_existing_json(parent_id)
            else:
                word_json = self._build_word_json(parent)
                outputs[parent_id] = word_json

                if self.log_json:
                    parent_folder = self.working_folder / parent_id
                    parent_folder.mkdir(parents=True, exist_ok=True)
                    out_path = parent_folder / f"{parent_id}_word_doc.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(word_json, f, indent=2, ensure_ascii=False)

            self._render_word_document(parent_id, word_json)

            if bar:
                bar.update(1, label=f"Cluster {parent_id}")

        return outputs
