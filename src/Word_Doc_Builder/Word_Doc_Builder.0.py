import json
from pathlib import Path
from Utilities import Simple_Progress_Bar
from docxtpl import DocxTemplate


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
        word_template_path: str = None
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

        b_context = self._load_info_file(parent_id, "B_Context")

        result = {
            "internal_process_name": b_context.get("process_name") if b_context else "",
            "internal_B_Context": b_context.get("business_context") if b_context else "",
            "data_elements": {
                "BOs": []
            },
            "leaf_processes": []
        }

        for leaf in self._extract_children(parent_node):
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
                        "bo_name": bo.get("bo_name", ""),
                        "bo_description": bo.get("bo_description", "")
                    })

            # Leaf process entry
            leaf_entry = {
                "leaf_process_name": process_b.get("process_name") if process_b else "",
                "leaf_process_description": process_b.get("process_description") if process_b else "",
                "tasks": {
                    "task": []
                },
                "qa": {
                    "elements": []
                }
            }

            # Steps
            if steps:
                for step in steps.get("process_steps", []):
                    d = step.get("details", {})
                    leaf_entry["tasks"]["task"].append({
                        "step_name": step.get("name", ""),
                        "step_context": d.get("context", ""),
                        "step_objective": d.get("objective", ""),
                        "step_explanation": d.get("explanation", ""),
                        "step_precondition": d.get("pre-condition", ""),
                        "step_postcondition": d.get("post-condition", ""),
                        "step_exceptions": d.get("exceptions", ""),
                        "step_warnings": d.get("warnings", "")
                    })

            # Q&A
            qa_list = []
            if what:
                qa_list.extend(what.get("WHAT_Answers", []))
            if why:
                qa_list.extend(why.get("WHY_Answers", []))

            for qa in qa_list:
                leaf_entry["qa"]["elements"].append({
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", "")
                })

            result["leaf_processes"].append(leaf_entry)

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
        outputs = {}

        bar = Simple_Progress_Bar(total=len(parent_clusters), enabled=True) if self.show_progress_bar else None

        for parent in parent_clusters:
            parent_id = parent["cluster_id"]
            self._log(f"Processing parent cluster {parent_id}")

            word_json = self._build_word_json(parent)
            outputs[parent_id] = word_json

            # Store JSON inside the parent cluster folder
            if self.log_json:
                parent_folder = self.working_folder / parent_id
                parent_folder.mkdir(parents=True, exist_ok=True)

                out_path = parent_folder / f"{parent_id}_word_doc.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(word_json, f, indent=2, ensure_ascii=False)

            # Word generation (still disabled)
            self._render_word_document(parent_id, word_json)

            if bar:
                bar.update(1, label=f"Cluster {parent_id}")

        return outputs
