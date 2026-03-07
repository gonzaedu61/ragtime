import re

INVALID_XML_CHARS = r"[\x00-\x08\x0B\x0C\x0E-\x1F]"

def sanitize(text):
    if not isinstance(text, str):
        return ""
    # Remove illegal XML chars
    text = re.sub(INVALID_XML_CHARS, "", text)
    # Remove BOM and zero-width spaces
    text = text.replace("\uFEFF", "").replace("\u200B", "")
    # Normalize line separators
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    return text


def _build_word_json(self, parent_node):
    parent_id = parent_node["cluster_id"]

    # Load B_Context for parent or leaf
    b_context = self._load_info_file(parent_id, "B_Context")

    result = {
        "internal_process_name": sanitize(b_context.get("process_name", "")) if b_context else "",
        "internal_B_Context": sanitize(b_context.get("business_context", "")) if b_context else "",
        "data_elements": {
            "BOs_title": "Business Objects",   # optional, avoids missing key
            "BOs": []
        },
        "leaf_processes": []
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

        result["leaf_processes"].append(leaf_entry)
        return result

    # ---------------------------------------------------------
    # CASE 2: parent_node has children → original behavior
    # ---------------------------------------------------------
    for leaf in children:
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

        result["leaf_processes"].append(leaf_entry)

    return result
