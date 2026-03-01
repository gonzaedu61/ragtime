import os
import json
import sys
sys.stdout.reconfigure(line_buffering=True)


# --- Utilities ---
from .utils import Prompt_Loader
from .utils import LLM_Wrapper

# --- Step modules ---
from .steps import step_entities_1_stubs, step_entities_2_enrich, step_entities_3_finalize
from .steps import step_relationships_1_skeletons, step_relationships_2_enrich
from .steps import step_processes_1_models, step_processes_2_enrich, step_processes_3_finalize
from .steps import step_attributes_1_extract

class Ontology_Foundation_Builder:
    """
    Main orchestrator for the ontology foundation pipeline.

    Responsibilities:
    - Load hierarchy + cluster baselines
    - Manage pipeline state (resumable)
    - Provide shared utilities to step modules
    - Execute all refinement steps in correct order
    - Maintain directory structure for intermediate JSONs
    """

    STATE_FILE_NAME = "pipeline_state.json"

    def __init__(
        self,
        hierarchy_path: str,
        baseline_dir: str,
        foundation_dir: str,
        llm,
        vdb,
        max_workers: int = 8,
        progress_enabled: bool = True,
    ):
        self.hierarchy_path = hierarchy_path
        self.baseline_dir = baseline_dir
        self.foundation_dir = foundation_dir
        self.llm = llm
        self.vdb = vdb
        self.max_workers = max_workers
        self.progress_enabled = progress_enabled

        # Ensure foundation directory exists
        os.makedirs(self.foundation_dir, exist_ok=True)

        # Element root dirs
        self.entities_dir = os.path.join(self.foundation_dir, "Entities")
        self.relationships_dir = os.path.join(self.foundation_dir, "Relationships")
        self.processes_dir = os.path.join(self.foundation_dir, "Processes")
        self.attributes_dir = os.path.join(self.foundation_dir, "Attributes")

        for d in [
            self.entities_dir,
            self.relationships_dir,
            self.processes_dir,
            self.attributes_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        # Load hierarchy
        self.hierarchy = self._load_json(self.hierarchy_path)

        # Load pipeline state
        self.state_path = os.path.join(self.foundation_dir, self.STATE_FILE_NAME)
        self.state = self._load_state()

        # Prompt loader + LLM wrapper
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.prompt_loader = Prompt_Loader(project_root)
        self.llm_wrapper = LLM_Wrapper(self.llm)

        # Cache for cluster baselines
        self.cluster_baselines = {}

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def run(self):
        """
        Execute the full ontology foundation pipeline.
        Resumable thanks to pipeline_state.json.
        """

        # --- ENTITIES ---
        self._run_step("Entities", 1, lambda: step_entities_1_stubs(self))
        self._run_step("Entities", 2, lambda: step_entities_2_enrich(self))
        self._run_step("Entities", 3, lambda: step_entities_3_finalize(self))

        # --- RELATIONSHIPS ---
        #self._run_step("Relationships", 1, lambda: step_relationships_1_skeletons(self))
        #self._run_step("Relationships", 2, lambda: step_relationships_2_enrich(self))

        # --- PROCESSES ---
        #self._run_step("Processes", 1, lambda: step_processes_1_models(self))
        #self._run_step("Processes", 2, lambda: step_processes_2_enrich(self))
        #self._run_step("Processes", 3, lambda: step_processes_3_finalize(self))


        # --- ATTRIBUTES ---
        #self._run_step("Attributes", 1, lambda: step_attributes_1_extract(self))

    # -------------------------------------------------------------------------
    # STEP ORCHESTRATION
    # -------------------------------------------------------------------------

    def _run_step(self, element: str, step: int, fn):
        key = f"{element}_Step_{step}"

        # Always mark as running (informational only)
        self._mark_step_running(key)

        # Always call the step function
        fn()

        # Mark as completed after the function returns
        self._mark_step_completed(key)

    # -------------------------------------------------------------------------
    # HIERARCHY + BASELINES
    # -------------------------------------------------------------------------

    def _collect_cluster_ids(self):
        """
        Collect ALL cluster_ids from the hierarchy, regardless of how the
        'children' field is structured (null, {}, missing, or containing clusters).
        """
        ids = []

        def recurse(node):
            # 1. Record this cluster_id
            cid = node.get("cluster_id")
            if cid:
                ids.append(cid)

            # 2. Extract children safely
            children = node.get("children")

            # Case A: children is None → leaf node → stop
            if children is None:
                return

            # Case B: children is a dict → may contain "clusters"
            if isinstance(children, dict):
                subclusters = children.get("clusters", [])
                for child in subclusters:
                    recurse(child)
                return

            # Case C: children exists but is not dict (rare malformed case)
            # → ignore safely
            return

        # Start recursion from top-level clusters
        for root in self.hierarchy.get("clusters", []):
            recurse(root)

        return ids

    def _load_cluster_baseline(self, cluster_id: str):
        """
        Load baseline JSON for a given cluster_id.
        """
        if cluster_id in self.cluster_baselines:
            return self.cluster_baselines[cluster_id]

        fname = f"base_{cluster_id}_knowledge.json"
        path = os.path.join(self.baseline_dir, fname)

        if os.path.exists(path):
            baseline = self._load_json(path)
        else:
            baseline = {
                "label": "",
                "summary": "",
                "keywords": [],
                "entities": [],
                "processes": [],
            }

        self.cluster_baselines[cluster_id] = baseline
        return baseline

    # -------------------------------------------------------------------------
    # DIRECTORY HELPERS
    # -------------------------------------------------------------------------

    def _ensure_step_dir(self, element_root: str, step: int):
        step_dir = os.path.join(element_root, f"Step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        return step_dir

    def _sanitize_id(self, s: str):
        return (
            s.replace("::", "__")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(">", "_")
            .replace("<", "_")
        )

    # -------------------------------------------------------------------------
    # STATE MANAGEMENT
    # -------------------------------------------------------------------------

    def _load_state(self):
        if os.path.exists(self.state_path):
            return self._load_json(self.state_path)
        return {"steps": {}}

    def _save_state(self):
        self._save_json(self.state_path, self.state)

    def _is_step_completed(self, key: str):
        return self.state["steps"].get(key, {}).get("status") == "completed"

    def _mark_step_running(self, key: str):
        self.state["steps"].setdefault(key, {})
        self.state["steps"][key]["status"] = "running"
        self._save_state()

    def _mark_step_completed(self, key: str):
        self.state["steps"].setdefault(key, {})
        self.state["steps"][key]["status"] = "completed"
        self._save_state()

    # -------------------------------------------------------------------------
    # JSON IO
    # -------------------------------------------------------------------------

    def _load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json(self, path: str, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
