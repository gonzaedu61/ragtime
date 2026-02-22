import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utilities import Simple_Progress_Bar


# ------------------------------------------------------------
# STEP 1 — PROCESS MODELS (one worker per process)
# ------------------------------------------------------------
def step_processes_1_models(builder):
    cluster_ids = builder._collect_cluster_ids()
    step_dir = builder._ensure_step_dir(builder.processes_dir, 1)

    # Build flat list of (cluster_id, process_name)
    tasks = []
    for cluster_id in cluster_ids:
        baseline = builder._load_cluster_baseline(cluster_id)
        for name in baseline.get("processes", []):
            tasks.append((cluster_id, name))

    pb = Simple_Progress_Bar(total=len(tasks), enabled=builder.progress_enabled)
    label = "Processes / Step_1 (models)"

    prompt_template = builder.prompt_loader.load("processes/step1_process_model.txt")
    pb.update(step=0, label=label)

    def process_process(cluster_id, name):
        baseline = builder._load_cluster_baseline(cluster_id)
        proc_id = f"{cluster_id}::{name}"
        out_path = os.path.join(step_dir, f"{builder._sanitize_id(proc_id)}_step1.json")

        if os.path.exists(out_path):
            return

        prompt = builder.prompt_loader.fill(
            prompt_template,
            cluster_baseline=baseline,
            process_name=name,
            cluster_id=cluster_id,
        )

        llm_output = builder.llm_wrapper.call(prompt)

        try:
            proc = json.loads(llm_output)
        except Exception:
            proc = {
                "id": proc_id,
                "name": name,
                "cluster_id": cluster_id,
                "description": "",
                "steps": [],
                "inputs": [],
                "outputs": [],
                "entities_involved": [],
                "relationships_involved": [],
                "pre_requisites": [],
                "constraints": [],
            }

        builder._save_json(out_path, proc)

    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {
            executor.submit(process_process, cluster_id, name): (cluster_id, name)
            for cluster_id, name in tasks
        }
        for _ in as_completed(futures):
            pb.update(step=1, label=label)


# ------------------------------------------------------------
# STEP 2 — PROCESS ENRICHMENT (one worker per process file)
# ------------------------------------------------------------
def step_processes_2_enrich(builder):
    prev_step_dir = builder._ensure_step_dir(builder.processes_dir, 1)
    step_dir = builder._ensure_step_dir(builder.processes_dir, 2)

    files = [f for f in os.listdir(prev_step_dir) if f.endswith("_step1.json")]
    pb = Simple_Progress_Bar(total=len(files), enabled=builder.progress_enabled)
    label = "Processes / Step_2 (enrich)"

    prompt_template = builder.prompt_loader.load("processes/step2_enrichment.txt")
    pb.update(step=0, label=label)

    def process_proc_file(fname):
        in_path = os.path.join(prev_step_dir, fname)
        proc = builder._load_json(in_path)

        prompt = builder.prompt_loader.fill(prompt_template, process_json=proc)
        llm_output = builder.llm_wrapper.call(prompt)

        try:
            enriched = json.loads(llm_output)
        except Exception:
            enriched = proc

        out_path = os.path.join(step_dir, fname.replace("_step1.json", "_step2.json"))
        builder._save_json(out_path, enriched)

    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {executor.submit(process_proc_file, f): f for f in files}
        for _ in as_completed(futures):
            pb.update(step=1, label=label)
