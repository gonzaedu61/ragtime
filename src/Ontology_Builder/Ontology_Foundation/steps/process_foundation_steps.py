import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utilities import Simple_Progress_Bar


def _is_valid_json(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


# ------------------------------------------------------------
# STEP 1 — PROCESS MODELS (one worker per process)
# ------------------------------------------------------------
def step_processes_1_models(builder):
    cluster_ids = builder._collect_cluster_ids()
    step_dir = builder._ensure_step_dir(builder.processes_dir, 1)

    tasks = []
    for cluster_id in cluster_ids:
        baseline = builder._load_cluster_baseline(cluster_id)
        for idx, name in enumerate(baseline.get("processes", [])):
            tasks.append((cluster_id, idx, name))

    already_done = 0
    for cluster_id, idx, name in tasks:
        out_path = os.path.join(step_dir, f"{cluster_id}__proc_{idx}_step1.json")
        if os.path.exists(out_path) and _is_valid_json(out_path):
            already_done += 1

    pb = Simple_Progress_Bar(total=len(tasks), enabled=builder.progress_enabled)
    label = "Processes / Step_1 (models)"
    pb.current = already_done
    pb.update(step=0, label=label)

    prompt_template = builder.prompt_loader.load("processes/step1_process_model.txt")

    def process_process(cluster_id, idx, name):
        out_path = os.path.join(step_dir, f"{cluster_id}__proc_{idx}_step1.json")
        if os.path.exists(out_path) and _is_valid_json(out_path):
            return

        baseline = builder._load_cluster_baseline(cluster_id)

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
                "id": f"{cluster_id}::proc::{idx}",
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
            executor.submit(process_process, cluster_id, idx, name): (cluster_id, idx)
            for cluster_id, idx, name in tasks
            if not (os.path.exists(os.path.join(step_dir, f"{cluster_id}__proc_{idx}_step1.json"))
                    and _is_valid_json(os.path.join(step_dir, f"{cluster_id}__proc_{idx}_step1.json")))
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

    already_done = 0
    for fname in files:
        out_path = os.path.join(step_dir, fname.replace("_step1.json", "_step2.json"))
        if os.path.exists(out_path) and _is_valid_json(out_path):
            already_done += 1

    pb = Simple_Progress_Bar(total=len(files), enabled=builder.progress_enabled)
    label = "Processes / Step_2 (enrich)"
    pb.current = already_done
    pb.update(step=0, label=label)

    prompt_template = builder.prompt_loader.load("processes/step2_enrichment.txt")

    def process_proc_file(fname):
        in_path = os.path.join(prev_step_dir, fname)
        out_path = os.path.join(step_dir, fname.replace("_step1.json", "_step2.json"))

        if os.path.exists(out_path) and _is_valid_json(out_path):
            return

        proc = builder._load_json(in_path)
        proc = proc.replace("{", "{{").replace("}", "}}")
        prompt = builder.prompt_loader.fill(prompt_template, process_json=proc)
        llm_output = builder.llm_wrapper.call(prompt)

        try:
            enriched = json.loads(llm_output)
        except Exception:
            enriched = proc

        builder._save_json(out_path, enriched)

    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {
            executor.submit(process_proc_file, f): f
            for f in files
            if not (os.path.exists(os.path.join(step_dir, f.replace("_step1.json", "_step2.json")))
                    and _is_valid_json(os.path.join(step_dir, f.replace("_step1.json", "_step2.json"))))
        }

        for _ in as_completed(futures):
            pb.update(step=1, label=label)


# ------------------------------------------------------------
# STEP 3 — PROCESS FINALIZATION (one worker per process file)
# ------------------------------------------------------------
def step_processes_3_finalize(builder):
    prev_step_dir = builder._ensure_step_dir(builder.processes_dir, 2)
    step_dir = builder._ensure_step_dir(builder.processes_dir, 3)

    # All step2 outputs
    files = [f for f in os.listdir(prev_step_dir) if f.endswith("_step2.json")]

    # Count already completed
    already_done = 0
    for fname in files:
        out_path = os.path.join(step_dir, fname.replace("_step2.json", "_step3.json"))
        if os.path.exists(out_path) and _is_valid_json(out_path):
            already_done += 1

    pb = Simple_Progress_Bar(total=len(files), enabled=builder.progress_enabled)
    label = "Processes / Step_3 (final)"
    pb.current = already_done
    pb.update(step=0, label=label)

    prompt_template = builder.prompt_loader.load("processes/step3_finalization.txt")

    def process_proc_file(fname):
        try:
            in_path = os.path.join(prev_step_dir, fname)
            out_path = os.path.join(step_dir, fname.replace("_step2.json", "_step3.json"))

            if os.path.exists(out_path) and _is_valid_json(out_path):
                return

            proc = builder._load_json(in_path)

            # Escape JSON for .format()
            proc_json_str = json.dumps(proc, ensure_ascii=False)
            proc_json_str = proc_json_str.replace("{", "{{").replace("}", "}}")

            prompt = builder.prompt_loader.fill(
                prompt_template,
                process_json=proc_json_str
            )

            llm_output = builder.llm_wrapper.call(prompt)

            try:
                final = json.loads(llm_output)
            except Exception:
                final = proc  # fallback

            builder._save_json(out_path, final)

        except Exception as e:
            print(f"ERROR in Process Step 3 for {fname}: {e}", flush=True)
            raise

    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {
            executor.submit(process_proc_file, f): f
            for f in files
            if not (
                os.path.exists(os.path.join(step_dir, f.replace("_step2.json", "_step3.json")))
                and _is_valid_json(os.path.join(step_dir, f.replace("_step2.json", "_step3.json")))
            )
        }

        for _ in as_completed(futures):
            pb.update(step=1, label=label)

