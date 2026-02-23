import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utilities import Simple_Progress_Bar

import sys

def _is_valid_json(path: str) -> bool:

    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


# ------------------------------------------------------------
# STEP 1 — ENTITY STUB GENERATION (one worker per entity)
# ------------------------------------------------------------
def step_entities_1_stubs(builder):
    cluster_ids = builder._collect_cluster_ids()
    step_dir = builder._ensure_step_dir(builder.entities_dir, 1)

    tasks = []
    for cluster_id in cluster_ids:
        baseline = builder._load_cluster_baseline(cluster_id)
        for idx, name in enumerate(baseline.get("entities", [])):
            tasks.append((cluster_id, idx, name))

    already_done = 0
    for cluster_id, idx, name in tasks:
        out_path = os.path.join(step_dir, f"{cluster_id}__{idx}_step1.json")
        if os.path.exists(out_path) and _is_valid_json(out_path):
            already_done += 1

    pb = Simple_Progress_Bar(total=len(tasks), enabled=builder.progress_enabled)
    label = "Entities / Step_1 (stubs)"
    pb.current = already_done
    pb.update(step=0, label=label)

    prompt_template = builder.prompt_loader.load("entities/step1_stub_generation.txt")

    def process_entity(cluster_id, idx, name):
        out_path = os.path.join(step_dir, f"{cluster_id}__{idx}_step1.json")
        if os.path.exists(out_path) and _is_valid_json(out_path):
            return

        baseline = builder._load_cluster_baseline(cluster_id)

        prompt = builder.prompt_loader.fill(
            prompt_template,
            cluster_baseline=baseline,
            entity_name=name,
            cluster_id=cluster_id,
        )

        llm_output = builder.llm_wrapper.call(prompt)

        try:
            stub = json.loads(llm_output)
        except Exception:
            stub = {
                "id": f"{cluster_id}::{idx}",
                "name": name,
                "cluster_id": cluster_id,
                "description": "",
                "aliases": [],
                "attributes": [],
                "relationships": [],
                "processes": [],
            }

        builder._save_json(out_path, stub)

    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {
            executor.submit(process_entity, cluster_id, idx, name): (cluster_id, idx)
            for cluster_id, idx, name in tasks
            if not (os.path.exists(os.path.join(step_dir, f"{cluster_id}__{idx}_step1.json"))
                    and _is_valid_json(os.path.join(step_dir, f"{cluster_id}__{idx}_step1.json")))
        }

        for _ in as_completed(futures):
            pb.update(step=1, label=label)


# ------------------------------------------------------------
# STEP 2 — ENTITY ENRICHMENT (one worker per entity file)
# ------------------------------------------------------------
def step_entities_2_enrich(builder):
    prev_step_dir = builder._ensure_step_dir(builder.entities_dir, 1)
    step_dir = builder._ensure_step_dir(builder.entities_dir, 2)

    # All step1 outputs, regardless of naming, as long as they end with _step1.json
    files = [f for f in os.listdir(prev_step_dir) if f.endswith("_step1.json")]

    # Count already completed (valid JSON) outputs
    already_done = 0
    for fname in files:
        out_path = os.path.join(step_dir, fname.replace("_step1.json", "_step2.json"))
        if os.path.exists(out_path) and _is_valid_json(out_path):
            already_done += 1

    pb = Simple_Progress_Bar(total=len(files), enabled=builder.progress_enabled)
    label = "Entities / Step_2 (enrich)"
    pb.current = already_done
    pb.update(step=0, label=label)

    prompt_template = builder.prompt_loader.load("entities/step2_enrichment.txt")

    def process_entity_file(fname):
        try:
            in_path = os.path.join(prev_step_dir, fname)
            out_path = os.path.join(step_dir, fname.replace("_step1.json", "_step2.json"))

            if os.path.exists(out_path) and _is_valid_json(out_path):
                return

            entity = builder._load_json(in_path)

            entity_json_str = json.dumps(entity, ensure_ascii=False)
            entity_json_str = entity_json_str.replace("{", "{{").replace("}", "}}")

            prompt = builder.prompt_loader.fill(prompt_template, entity_json=entity_json_str)

            #print("Processing:", fname, flush=True)
            #print("Prompt:", flush=True)
            #print('-----------------------------------------------------', flush=True)

            llm_output = builder.llm_wrapper.call(prompt)

            enriched = json.loads(llm_output)

            builder._save_json(out_path, enriched)

        except Exception as e:
            print(f"ERROR in {fname}: {e}", flush=True)
            raise



    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {
            executor.submit(process_entity_file, f): f
            for f in files
            if not (
                os.path.exists(os.path.join(step_dir, f.replace("_step1.json", "_step2.json")))
                and _is_valid_json(os.path.join(step_dir, f.replace("_step1.json", "_step2.json")))
            )
        }
        
        for _ in as_completed(futures):
            pb.update(step=1, label=label)


# ------------------------------------------------------------
# STEP 3 — ENTITY FINALIZATION (one worker per entity file)
# ------------------------------------------------------------
def step_entities_3_finalize(builder):
    prev_step_dir = builder._ensure_step_dir(builder.entities_dir, 2)
    step_dir = builder._ensure_step_dir(builder.entities_dir, 3)

    # All step2 outputs, regardless of naming, as long as they end with _step2.json
    files = [f for f in os.listdir(prev_step_dir) if f.endswith("_step2.json")]

    # Count already completed (valid JSON) outputs
    already_done = 0
    for fname in files:
        out_path = os.path.join(step_dir, fname.replace("_step2.json", "_step3.json"))
        if os.path.exists(out_path) and _is_valid_json(out_path):
            already_done += 1

    pb = Simple_Progress_Bar(total=len(files), enabled=builder.progress_enabled)
    label = "Entities / Step_3 (final)"
    pb.current = already_done
    pb.update(step=0, label=label)

    prompt_template = builder.prompt_loader.load("entities/step3_finalization.txt")

    def process_entity_file(fname):
        in_path = os.path.join(prev_step_dir, fname)
        out_path = os.path.join(step_dir, fname.replace("_step2.json", "_step3.json"))

        # Skip if already done and valid
        if os.path.exists(out_path) and _is_valid_json(out_path):
            return

        entity = builder._load_json(in_path)

        entity_json_str = json.dumps(entity, ensure_ascii=False)
        entity_json_str = entity_json_str.replace("{", "{{").replace("}", "}}")
        prompt = builder.prompt_loader.fill(prompt_template, entity_json=entity_json_str)

        llm_output = builder.llm_wrapper.call(prompt)

        try:
            final = json.loads(llm_output)
        except Exception:
            final = entity

        builder._save_json(out_path, final)

    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {
            executor.submit(process_entity_file, f): f
            for f in files
            if not (
                os.path.exists(os.path.join(step_dir, f.replace("_step2.json", "_step3.json")))
                and _is_valid_json(os.path.join(step_dir, f.replace("_step2.json", "_step3.json")))
            )
        }

        for _ in as_completed(futures):
            pb.update(step=1, label=label)
