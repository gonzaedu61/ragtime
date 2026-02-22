import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utilities import Simple_Progress_Bar


# ------------------------------------------------------------
# STEP 1 — ENTITY STUB GENERATION (one worker per entity)
# ------------------------------------------------------------
def step_entities_1_stubs(builder):
    cluster_ids = builder._collect_cluster_ids()
    step_dir = builder._ensure_step_dir(builder.entities_dir, 1)

    # Build flat list of (cluster_id, entity_name)
    tasks = []
    for cluster_id in cluster_ids:
        baseline = builder._load_cluster_baseline(cluster_id)
        for name in baseline.get("entities", []):
            tasks.append((cluster_id, name))

    pb = Simple_Progress_Bar(total=len(tasks), enabled=builder.progress_enabled)
    label = "Entities / Step_1 (stubs)"

    prompt_template = builder.prompt_loader.load("entities/step1_stub_generation.txt")
    pb.update(step=0, label=label)

    def process_entity(cluster_id, name):
        baseline = builder._load_cluster_baseline(cluster_id)
        entity_id = f"{cluster_id}::{name}"
        out_path = os.path.join(step_dir, f"{builder._sanitize_id(entity_id)}_step1.json")

        if os.path.exists(out_path):
            return

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
                "id": entity_id,
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
            executor.submit(process_entity, cluster_id, name): (cluster_id, name)
            for cluster_id, name in tasks
        }
        for _ in as_completed(futures):
            pb.update(step=1, label=label)


# ------------------------------------------------------------
# STEP 2 — ENTITY ENRICHMENT (one worker per entity file)
# ------------------------------------------------------------
def step_entities_2_enrich(builder):
    prev_step_dir = builder._ensure_step_dir(builder.entities_dir, 1)
    step_dir = builder._ensure_step_dir(builder.entities_dir, 2)

    files = [f for f in os.listdir(prev_step_dir) if f.endswith("_step1.json")]
    pb = Simple_Progress_Bar(total=len(files), enabled=builder.progress_enabled)
    label = "Entities / Step_2 (enrich)"

    prompt_template = builder.prompt_loader.load("entities/step2_enrichment.txt")
    pb.update(step=0, label=label)

    def process_entity_file(fname):
        in_path = os.path.join(prev_step_dir, fname)
        entity = builder._load_json(in_path)

        prompt = builder.prompt_loader.fill(prompt_template, entity_json=entity)
        llm_output = builder.llm_wrapper.call(prompt)

        try:
            enriched = json.loads(llm_output)
        except Exception:
            enriched = entity

        out_path = os.path.join(step_dir, fname.replace("_step1.json", "_step2.json"))
        builder._save_json(out_path, enriched)

    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {executor.submit(process_entity_file, f): f for f in files}
        for _ in as_completed(futures):
            pb.update(step=1, label=label)


# ------------------------------------------------------------
# STEP 3 — ENTITY FINALIZATION (one worker per entity file)
# ------------------------------------------------------------
def step_entities_3_finalize(builder):
    prev_step_dir = builder._ensure_step_dir(builder.entities_dir, 2)
    step_dir = builder._ensure_step_dir(builder.entities_dir, 3)

    files = [f for f in os.listdir(prev_step_dir) if f.endswith("_step2.json")]
    pb = Simple_Progress_Bar(total=len(files), enabled=builder.progress_enabled)
    label = "Entities / Step_3 (final)"

    prompt_template = builder.prompt_loader.load("entities/step3_finalization.txt")
    pb.update(step=0, label=label)

    def process_entity_file(fname):
        in_path = os.path.join(prev_step_dir, fname)
        entity = builder._load_json(in_path)

        prompt = builder.prompt_loader.fill(prompt_template, entity_json=entity)
        llm_output = builder.llm_wrapper.call(prompt)

        try:
            final = json.loads(llm_output)
        except Exception:
            final = entity

        out_path = os.path.join(step_dir, fname.replace("_step2.json", "_step3.json"))
        builder._save_json(out_path, final)

    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {executor.submit(process_entity_file, f): f for f in files}
        for _ in as_completed(futures):
            pb.update(step=1, label=label)
