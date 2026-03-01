import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utilities import Simple_Progress_Bar


# ------------------------------------------------------------
# Helper: validate JSON before skipping
# ------------------------------------------------------------
def _is_valid_json(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


# ------------------------------------------------------------
# STEP 1 — RELATIONSHIP SKELETONS (one worker per relationship)
# ------------------------------------------------------------
def step_relationships_1_skeletons(builder):
    cluster_ids = builder._collect_cluster_ids()
    step_dir = builder._ensure_step_dir(builder.relationships_dir, 1)

    # Build flat list of (cluster_id, index, relationship_dict)
    tasks = []
    for cluster_id in cluster_ids:
        baseline = builder._load_cluster_baseline(cluster_id)
        for idx, rel in enumerate(baseline.get("relationships", [])):
            tasks.append((cluster_id, idx, rel))

    # Count already completed outputs
    already_done = 0
    for cluster_id, idx, rel in tasks:
        out_path = os.path.join(step_dir, f"{cluster_id}__rel_{idx}_step1.json")
        if os.path.exists(out_path) and _is_valid_json(out_path):
            already_done += 1

    pb = Simple_Progress_Bar(total=len(tasks), enabled=builder.progress_enabled)
    label = "Relationships / Step_1 (skeletons)"
    pb.current = already_done
    pb.update(step=0, label=label)

    prompt_template = builder.prompt_loader.load(
        "relationships/step1_skeleton_generation.txt"
    )

    def process_relationship(cluster_id, idx, rel):
        out_path = os.path.join(step_dir, f"{cluster_id}__rel_{idx}_step1.json")

        if os.path.exists(out_path) and _is_valid_json(out_path):
            return

        baseline = builder._load_cluster_baseline(cluster_id)

        prompt = builder.prompt_loader.fill(
            prompt_template,
            source_entity=rel["source"],
            target_entity=rel["target"],
            cluster_baseline=baseline,
            cluster_id=cluster_id,
        )

        llm_output = builder.llm_wrapper.call(prompt)

        try:
            skeleton = json.loads(llm_output)
        except Exception:
            skeleton = {
                "id": f"{cluster_id}::rel::{idx}",
                "source": rel["source"],
                "target": rel["target"],
                "type": rel.get("type", "related_to"),
                "description": "",
                "confidence": 0.5,
                "attributes": [],
                "constraints": [],
                "cluster_id": cluster_id,
            }

        builder._save_json(out_path, skeleton)

    # Submit only tasks that need processing
    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {
            executor.submit(process_relationship, cluster_id, idx, rel): (cluster_id, idx)
            for cluster_id, idx, rel in tasks
            if not (
                os.path.exists(os.path.join(step_dir, f"{cluster_id}__rel_{idx}_step1.json"))
                and _is_valid_json(os.path.join(step_dir, f"{cluster_id}__rel_{idx}_step1.json"))
            )
        }

        for _ in as_completed(futures):
            pb.update(step=1, label=label)


# ------------------------------------------------------------
# STEP 2 — RELATIONSHIP ENRICHMENT (one worker per relationship file)
# ------------------------------------------------------------
def step_relationships_2_enrich(builder):
    prev_step_dir = builder._ensure_step_dir(builder.relationships_dir, 1)
    step_dir = builder._ensure_step_dir(builder.relationships_dir, 2)

    # All step1 outputs, regardless of naming, as long as they end with _step1.json
    files = [f for f in os.listdir(prev_step_dir) if f.endswith("_step1.json")]

    # Count already completed outputs
    already_done = 0
    for fname in files:
        out_path = os.path.join(step_dir, fname.replace("_step1.json", "_step2.json"))
        if os.path.exists(out_path) and _is_valid_json(out_path):
            already_done += 1

    pb = Simple_Progress_Bar(total=len(files), enabled=builder.progress_enabled)
    label = "Relationships / Step_2 (enrich)"
    pb.current = already_done
    pb.update(step=0, label=label)

    prompt_template = builder.prompt_loader.load(
        "relationships/step2_enrichment.txt"
    )

    def process_rel_file(fname):
        in_path = os.path.join(prev_step_dir, fname)
        out_path = os.path.join(step_dir, fname.replace("_step1.json", "_step2.json"))

        if os.path.exists(out_path) and _is_valid_json(out_path):
            return

        rel = builder._load_json(in_path, ensure_ascii=False)
        rel = rel.replace("{", "{{").replace("}", "}}")
        prompt = builder.prompt_loader.fill(prompt_template, relationship_json=rel)
        llm_output = builder.llm_wrapper.call(prompt)

        try:
            enriched = json.loads(llm_output)
        except Exception:
            enriched = rel

        builder._save_json(out_path, enriched)

    # Submit only tasks that need processing
    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {
            executor.submit(process_rel_file, f): f
            for f in files
            if not (
                os.path.exists(os.path.join(step_dir, f.replace("_step1.json", "_step2.json")))
                and _is_valid_json(os.path.join(step_dir, f.replace("_step1.json", "_step2.json")))
            )
        }

        for _ in as_completed(futures):
            pb.update(step=1, label=label)
