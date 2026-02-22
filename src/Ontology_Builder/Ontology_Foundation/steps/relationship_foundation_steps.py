import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utilities import Simple_Progress_Bar


# ------------------------------------------------------------
# STEP 1 — RELATIONSHIP SKELETONS (one worker per relationship)
# ------------------------------------------------------------
def step_relationships_1_skeletons(builder):
    cluster_ids = builder._collect_cluster_ids()
    step_dir = builder._ensure_step_dir(builder.relationships_dir, 1)

    # Build flat list of (cluster_id, relationship_dict)
    tasks = []
    for cluster_id in cluster_ids:
        baseline = builder._load_cluster_baseline(cluster_id)
        for rel in baseline.get("relationships", []):
            tasks.append((cluster_id, rel))

    pb = Simple_Progress_Bar(total=len(tasks), enabled=builder.progress_enabled)
    label = "Relationships / Step_1 (skeletons)"

    prompt_template = builder.prompt_loader.load("relationships/step1_skeleton_generation.txt")
    pb.update(step=0, label=label)

    def process_relationship(cluster_id, rel):
        baseline = builder._load_cluster_baseline(cluster_id)

        src = rel["source"]
        tgt = rel["target"]
        rtype = rel.get("type", "related_to")

        rel_id = f"{cluster_id}::{src}->{rtype}->{tgt}"
        out_path = os.path.join(step_dir, f"{builder._sanitize_id(rel_id)}_step1.json")

        if os.path.exists(out_path):
            return

        prompt = builder.prompt_loader.fill(
            prompt_template,
            source_entity=src,
            target_entity=tgt,
            cluster_baseline=baseline,
            cluster_id=cluster_id,
        )

        llm_output = builder.llm_wrapper.call(prompt)

        try:
            skeleton = json.loads(llm_output)
        except Exception:
            skeleton = {
                "id": rel_id,
                "source": src,
                "target": tgt,
                "type": rtype,
                "description": "",
                "confidence": 0.5,
                "attributes": [],
                "constraints": [],
                "cluster_id": cluster_id,
            }

        builder._save_json(out_path, skeleton)

    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {
            executor.submit(process_relationship, cluster_id, rel): (cluster_id, rel)
            for cluster_id, rel in tasks
        }
        for _ in as_completed(futures):
            pb.update(step=1, label=label)


# ------------------------------------------------------------
# STEP 2 — RELATIONSHIP ENRICHMENT (one worker per relationship file)
# ------------------------------------------------------------
def step_relationships_2_enrich(builder):
    prev_step_dir = builder._ensure_step_dir(builder.relationships_dir, 1)
    step_dir = builder._ensure_step_dir(builder.relationships_dir, 2)

    files = [f for f in os.listdir(prev_step_dir) if f.endswith("_step1.json")]
    pb = Simple_Progress_Bar(total=len(files), enabled=builder.progress_enabled)
    label = "Relationships / Step_2 (enrich)"

    prompt_template = builder.prompt_loader.load("relationships/step2_enrichment.txt")
    pb.update(step=0, label=label)

    def process_rel_file(fname):
        in_path = os.path.join(prev_step_dir, fname)
        rel = builder._load_json(in_path)

        prompt = builder.prompt_loader.fill(prompt_template, relationship_json=rel)
        llm_output = builder.llm_wrapper.call(prompt)

        try:
            enriched = json.loads(llm_output)
        except Exception:
            enriched = rel

        out_path = os.path.join(step_dir, fname.replace("_step1.json", "_step2.json"))
        builder._save_json(out_path, enriched)

    with ThreadPoolExecutor(max_workers=builder.max_workers) as executor:
        futures = {executor.submit(process_rel_file, f): f for f in files}
        for _ in as_completed(futures):
            pb.update(step=1, label=label)
