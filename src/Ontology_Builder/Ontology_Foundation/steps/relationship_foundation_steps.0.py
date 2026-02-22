import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utilities import Simple_Progress_Bar


# ------------------------------------------------------------
# STEP 1 — RELATIONSHIP SKELETONS
# ------------------------------------------------------------
def step_relationships_1_skeletons(builder):
    cluster_ids = builder._collect_cluster_ids()
    step_dir = builder._ensure_step_dir(builder.relationships_dir, 1)

    pb = Simple_Progress_Bar(total=len(cluster_ids), enabled=builder.progress_enabled)
    label = "Relationships / Step_1 (skeletons)"

    prompt_template = builder.prompt_loader.load("relationships/step1_skeleton_generation.txt")
    pb.update(step=0, label=label)

    def process_cluster(cluster_id):
        baseline = builder._load_cluster_baseline(cluster_id)
        candidate_rels = baseline.get("relationships", [])

        for rel in candidate_rels:
            src = rel["source"]
            tgt = rel["target"]
            rtype = rel.get("type", "related_to")

            rel_id = f"{cluster_id}::{src}->{rtype}->{tgt}"
            out_path = os.path.join(step_dir, f"{builder._sanitize_id(rel_id)}_step1.json")

            if os.path.exists(out_path):
                continue

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
        futures = {executor.submit(process_cluster, cid): cid for cid in cluster_ids}
        for i, future in enumerate(as_completed(futures), start=1):
            pb.update(step=1, label=label)
            future.result()


# ------------------------------------------------------------
# STEP 2 — RELATIONSHIP ENRICHMENT
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
        for i, future in enumerate(as_completed(futures), start=1):
            pb.update(step=1, label=label)
            future.result()
