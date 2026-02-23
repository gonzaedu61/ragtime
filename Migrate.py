import os
import json
import re

# ------------------------------------------------------------
# CONFIG — adjust these paths if needed
# ------------------------------------------------------------
BASELINE_DIR = r"C:\Users\gonza\OneDrive\Desktop\Python\ragtime\DATA\KBs\Test_KB/7_Ontology_Files\Clusters_Baseline"
STEP1_DIR = r"C:\Users\gonza\OneDrive\Desktop\Python\ragtime\DATA\KBs\Test_KB/7_Ontology_Files\Foundation\Entities\Step_1"


# Old sanitize logic (to detect old filenames)
def old_sanitize(s: str) -> str:
    return (
        s.replace("::", "__")
         .replace(" ", "_")
         .replace("/", "_")
         .replace("\\", "_")
         .replace(">", "_")
         .replace("<", "_")
    )

# ------------------------------------------------------------
# LOAD BASELINES
# ------------------------------------------------------------
baselines = []
for fname in os.listdir(BASELINE_DIR):
    if fname.startswith("base_") and fname.endswith("_knowledge.json"):
        cluster_id = fname[len("base_"):-len("_knowledge.json")]
        path = os.path.join(BASELINE_DIR, fname)
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            baselines.append((cluster_id, data))
        except Exception as e:
            print(f"[ERROR] Cannot load baseline {fname}: {e}")

print(f"\nLoaded {len(baselines)} baselines.\n")

# ------------------------------------------------------------
# PROCESS EACH ENTITY
# ------------------------------------------------------------
renamed = []
missing_old = []
errors = []

for cluster_id, data in baselines:
    entities = data.get("entities", [])
    for idx, name in enumerate(entities):

        # New filename
        new_filename = f"{cluster_id}__{idx}_step1.json"
        new_path = os.path.join(STEP1_DIR, new_filename)

        # Old filename (sanitized)
        old_id = f"{cluster_id}::{name}"
        old_filename = old_sanitize(old_id) + "_step1.json"
        old_path = os.path.join(STEP1_DIR, old_filename)

        # If new file already exists, skip
        if os.path.exists(new_path):
            continue

        # If old file does not exist, record missing
        if not os.path.exists(old_path):
            missing_old.append((cluster_id, name, old_filename))
            continue

        # Try renaming
        try:
            os.rename(old_path, new_path)
            renamed.append((old_filename, new_filename))
        except Exception as e:
            errors.append((old_filename, str(e)))

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------
print("\n=== RENAME SUMMARY ===")
print(f"Renamed {len(renamed)} files.\n")
for old, new in renamed:
    print(f"{old}  -->  {new}")

print("\n=== MISSING OLD FILES ===")
print(f"{len(missing_old)} missing old filenames.\n")
for cid, name, old in missing_old:
    print(f"{cid} :: {name}  (expected old file: {old})")

print("\n=== ERRORS ===")
print(f"{len(errors)} errors.\n")
for old, err in errors:
    print(f"{old}  --> ERROR: {err}")

print("\nMigration complete.\n")
