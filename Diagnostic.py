import os
import json
import re

# ------------------------------------------------------------
# CONFIG — adjust these paths if needed
# ------------------------------------------------------------
BASELINE_DIR = r"C:\Users\gonza\OneDrive\Desktop\Python\ragtime\DATA\KBs\Test_KB/7_Ontology_Files\Clusters_Baseline"
ENTITIES_STEP1_DIR = r"C:\Users\gonza\OneDrive\Desktop\Python\ragtime\DATA\KBs\Test_KB/7_Ontology_Files\Foundation\Entities\Step_1"

# Same sanitize logic as your builder
def sanitize(s: str) -> str:
    return (
        s.replace("::", "__")
         .replace(" ", "_")
         .replace("/", "_")
         .replace("\\", "_")
         .replace(">", "_")
         .replace("<", "_")
    )

# Invalid filename characters on Windows
INVALID_CHARS = r'[<>:"/\\|?*\x00-\x1F]'

# ------------------------------------------------------------
# LOAD ALL BASELINES
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
# COLLECT ENTITY NAMES AND SANITIZED FILENAMES
# ------------------------------------------------------------
entities = []  # list of (cluster_id, entity_name, sanitized_filename)
duplicates = {}  # sanitized_filename -> list of (cluster_id, entity_name)
invalid_names = []  # list of (cluster_id, entity_name, reason)

for cluster_id, data in baselines:
    for name in data.get("entities", []):
        if not isinstance(name, str) or not name.strip():
            invalid_names.append((cluster_id, name, "Empty or non-string"))
            continue

        # Check invalid characters
        if re.search(INVALID_CHARS, name):
            invalid_names.append((cluster_id, name, "Invalid filename characters"))
        
        sanitized = sanitize(f"{cluster_id}::{name}") + "_step1.json"
        entities.append((cluster_id, name, sanitized))

        duplicates.setdefault(sanitized, []).append((cluster_id, name))

# ------------------------------------------------------------
# REPORT DUPLICATES AFTER SANITIZATION
# ------------------------------------------------------------
print("=== DUPLICATES AFTER SANITIZATION ===")
dupes = {k: v for k, v in duplicates.items() if len(v) > 1}
if not dupes:
    print("No duplicates found.\n")
else:
    for sanitized, items in dupes.items():
        print(f"\nSanitized filename: {sanitized}")
        for cid, name in items:
            print(f"  - {cid} :: {name}")
    print()

# ------------------------------------------------------------
# REPORT INVALID ENTITY NAMES
# ------------------------------------------------------------
print("=== INVALID ENTITY NAMES ===")
if not invalid_names:
    print("No invalid names.\n")
else:
    for cid, name, reason in invalid_names:
        print(f"{cid} :: {name}  --> {reason}")
    print()

# ------------------------------------------------------------
# DETECT MISSING STEP 1 OUTPUT FILES
# ------------------------------------------------------------
print("=== MISSING STEP 1 OUTPUT FILES ===")
missing = []
for cid, name, sanitized in entities:
    out_path = os.path.join(ENTITIES_STEP1_DIR, sanitized)
    if not os.path.exists(out_path):
        missing.append((cid, name, sanitized))

if not missing:
    print("No missing Step 1 files.\n")
else:
    print(f"{len(missing)} missing files:\n")
    for cid, name, sanitized in missing:
        print(f"Missing: {cid} :: {name}  --> {sanitized}")
    print()
