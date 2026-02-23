import os, json

STEP1 = r"C:\Users\gonza\OneDrive\Desktop\Python\ragtime\DATA\KBs\Test_KB/7_Ontology_Files\Foundation\Entities\Step_1"

bad = []
for f in os.listdir(STEP1):
    if f.endswith("_step1.json"):
        try:
            json.load(open(os.path.join(STEP1, f), "r", encoding="utf-8"))
        except:
            bad.append(f)

print("Invalid JSON files:", len(bad))
for f in bad[:20]:
    print(" -", f)
