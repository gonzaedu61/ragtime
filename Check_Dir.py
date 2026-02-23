import os, json

BASE = r"C:\Users\gonza\OneDrive\Desktop\Python\ragtime\DATA\KBs\Test_KB/7_Ontology_Files\Foundation\Entities"

print("=== CHECKING DIRECTORIES ===")
print("Entities dir:", BASE)
print("Step_1 exists:", os.path.exists(os.path.join(BASE, "Step_1")))
print("Step_2 exists:", os.path.exists(os.path.join(BASE, "Step_2")))

print("\n=== STEP 1 FILE COUNT ===")
step1 = os.path.join(BASE, "Step_1")
files1 = [f for f in os.listdir(step1) if f.endswith("_step1.json")]
print("Step_1 files:", len(files1))

print("\n=== STEP 2 FILE COUNT ===")
step2 = os.path.join(BASE, "Step_2")
files2 = [f for f in os.listdir(step2) if f.endswith("_step2.json")]
print("Step_2 files:", len(files2))

print("\n=== SAMPLE STEP 1 FILENAMES ===")
print(files1[:10])

print("\n=== SAMPLE STEP 2 FILENAMES ===")
print(files2[:10])
