from VectorDB_Factory.VectorDB_Factory import create_vectordb

import os
import argparse
from dotenv import load_dotenv
load_dotenv()

# CLI arguments
parser = argparse.ArgumentParser(description="Count records in a vector DB collection")
parser.add_argument("--collection", type=str, default="Structural_Chunks",
                    help="Name of the VDB collection to inspect")
args = parser.parse_args()

# Config Constants
KB_NAME = os.getenv("KB_NAME")
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME = args.collection
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"

print(f"📊 Counting records in collection: {COLLECTION_NAME}")

# Initialize vector DB backend
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)

# Retrieve all IDs (limit=None means full collection)
all_docs = vectordb.get(limit=None)

# Count
total = len(all_docs["ids"])

print(f"Total records: {total}")
