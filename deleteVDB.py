#!/usr/bin/env python3

import os
import argparse
from dotenv import load_dotenv
load_dotenv()

import chromadb

# ------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Delete a ChromaDB collection")
parser.add_argument(
    "--collection",
    type=str,
    required=True,
    help="Name of the Chroma collection to delete"
)
args = parser.parse_args()

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
KB_NAME = os.getenv("KB_NAME")
VECTOR_DB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"

print(f"🗑️  Attempting to delete collection: {args.collection}")
print(f"📁 Chroma path: {VECTOR_DB_PATH}")

# ------------------------------------------------------------
# Connect to Chroma persistent client
# ------------------------------------------------------------
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

# ------------------------------------------------------------
# Check if collection exists
# ------------------------------------------------------------
existing_collections = [c.name for c in client.list_collections()]

if args.collection not in existing_collections:
    print(f"⚠️  Collection '{args.collection}' does not exist.")
    exit(0)

# ------------------------------------------------------------
# Delete the collection
# ------------------------------------------------------------
client.delete_collection(name=args.collection)

print(f"✅ Collection '{args.collection}' has been deleted successfully.")
