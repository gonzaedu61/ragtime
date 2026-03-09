from VectorDB_Factory import create_vectordb

import os
from dotenv import load_dotenv
load_dotenv()

# Config Constants
KB_NAME = os.getenv("KB_NAME")
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"


# Initialize vector DB backend (Chroma or others)
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)

# Dump records from vector DB
all_docs = vectordb.get()
for i in range(len(all_docs["ids"])):
    print("\n--- Document", i+1, "---")
    print("ID:", all_docs["ids"][i])
    print("Metadata:", all_docs["metadatas"][i])
    print("Text:", all_docs["documents"][i])

