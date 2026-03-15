import sys
from Embedders import HFEmbeddingBackend
from VectorDB_Factory import create_vectordb
from Ontology_Builder import Hierarchy_Embedder
import os
from dotenv import load_dotenv
load_dotenv()


# Config Constants
KB_NAME = os.getenv("KB_NAME")
VECTOR_DB_NAME = "chroma"
CHUNKS_COLLECTION_NAME="Structural_Chunks"
CLUSTERS_COLLECTION_NAME="Clusters"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"

CLUSTERS_DIR = f"./DATA/KBs/{KB_NAME}/6_Topics_Hierarchy/clusters"
HIERARCHY_FILE = f"./DATA/KBs/{KB_NAME}/6_Topics_Hierarchy/Labeled_Topics_Hierarchy.json"
BRANCH_ID = None

# Initialize embedding_backend
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "C:/Models"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DISABLE_TRANSFORMERS_AVX_CHECK"] = "1"
embedding_backend = HFEmbeddingBackend("C:/Models/multilingual-e5-large/")


# Initialize chunks vector DB backend (Chroma or others)
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=CHUNKS_COLLECTION_NAME,
    persist_dir=VDB_PATH
)

# Initialize clusters vector DB backend (Chroma or others)
clusters_vdb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=CLUSTERS_COLLECTION_NAME,
    persist_dir=VDB_PATH
)



# Hierarchy Embedder
hierarchy_embedder = Hierarchy_Embedder(
    vector_db=vectordb,
    clusters_vdb=clusters_vdb,
    embedder=embedding_backend,
    category_base_dir = CLUSTERS_DIR,
    verbose = True,
    progress_bar = True,
    language="DE",
    save_cluster_files = True,
)


hierarchy_embedder.process(HIERARCHY_FILE, BRANCH_ID)


