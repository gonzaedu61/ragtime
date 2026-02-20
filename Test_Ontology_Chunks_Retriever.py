import sys
from Embedders import HFEmbeddingBackend
from VectorDB_Factory import create_vectordb
from Ontology_Builder import Ontology_Chunks_Retriever
import os

# Config Constants
KB_NAME = "Test_KB"
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"


ONTOLOGY_DIR = "7_Ontology_Files"
ONTOLOGY_PATH = f"./DATA/KBs/{KB_NAME}/{ONTOLOGY_DIR}"
INPUT_CLUSTERS_FILE = f"./DATA/KBs/{KB_NAME}/6_Topics_Hierarchy/Labeled_Topics_Hierarchy.json"
RETRIEVED_CHUNKS_FILE = f"{ONTOLOGY_PATH}/Ontology_Retrieved_Chunks.json"
FLATTENED_CLUSTERS_FILE = f"{ONTOLOGY_PATH}/Flattened_Clusters.json"


# Initialize embedding_backend
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "C:/Models"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DISABLE_TRANSFORMERS_AVX_CHECK"] = "1"
embedding_backend = HFEmbeddingBackend("C:/Models/multilingual-e5-large/")


# Initialize vector DB backend (Chroma or others)
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)


retriever = Ontology_Chunks_Retriever(
    vector_db=vectordb,
    embedder=embedding_backend,
    language="DE",
    top_n=40,
    verbose=True,          
    progress_bar=True    
)


os.makedirs(ONTOLOGY_PATH, exist_ok=True)

retriever.retrieve(
    input_json_path=INPUT_CLUSTERS_FILE,
    output_json_path=RETRIEVED_CHUNKS_FILE,
    flattened_debug_path=FLATTENED_CLUSTERS_FILE
)

