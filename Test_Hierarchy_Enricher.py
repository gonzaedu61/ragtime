import sys
from VectorDB_Factory import create_vectordb
from Ontology_Builder import Hierarchy_Enricher
import os

# Config Constants
KB_NAME = "Test_KB"
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"


ONTOLOGY_DIR = "7_Ontology_Files"
ONTOLOGY_PATH = f"./DATA/KBs/{KB_NAME}/{ONTOLOGY_DIR}"
INPUT_CLUSTERS_FILE = f"./DATA/KBs/{KB_NAME}/6_Topics_Hierarchy/Topics_Hierarchy.json"
RETRIEVED_CHUNKS_FILE = f"{ONTOLOGY_PATH}/Cluster_Retrieved_Chunks.json"
ENRICHED_CLUSTERS_FILE = f"{ONTOLOGY_PATH}/Enriched_Clusters.json"


# Initialize vector DB backend (Chroma or others)
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)


enricher = Hierarchy_Enricher(
    vdb_client=vectordb,
    #verbose=True,          
    #progress_bar=True    
)


enricher.enrich(
    hierarchy_path=INPUT_CLUSTERS_FILE,
    retrieved_path=RETRIEVED_CHUNKS_FILE,
    output_path=ENRICHED_CLUSTERS_FILE
)

