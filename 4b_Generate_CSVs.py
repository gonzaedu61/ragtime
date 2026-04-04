import os
from Hierarchy_CSV_Generator import Hierarchy_CSV_Generator
from VectorDB_Factory import create_vectordb
from dotenv import load_dotenv
load_dotenv()

# Config Constants
KB_NAME = os.getenv("KB_NAME")
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"


HIERARCHY_PATH = f"./DATA/KBs/{KB_NAME}/6_Topics_Hierarchy"
CLUSTERS_PATH = f"{HIERARCHY_PATH}/Clusters"
LABELED_TOPICS_FILE = "Labeled_Topics_Hierarchy_DE.json"
CHUNKS_CSV = "PDF_Chunks.csv"
CLUSTERS_CSV = "KB_Clusters.csv"

# Initialize vector DB backend (Chroma or others)
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)

os.makedirs(CLUSTERS_PATH, exist_ok=True)

generator = Hierarchy_CSV_Generator(HIERARCHY_PATH + '/' + LABELED_TOPICS_FILE,
                                    CLUSTERS_PATH,
                                    vectordb,
                                    CHUNKS_CSV,
                                    CLUSTERS_CSV)
generator.generate()

