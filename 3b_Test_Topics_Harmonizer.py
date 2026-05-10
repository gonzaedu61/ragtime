import os
from Topic_Hierarchy_Harmonizer import Topic_Hierarchy_Harmonizer
from VectorDB_Factory import create_vectordb
from dotenv import load_dotenv
load_dotenv()

# Config Constants
KB_NAME = os.getenv("KB_NAME")
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"

TOPICS_PATH = f"./DATA/KBs/{KB_NAME}/6_Topics_Hierarchy"
TOPICS_FILE = "Topics_Hierarchy.json" 


vector_db = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)

MAX_TOKENS = 5000
MIN_TOKENS = 1500


harmonizer = Topic_Hierarchy_Harmonizer(
    vector_db,
    TOPICS_PATH + '/' + TOPICS_FILE,
    MAX_TOKENS,
    MIN_TOKENS,
    TOPICS_PATH,
    verbose = True
)

harmonizer.run()



