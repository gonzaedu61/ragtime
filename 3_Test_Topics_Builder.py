import os
from Topic_Hierarchy_Builder import Topic_Hierarchy_Builder
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


topics_builder = Topic_Hierarchy_Builder(
    vector_db=vector_db,
    metadata_keys=["heading_path","document_name"],
    metadata_weight=0.2,
    postprocess_rules=None,
    verbose=False
)

hierarchy = topics_builder.build(minimal=True)



os.makedirs(TOPICS_PATH, exist_ok=True)
topics_builder.save(hierarchy, filename=TOPICS_PATH + '/' + TOPICS_FILE)

#topics_builder.print_tree(hierarchy)
