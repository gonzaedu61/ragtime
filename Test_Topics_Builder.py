import os
from Topic_Hierarchy_Builder import Topic_Hierarchy_Builder
from VectorDB_Factory import create_vectordb

# Config Constants
KB_NAME = "Test_KB"
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"



vector_db = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)


topics_builder = Topic_Hierarchy_Builder(
    vector_db=vector_db,
    metadata_keys=["heading_path"],
    metadata_weight=0.3,
    postprocess_rules=None,
    verbose=False
)

hierarchy = topics_builder.build()

topics_builder.save(hierarchy)

topics_builder.print_tree(hierarchy)
