import os
from VectorDB_Factory import create_vectordb
from Topic_Hierarchy_Labeler import Topic_Hierarchy_Labeler
from LLM_Factory import create_llm
from dotenv import load_dotenv

load_dotenv()


# Config Constants
KB_NAME = "Test_KB"
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"

TOPICS_PATH = f"./DATA/KBs/{KB_NAME}/6_Topics_Hierarchy"
TOPICS_FILE = "Topics_Hierarchy.json" 
LABELED_TOPICS_FILE = "Labeled_Topics_Hierarchy.json"
TOPICS_CACHE=TOPICS_PATH + '/' + "hierarchy_label_cache.json"



LLM_BACKEND = "azure"
LLM_NAME = "o4-mini"
LLM_DEPLOYMENT = "o4-mini"
LLM_API_VERSION = "2024-12-01-preview"
API_KEY = os.getenv("AZURE_AI_PROJECT_API_KEY")
END_POINT = "https://ragtime-openai.openai.azure.com/"

# Initialize LLM
llm = create_llm(
    backend=LLM_BACKEND,
    endpoint=END_POINT,
    api_key=API_KEY,
    deployment=LLM_DEPLOYMENT,
    model_name=LLM_NAME,
    api_version=LLM_API_VERSION
)


# Initialize vector DB backend (Chroma or others)
print('Initializing vector DB ...')
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)
print('Vector DB Object created ...')



labeler = Topic_Hierarchy_Labeler(llm, vectordb, store_summaries=True, verbose=True, cache_path=TOPICS_CACHE)

labeler.label_hierarchy_file( input_json_path=TOPICS_PATH + '/' + TOPICS_FILE,
                              output_json_path=TOPICS_PATH + '/' + LABELED_TOPICS_FILE )

