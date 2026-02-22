import os
from VectorDB_Factory import create_vectordb
from LLM_Factory import create_llm
from dotenv import load_dotenv
from Ontology_Builder import Ontology_Foundation_Builder

load_dotenv()


# Config Constants
KB_NAME = "Test_KB"
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"


ONTOLOGY_DIR = "7_Ontology_Files"
ONTOLOGY_PATH = f"./DATA/KBs/{KB_NAME}/{ONTOLOGY_DIR}"
INPUT_CLUSTERS_FILE = f"./DATA/KBs/{KB_NAME}/6_Topics_Hierarchy/Topics_Hierarchy.json"
CLUSTERS_BASELINE_PATH = f"{ONTOLOGY_PATH}/Clusters_Baseline"
FOUNDATION_FILES_PATH = f"{ONTOLOGY_PATH}/Foundation"


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
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)


foundation_builder = Ontology_Foundation_Builder(INPUT_CLUSTERS_FILE,
                                                 CLUSTERS_BASELINE_PATH,
                                                 FOUNDATION_FILES_PATH,
                                                 llm, vectordb, max_workers=8,
                                                 progress_enabled=True)

foundation_builder.run()

