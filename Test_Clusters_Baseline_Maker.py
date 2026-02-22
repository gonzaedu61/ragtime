import os
from VectorDB_Factory import create_vectordb
from LLM_Factory import create_llm
from dotenv import load_dotenv
from Ontology_Builder import Clusters_Baseline_Maker

load_dotenv()


# Config Constants
KB_NAME = "Test_KB"
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"


ONTOLOGY_DIR = "7_Ontology_Files"
ONTOLOGY_PATH = f"./DATA/KBs/{KB_NAME}/{ONTOLOGY_DIR}"
ENRICHED_CLUSTERS_FILE = f"{ONTOLOGY_PATH}/Enriched_Clusters.json"
CLUSTERS_BASELINE_PATH = f"{ONTOLOGY_PATH}/Clusters_Baseline"


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

os.makedirs(CLUSTERS_BASELINE_PATH, exist_ok=True)

baseline_maker = Clusters_Baseline_Maker(ENRICHED_CLUSTERS_FILE, CLUSTERS_BASELINE_PATH, llm, vectordb, num_threads=4)
baseline_maker.run()