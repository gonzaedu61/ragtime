
import os
from VectorDB_Factory import create_vectordb
from LLM_Factory import create_llm
from Cluster_Info_Extractor import Cluster_Process_Extractor
from dotenv import load_dotenv
load_dotenv()

# Define KB
KB_NAME = os.getenv("KB_NAME")
BRANCH_ID = "0.0.0.4.1.0"

# Initialize LLM
LLM_BACKEND = "azure"
LLM_NAME = "o4-mini"
LLM_DEPLOYMENT = "o4-mini"
LLM_API_VERSION = "2024-12-01-preview"
API_KEY = os.getenv("AZURE_AI_PROJECT_API_KEY")
END_POINT = "https://ragtime-openai.openai.azure.com/"
llm = create_llm(
    backend=LLM_BACKEND,
    endpoint=END_POINT,
    api_key=API_KEY,
    deployment=LLM_DEPLOYMENT,
    model_name=LLM_NAME,
    api_version=LLM_API_VERSION
)

# Initialize vector DB backend (Chroma or others)
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)


# Define Folders & Filenames
TOPICS_PATH = f"./DATA/KBs/{KB_NAME}/6_Topics_Hierarchy/Clusters"
TOPICS_FILE = "Topics_Hierarchy.json" 


processor = Cluster_Process_Extractor(
    llm=llm,
    vectordb=vectordb,
    output_folder=TOPICS_PATH,
    verbose=True,
    show_progress_bar=True,
    max_concurrent_llm_calls=1,
    log_prompts=True,
    branch_id=BRANCH_ID
)

processor.process_hierarchy(TOPICS_PATH + '/' + TOPICS_FILE)
