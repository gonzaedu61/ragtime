import os
from VectorDB_Factory import create_vectordb
from LLM_Factory import create_llm
from Cluster_Info_Extractor import Cluster_Info_Extractor
from dotenv import load_dotenv
load_dotenv()

# Config Constants
KB_NAME = os.getenv("KB_NAME")
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
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)

# Prompt template (must contain {text})
BRANCH_ID = "0.0.1.18"
INFO_TYPE = 'B_Context'
LEAF_PROMPT = ""
INTERNAL_PROMPT = """
You are analyzing a list of json items, each one of them describing a business context.

Each input business context item in the list has this information:
- Process Name (The process name to which the business context belongs. It may be null)
- Context Label (A summary label for the business context)
- Business Context (a narrative description of the business context)
- JSON item structure:
{
  "process_name": "...",
  "context_label": "...",
  "business_context": "..."
}

AGREGATION TASK:
- Pick each child business context description and merge them all, generating a new single summary business context description (8-20 sentences)

OUTPUT TASK: Return a json for the new aggregated business context description and from this description a short context_label (max. 6 words), with the following json format:
{ 
  "context_label": "..."
  "business_context": "..."
}

OUTPUT LANGUAGE: German

INPUT:
{json_list}

OTHER FORMAT RULES:
- Respond ONLY with valid JSON.
- Do NOT output multiple JSON objects.
- Do NOT output text before or after the JSON.
- The FIRST character must be '{'. The LAST must be '}'.
- Return a single JSON for the process.

"""


extractor = Cluster_Info_Extractor( llm,
                                    vectordb, 
                                    LEAF_PROMPT,
                                    INTERNAL_PROMPT,
                                    info_type = INFO_TYPE,
                                    output_folder = TOPICS_PATH + "/Clusters",
                                    verbose=True,
                                    show_progress_bar=True,                                   
                                    max_concurrent_llm_calls = 1,
                                    log_prompts = True,
                                    branch_id = BRANCH_ID )


extractor.process_hierarchy_file(TOPICS_PATH + '/' + TOPICS_FILE)

