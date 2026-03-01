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
BRANCH_ID = "0.0.0.6"
INFO_TYPE = 'BO'
LEAF_PROMPT = """
You are analyzing a set of text chunks that belong to the same topic.

TASK:
Identify and describe the Business Objects (BO) present in the input text.  
Return a JSON object containing a list of BOs.  
Each BO must include:
- bo_name: A short label (max 6 words).
- bo_description: A 4–8 sentence explanation of the BO.
- bo_transitions: A list of status transitions.  
  Each transition must be a JSON object with:
    - state_1: previous state (string)
    - event_label: short label (max 3 words)
    - event_description: 1–2 line description
    - state_2: resulting state (string)

BUSINESS OBJECT RECOGNITION GUIDES:
- A BO represents a coherent set of business data with a clear lifecycle.
- A BO often corresponds to a business document passed through processes.
- Business processes trigger events that change BO status.

TEXTS:
{text}

OUTPUT LANGUAGE: German

FORMAT RULES:
- Respond ONLY with valid JSON.
- Do NOT output text before or after the JSON.
- The FIRST character must be '{' and the LAST must be '}'.
- Return a single JSON object with this structure:

{
  "business_objects": [
    {
      "bo_name": "...",
      "bo_description": "...",
      "bo_transitions": [
        {
          "state_1": "...",
          "event_label": "...",
          "event_description": "...",
          "state_2": "..."
        }
      ]
    }
  ]
}

"""


INTERNAL_PROMPT = """
You are analyzing a list of Business Object (BO) JSON items.

Each BO item has this structure:
{
  "bo_name": "...",
  "bo_description": "...",
  "bo_transitions": [
    {
      "state_1": "...",
      "event_label": "...",
      "event_description": "...",
      "state_2": "..."
    }
  ]
}

AGGREGATION TASK:
- Merge BOs that have the same name or represent the same conceptual BO.
- When merging, combine all transitions from all matching BOs.
- Keep only one BO entry per unique BO name.

OUTPUT TASK:
Return a single JSON object containing the aggregated list of BOs.

INPUT:
{json_list}

OUTPUT LANGUAGE: German

FORMAT RULES:
- Respond ONLY with valid JSON.
- Do NOT output text before or after the JSON.
- The FIRST character must be '{' and the LAST must be '}'.
- Return a single JSON object with this structure:

{
  "business_objects": [
    {
      "bo_name": "...",
      "bo_description": "...",
      "bo_transitions": [
        {
          "state_1": "...",
          "event_label": "...",
          "event_description": "...",
          "state_2": "..."
        }
      ]
    }
  ]
}

"""


extractor = Cluster_Info_Extractor( llm,
                                    vectordb, 
                                    LEAF_PROMPT,
                                    INTERNAL_PROMPT,
                                    info_type = INFO_TYPE,
                                    output_folder = TOPICS_PATH,
                                    store_cache = False,
                                    cache_path = TOPICS_CACHE,
                                    verbose=False,
                                    show_progress_bar=True,                                   
                                    max_concurrent_llm_calls = 1,
                                    log_prompts = True,
                                    branch_id = BRANCH_ID )


extractor.process_hierarchy_file(TOPICS_PATH + '/' + TOPICS_FILE)

