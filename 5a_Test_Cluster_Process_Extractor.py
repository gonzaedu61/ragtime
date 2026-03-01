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
INFO_TYPE = 'process_b'
LEAF_PROMPT = """
You are analyzing a set of text chunks that belong to the same topic.

TASK:
Identify and describe the the process covering the full input text, with the following information:
- Proces name: A short label (max 6 words). YOU MUST ALWAYS PROVIDE A LABEL.
- Process description: A 4-8 sentences explaining what the process is about
- Process steps: An ordered list of the key steps composing the process. Give them a sequential numeric id (the list index), a short label and a short one-line description.

COVERAGE RULE:
If the process is unclear, too small, or ambiguous, you MUST STILL provide the best possible descriptive label, warning about the quality of the given details. Never return an empty process. All text parts should map to at least one process

TEXTS:
{text}

OUTPUT LANGUAGE: German

FORMAT RULES:
- Respond ONLY with valid JSON.
- Do NOT output multiple JSON objects.
- Do NOT output text before or after the JSON.
- The FIRST character must be '{'. The LAST must be '}'.
- Return a single JSON for the process.
- The JSON must exactly follow this structure:
{
  "process_name": "...",
  "process_description": "...",
  "process_steps": [{id, name, description}, ... ],
}

"""


INTERNAL_PROMPT = """
You are analyzing a list of json items, each one of them describing a process.

Each input process item in the list has this information:
- Process name
- Process description
- Process steps (an ordered list of the key steps composing the process)
- JSON item structure:
{
  "process_name": "...",
  "process_description": "...",
  "process_steps": [{id, name, description}, ... ],
}

AGREGATION TASK:
- For each child process item, pick the steps and agregate them into one single step description
- The collection of the resulting aggregated steps from each child process will become the list of steps for a new aggregated process (the parent process) encompasing them.

OUTPUT TASK: Produce a json for the newly created parent process with the following information:
- Process description: A 4-8 sentences explaining what this parent process is about, from the combination of the child process descriptions
- Proces name: A short label (max 6 words). YOU MUST ALWAYS PROVIDE A LABEL based on the generated description.
- Process steps: An ordered list of the key steps composing the process (the aggregated steps of each input child process). Give them a sequential numeric id (the list index), a short label and a short one-line description.

OUTPUT LANGUAGE: German

COVERAGE RULE:
If the process is unclear, too small, or ambiguous, you MUST STILL provide the best possible descriptive label, warning about the quality of the given details. Never return an empty process. All input child processes should map to at least one parent process

INPUT:
{json_list}

FORMAT RULES:
- Respond ONLY with valid JSON.
- Do NOT output multiple JSON objects.
- Do NOT output text before or after the JSON.
- The FIRST character must be '{'. The LAST must be '}'.
- Return a single JSON for the process.
- The JSON must exactly follow this structure:
{
  "process_name": "...",
  "process_description": "...",
  "process_steps": [{id, name, description}, ... ],
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
                                    verbose=True,
                                    show_progress_bar=True,                                   
                                    max_concurrent_llm_calls = 1,
                                    log_prompts = True,
                                    branch_id = BRANCH_ID )


extractor.process_hierarchy_file(TOPICS_PATH + '/' + TOPICS_FILE)

