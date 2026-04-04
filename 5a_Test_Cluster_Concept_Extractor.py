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
BRANCH_ID = "0.0.1.7.8"
INFO_TYPE = 'concept'
LEAF_PROMPT = """
You are analyzing a set of text chunks that belong to the same topic.

TASK:
Identify and describe the key concepts covered by the input text, with the following information:
- Concept name: A short label (max 6 words). YOU MUST ALWAYS PROVIDE A LABEL.
- Concept description: A 4-8 sentences explaining what the key elements of the concept are and how they relate to each other
- Concept structure: Extract the structural elements which make up the concept and a 3-6 sentences description of what they are and the relationships which might exist between them (i.e. X has 1 or more Y, Z is included in W, Q is part of H, etc.)
- IMPORTANT: If the text does not seem to refer to a business concept, then DO NOT INFERE or CREATE FACTS. Just return all field blanks

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
  "concept_name": "...",
  "concept_description": "...",
  "concept_structure": [{concept_element_name, description}, ... ],
}

"""


INTERNAL_PROMPT = """
You are analyzing a list of json items, each one of them describing a concept.

Each input concept item in the list has this information:
- Concept name
- Concept description
- Concept structure (a list of the key elements supporting the concept)
- JSON item structure:
{
  "concept_name": "...",
  "concept_description": "...",
  "concept_structure": [{concept_element_name, description}, ... ],
}

AGREGATION TASK:
- For each child concept item, pick the concept_structure and agregate them into one single concept_structure
- The collection of the resulting aggregated concept_structure from each child concept will become the list of concept_structures for a new aggregated concept (the parent concept) encompasing them.

OUTPUT TASK: Produce a json for the newly created parent concept with the following information:
- Concept description: A 4-8 sentences explaining what this parent concept is about, from the combination of the child concept descriptions
- Concept name: A short label (max 6 words). YOU MUST ALWAYS PROVIDE A LABEL based on the generated description.
- Concept structure: An ordered list of the key structure elements composing the concept (the aggregated structural elements of each input child concept). Give them a sequential numeric id (the list index), a label and a 3-6 sentences description.

OUTPUT LANGUAGE: German

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
  "concept_name": "...",
  "concept_description": "...",
  "concept_structure": [{concept_element_name, description}, ... ],
}

"""


extractor = Cluster_Info_Extractor( llm,
                                    vectordb, 
                                    LEAF_PROMPT,
                                    INTERNAL_PROMPT,
                                    info_type = INFO_TYPE,
                                    output_folder = TOPICS_PATH,
                                    verbose=True,
                                    show_progress_bar=True,                                   
                                    max_concurrent_llm_calls = 1,
                                    log_prompts = True,
                                    branch_id = BRANCH_ID )


extractor.process_hierarchy_file(TOPICS_PATH + '/' + TOPICS_FILE)

