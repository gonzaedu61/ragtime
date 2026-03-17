import os
from VectorDB_Factory import create_vectordb
from LLM_Factory import create_llm
from Cluster_Info_Extractor import Cluster_Info_Extender
from Embedders import HFEmbeddingBackend
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

# Initialize embedding_backend
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "C:/Models"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DISABLE_TRANSFORMERS_AVX_CHECK"] = "1"
embedder = HFEmbeddingBackend("C:/Models/multilingual-e5-large/")


# Initialize vector DB backend (Chroma or others)
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)

# Prompt template (must contain {text})
BRANCH_ID = "0.0.0.0.0.0"
INFO_TYPE = 'steps'
INFO_TYPE_INPUT = "process_b"
LEAF_PROMPT = None
INTERNAL_PROMPT = """
You are analyzing a set of text chunks and an input json structure all belonging to the same topic. The json file contains a field named "process_steps" with a list of steps of the process described in the field "process_description".

TASK:
- For each step and based on the explanations in the given texts, generate a detailed explanation of the step, writing what needs to be done by the process actors and any other details available.
- Add also an explanation of what the objective of the process step is. What it exists and what the aim to achive is.
- The process actors are mainly the system user and the system itself. But include others actors if they exist and play a role in the process step.
- Focus specially (when the information is available), on the detailed interaction between the user and the system (i.e. what system modeule to use, what menu items or buttons to select or click, how to provide input and how to see the results, ...)
- Include a little narrative about the step context (the context in which this step is executed)
- Optionally, indicate pre- and post- conditions of this process step when applicable and if the information is available. Do not create or infere facts. If no conditions exist then leave them as ""
- Optionally, indicate exceptions and warnings when available in the input texts. Do not create or infere facts. If no warning exist then leave it as "".

Return a JSON object with the list of step details.   

TEXTS:
{extra_chunks}

INPUT JSON:
{input_json}


OUTPUT LANGUAGE: German

FORMAT RULES:
- Respond ONLY with valid JSON.
- Do NOT output text before or after the JSON.
- The FIRST character must be '{' and the LAST must be '}'.
- Return a single JSON object with this structure:

{
  "process_steps": [
    { "id": <original id>,
      "name": <original name>,
      "details": { 
        "objective": ...,
        "explanation": ...,
        "context": ..., 
        "pre-condition": ...,
        "post-condition": ...,
        "exceptions": ...,
        "warnings": ...        
      }      
    }
  ]
}

"""

extractor = Cluster_Info_Extender( llm,
                                    vectordb, 
                                    embedder,
                                    LEAF_PROMPT,
                                    INTERNAL_PROMPT,
                                    info_type = INFO_TYPE,
                                    info_type_input = INFO_TYPE_INPUT,
                                    output_folder = TOPICS_PATH,
                                    retrieve_semantic_chunks = True,
                                    top_number_of_chunks = 10,
                                    verbose=False,
                                    show_progress_bar=True,                                   
                                    max_concurrent_llm_calls = 1,
                                    log_prompts = True,
                                    branch_id = BRANCH_ID )

extractor.process_hierarchy_file(TOPICS_PATH + '/' + TOPICS_FILE)

