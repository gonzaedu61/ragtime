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
BRANCH_ID = "0.0.1.1"
INFO_TYPE = 'B_Context'
INFO_TYPE_INPUT = "process_b"
INTERNAL_PROMPT = """
You are analyzing a set of text chunks and an input json structure all belonging to the same topic. The json file contains information about a business process named after the field "process_name" and another field named "process_description" summarizing in narrative form the scope of  related child subprocesses.

TASK:
- Provide a comprehensive business context narrative description (8-20 sentences) based on the knowledge from the set of texts plus other internet sources related to the Industrial Printing Industry.
- Explain how this process fits into the business, what elements it deals with, what its objective is and why.
- Do not create or infere facts. Use as knoweldge sources only the given texts and some relevant and trustable internet sources.

Return a JSON object with the business context description.   

TEXTS:
{text}

INPUT JSON:
{input_json}

OUTPUT LANGUAGE: German

FORMAT RULES:
- Respond ONLY with valid JSON.
- Do NOT output text before or after the JSON.
- The FIRST character must be '{' and the LAST must be '}'.
- Return a single JSON object with this structure:

{ 
  "process_name": <original process name>,
  "business_context": "..."
}

"""
LEAF_PROMPT = INTERNAL_PROMPT


extractor = Cluster_Info_Extender( llm,
                                    vectordb, 
                                    embedder,
                                    LEAF_PROMPT,
                                    INTERNAL_PROMPT,
                                    info_type = INFO_TYPE,
                                    info_type_input = INFO_TYPE_INPUT,
                                    output_folder = TOPICS_PATH + "/Clusters",
                                    retrieve_semantic_chunks = True,
                                    top_number_of_chunks = 10,
                                    verbose=False,
                                    show_progress_bar=True,                                   
                                    max_concurrent_llm_calls = 1,
                                    log_prompts = True,
                                    branch_id = BRANCH_ID )

extractor.process_hierarchy_file(TOPICS_PATH + '/' + TOPICS_FILE)

