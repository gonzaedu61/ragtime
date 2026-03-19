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
BRANCH_ID = "0.0.0.0.21.0"
INFO_TYPE = 'WHAT'
INFO_TYPE_INPUT = "questions"
LEAF_PROMPT = """
You are analyzing a set of text chunks and an input json structure all belonging to the same topic. The json file contains a field named "WHAT" with a list of questions related to the topic.

TASK:
- For each question in the "WHAT" field list generate a 4-8 statements answer using the provided texts as the knowledge source.
- Add also a little business context introduction taking the knowledge from the same set of texts and from other internet sources related to the Industrial Printing Industry.
- Do not create or infere facts. Use as knoeldge sources only the given texts and relevant and trustable internet sources.

Return a JSON object with the list of step details.   

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

{ "WHAT_Answers":
  [
    { 
      "question": <original question>,
      "answer": "..."
    }
  ]
}

"""


INTERNAL_PROMPT = None


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

