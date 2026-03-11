
import os
from LLM_Factory import create_llm
from Word_Doc_Builder import Docx_Rewriter
from dotenv import load_dotenv
load_dotenv()

# Define KB
KB_NAME = os.getenv("KB_NAME")
DOCUMENT = "auf_pos"

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

# Define Folders & Filenames
TARGET_DIR = f"./DATA/KBs/{KB_NAME}/0_Word_Docs"

SOURCE_FILENAME = f"{DOCUMENT}.docx"
OUTPUT_FILENAME = f"{DOCUMENT}_(rewritten).docx"

PROMPT_TEMPLATE = """
ROLE: You are the writer of ERP Training documents
TASK: You have to revisit the text of an existing training document written in German to make it more understandable
CONTEXT: The document is part of an Industrial-Printing ERP User Manual
RULES:
  - The document text is provided in small chunks one-by-one
  - The provided text below here in the prompt is the current text chunk to re-write.
  - This text is of type {text_type}
  - Rephrase this given German text chunk into a new more user friendly German text
  - There is no need to always rewrite. If the text is short and already understandable just return it as it is.
  - Stick to the given text scope. Do not add any additional information if not explicitly required.
  - Only allowed addition: If the text_type is "paragraph", when finding a too technical, jargon, or system specific term, please add a more friendly explanation of what it is. Do not replace the term. Just complement it with an explanation/description.

TEXT:
{text}
"""


rewriter = Docx_Rewriter(TARGET_DIR,
                         SOURCE_FILENAME,
                         OUTPUT_FILENAME,
                         PROMPT_TEMPLATE,
                         llm,
                         log_prompts=True,
                         progress_enabled=True)

rewriter.rewrite()






