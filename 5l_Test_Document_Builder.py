import os
import sys
from Word_Doc_Builder.Word_Doc_Builder import WordDocBuilder
from dotenv import load_dotenv
load_dotenv()

KB_NAME = os.getenv("KB_NAME")
TOPICS_PATH = f"./DATA/KBs/{KB_NAME}/6_Topics_Hierarchy"
TOPICS_FILE = "Topics_Hierarchy.json" 
TREE_PATHNAME = f"{TOPICS_PATH}/{TOPICS_FILE}"
BRANCH_ID = "0.0.0.3.0.1"
WORD_TEMPLATE = "./src/Word_DOC_Builder/Word_Doc_Process_Template (DE).docx"

docBuilder = WordDocBuilder(TOPICS_PATH, TREE_PATHNAME, BRANCH_ID, show_progress_bar=True, log_json=True,
                            word_template_path = WORD_TEMPLATE,
                            enable_word_generation=True,
                            use_existing_json=False)

docBuilder.generate_word()


