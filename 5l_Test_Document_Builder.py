import os
import sys
from Word_Doc_Builder.Word_Doc_Builder import WordDocBuilder
from dotenv import load_dotenv
load_dotenv()

KB_NAME = os.getenv("KB_NAME")
TOPICS_PATH = f"./DATA/KBs/{KB_NAME}/6_Topics_Hierarchy"
TOPICS_FILE = "Topics_Hierarchy.json" 
TREE_PATHNAME = f"{TOPICS_PATH}/{TOPICS_FILE}"
BRANCH_ID = "0.0.0.6.0.0"
WORD_TEMPLATE = "./src/Word_DOC_Builder/Word_Doc_Process_Template.docx"

"""
from docxtpl import DocxTemplate
import json

with open(f"{TOPICS_PATH}/0.0.0.6.0.0/0.0.0.6.0.0_word_doc.json", "r", encoding="utf-8") as f:
    context = json.load(f)

doc = DocxTemplate(WORD_TEMPLATE)
doc.render(context)
print("Template OK")

sys.exit()
"""


docBuilder = WordDocBuilder(TOPICS_PATH, TREE_PATHNAME, BRANCH_ID, show_progress_bar=True, log_json=True,
                            word_template_path = WORD_TEMPLATE,
                            enable_word_generation=True)

docBuilder.generate_word()


