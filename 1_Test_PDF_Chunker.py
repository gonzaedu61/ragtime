import os
from PDF_Chunker import PDF_Chunker
from dotenv import load_dotenv
load_dotenv()


# Config Constants
KB_NAME = os.getenv("KB_NAME")

chunker = PDF_Chunker( kb_name=KB_NAME, export_spans=False, verbose=True)
chunker.parse()

