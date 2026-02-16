from PDF_Chunker import PDF_Chunker

KB_NAME = "Test_KB"

chunker = PDF_Chunker( kb_name=KB_NAME, export_spans=False, verbose=True)
chunker.parse()

