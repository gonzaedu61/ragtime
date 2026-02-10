from PDF_Chunker import PDF_Chunker

chunker = PDF_Chunker( pdf_path="DATA/PDFs",
                       document_name="auf_pos_vor103.pdf",
                       output_dir="DATA/PDFs/chunks",
                       export_spans=False,
                       export_blocks=False )

chunker.parse()

