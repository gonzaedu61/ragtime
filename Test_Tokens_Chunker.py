import os
from Tokens_Chunker import Tokens_Chunker


KB_NAME = "Test_KB"
CHUNKS_PATH = f"./DATA/KBs/{KB_NAME}/2_Structural_Chunks"
CHUNK_FILES_PATTERN = "*_chunks.json"
OUTPUT_PATH = f"./DATA/KBs/{KB_NAME}/4_Tokenized_Chunks"


# Create the chunker (parameters match your Embedder style)
tokens_chunker = Tokens_Chunker(
    chunks_path=CHUNKS_PATH,
    chunk_files_pattern=CHUNK_FILES_PATTERN,
    output_path=OUTPUT_PATH,
    model_name="intfloat/multilingual-e5-large",
    min_tokens=80,
    max_tokens=384,
    overlap=64,
    use_block_proximity=False,   # set True if you want bbox-based matching
    verbose=True
)

# Process all structural chunk files
tokens_chunker.run()

