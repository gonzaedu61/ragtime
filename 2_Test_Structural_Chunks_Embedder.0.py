import sys
from Embedders import Structural_Chunks_Embedder, HFEmbeddingBackend
from VectorDB_Factory import create_vectordb
import os
from dotenv import load_dotenv
load_dotenv()

# Config Constants
KB_NAME = os.getenv("KB_NAME")
VECTOR_DB_NAME = "chroma"
COLLECTION_NAME="Structural_Chunks"
VDB_PATH = f"./DATA/KBs/{KB_NAME}/5_Vector_DB"
CHUNKS_PATH = f"./DATA/KBs/{KB_NAME}/2_Structural_Chunks"
CHUNK_FILES_PATTERN = "*_chunks.json"


# Initialize embedding_backend
print('Creating embedding_backend ...')
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "C:/Models"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DISABLE_TRANSFORMERS_AVX_CHECK"] = "1"
embedding_backend = HFEmbeddingBackend("C:/Models/multilingual-e5-large/")
print('Embedder created ...')


# Initialize vector DB backend (Chroma or others)
print('Initializing vector DB ...')
vectordb = create_vectordb(
    backend=VECTOR_DB_NAME,
    collection_name=COLLECTION_NAME,
    persist_dir=VDB_PATH
)
print('Vector DB Object created ...')


# Initialize Embedder
print('Creating Embedder ...')
embedder = Structural_Chunks_Embedder(CHUNKS_PATH, CHUNK_FILES_PATTERN,embedding_backend, vectordb, COLLECTION_NAME, verbose=True)
print('Embedder created ...')


# Embed and store chunks
#print('About to embed and store chunks ...')
chunks = embedder.embed_and_store()
#print(f'Embedded and stored {chunks} chunks')



