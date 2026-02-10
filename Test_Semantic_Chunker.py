import sys
from Semantic_Chunker import Semantic_Chunker
from Semantic_Chunker import HFEmbeddingBackend
from VectorDB_Factory import create_vectordb
import os

# Initialize embedder
print('Creating embedder ...')
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "C:/Models"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DISABLE_TRANSFORMERS_AVX_CHECK"] = "1"
embedder = HFEmbeddingBackend("C:/Models/multilingual-e5-large")
print('Embedder created ...')


# Initialize vector DB backend (Chroma or others)
print('Initializing vector DB ...')
vectordbName = "chroma"
collection_name="default_collection"
persist_dir="./DATA/chroma_store"
vectordb = create_vectordb(
    backend=vectordbName,
    collection_name=collection_name,
    persist_dir=persist_dir
)
print('Vector DB Object created ...')


# Initialize Semantic Chunker
print('Creating Semantic_Chunker ...')
json_path="./DATA/PDFs/chunks/auf_pos_vor103.pdf_chunks.json"
semantic_chunker = Semantic_Chunker(json_path, embedder, vectordb, collection_name, persist_dir)
print('Semantic_Chunker created ...')


# Embed and store chunks
print('About to embed and store chunks ...')
chunks = semantic_chunker.embed_and_store()
print(f'Embedded and stored {chunks} chunks')



