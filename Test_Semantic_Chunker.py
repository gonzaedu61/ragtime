import sys
from Semantic_Chunker import Semantic_Chunker
#from .embedding_pipeline import EmbeddingPipeline
from Semantic_Chunker import HFEmbeddingBackend

import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "C:/Models"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DISABLE_TRANSFORMERS_AVX_CHECK"] = "1"


print('Creating embedder ...')
embedder = HFEmbeddingBackend("C:/Models/multilingual-e5-large")
print('Embedder created ...')





json_path="./DATA/PDFs/chunks/auf_pos_vor103.pdf_chunks.json"
vectordbName = "chroma"
collection_name="default_collection"
persist_dir="./DATA/chroma_store"

print('Creating Semantic_Chunker ...')
semantic_chunker = Semantic_Chunker(json_path, embedder, vectordbName, collection_name, persist_dir)
print('Semantic_Chunker created ...')

print('About to embed and store chunks ...')
chunks = semantic_chunker.embed_and_store()
print(f'Embedded and stored {chunks} chunks')

sys.exit()


#pipeline = EmbeddingPipeline(embedder, vectordb)
#pipeline.process_chunks(chunks)


