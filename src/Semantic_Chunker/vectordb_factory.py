from typing import Literal
from .vectordb_backends import ChromaBackend, VectorDBBackend

def create_vectordb(
    backend: Literal["chroma"],
    collection_name: str,
    persist_dir: str = "./chroma_store"
) -> VectorDBBackend:

    if backend == "chroma":
        return ChromaBackend(collection_name=collection_name, persist_dir=persist_dir)

    raise ValueError(f"Unknown vector DB backend: {backend}")
