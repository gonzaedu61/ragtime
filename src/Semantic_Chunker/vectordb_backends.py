# vectordb_backends.py

from typing import List, Dict, Any, Protocol


# ------------------------------------------------------------
# Vector DB Backend Interface (Protocol)
# ------------------------------------------------------------

class VectorDBBackend(Protocol):
    """
    A protocol defining the interface for vector database backends.
    Any backend must implement add() and query().
    """
    def add(self, ids: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        ...

    def query(self, query_embedding: List[float], top_k: int):
        ...


# ------------------------------------------------------------
# ChromaDB Backend (Local DuckDB + Parquet)
# ------------------------------------------------------------

class ChromaBackend:
    """
    Local ChromaDB backend using DuckDB + Parquet.
    Perfect for prototyping and local semantic search.
    """

    def __init__(self, collection_name: str, persist_dir: str = "./chroma_store"):

        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Create or load the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # cosine similarity is ideal for embeddings
        )

    def add(self, ids: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """
        Store embeddings + metadata in Chroma.
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata
        )

    def query(self, query_embedding: List[float], top_k: int):
        """
        Perform a semantic search in the vector DB.
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
