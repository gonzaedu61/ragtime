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
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids: List[str], texts: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadata
        )

    def delete(self, where: Dict[str, Any]):
        """
        Delete all entries matching a metadata filter.
        Supports boolean expressions like $and, $or, $not.
        """
        # Fetch all matching items
        existing = self.collection.get(where=where, include=[])

        ids_to_delete = existing.get("ids", [])

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)
    

    def upsert(self, ids, texts, embeddings, metadata, scope: Dict[str, Any]):

        # Ensure input IDs are unique
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs found in input batch.")

        # 1. Delete all chunks for this document
        self.delete(scope)

        # 2. Try inserting new chunks
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadata
            )

        # 3. Fallback: handle cross-document duplicate IDs
        except Exception as e:
            if "DuplicateIDError" in str(e):
                # Fetch all existing IDs
                existing = self.collection.get(include=[])
                existing_ids = set(existing.get("ids", []))

                # Find which new IDs already exist
                ids_to_delete = list(existing_ids.intersection(ids))

                # Delete only those
                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)

                # Retry insertion
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadata
                )
            else:
                raise



    def get(self):
        return self.collection.get(
            include=["documents", "embeddings", "metadatas"]
        )

    def query(self, query_embeddings, n_results=5):
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )


    def get_all(self):
        data = self.collection.get(
            include=["documents", "embeddings", "metadatas"]
        )
        return (
            data["embeddings"],
            data["ids"],
            data["documents"],
            data["metadatas"]
        )
    
    def get_by_id(self, chunk_id: str):
        result = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        if not result.get("documents"):
            return None
        return {
            "id": chunk_id,
            "document": result["documents"][0],
            "metadata": result["metadatas"][0],
        }


