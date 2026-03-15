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



    def get(self, limit: int = None):
        """
        Retrieve up to `limit` items from the collection.
        If limit is None, return all items.
        """
        return self.collection.get(
            include=["documents", "embeddings", "metadatas"],
            limit=limit
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
    
    def search(self, embedding, top_n=30, filter_ids=None):
        """
        Extended search() supporting optional ID-restricted search.
        If filter_ids is None → normal Chroma vector search (fast).
        If filter_ids is provided → manual similarity search over those IDs.
        """

        # ------------------------------------------------------------
        # FAST PATH: Normal vector search
        # ------------------------------------------------------------
        if filter_ids is None:
            result = self.query(query_embeddings=[embedding], n_results=top_n)
            return [
                {
                    "id": result["ids"][0][i],
                    "text": result["documents"][0][i],
                    "score": 1 - result["distances"][0][i],
                    "metadata": result["metadatas"][0][i],
                }
                for i in range(len(result["ids"][0]))
            ]

        # ------------------------------------------------------------
        # FILTERED PATH
        # ------------------------------------------------------------
        import numpy as np

        # 1) Deduplicate filter_ids while preserving order
        seen = set()
        unique_ids = []
        for cid in filter_ids:
            if cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)

        # 2) Fetch embeddings for all requested IDs
        embeddings = []
        metadatas = []
        texts = []
        valid_ids = []
        seen = set()  # dedupe again in case Chroma returns duplicates

        for cid in unique_ids:
            if cid in seen:
                continue
            seen.add(cid)

            item = self.collection.get(
                ids=[cid],
                include=["embeddings", "documents", "metadatas"]
            )

            emb = item.get("embeddings")
            docs = item.get("documents")
            meta = item.get("metadatas")


            valid_ids.append(cid)
            embeddings.append(np.array(emb[0], dtype=np.float32))
            texts.append(docs[0])
            metadatas.append(meta[0])

        if not embeddings:
            return []

        # 3) Compute cosine similarity manually
        query_vec = np.array(embedding, dtype=np.float32)
        chunk_matrix = np.vstack(embeddings)

        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        c_norm = chunk_matrix / (np.linalg.norm(chunk_matrix, axis=1, keepdims=True) + 1e-9)

        sims = np.dot(c_norm, q_norm)

        # 4) Sort by similarity
        top_idx = np.argsort(sims)[::-1][:top_n]

        # 5) Build final results with deduplication
        results = []
        seen_ids = set()

        for i in top_idx:
            cid = valid_ids[i]
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            results.append(
                {
                    "id": cid,
                    "text": texts[i],
                    "score": float(sims[i]),
                    "metadata": metadatas[i],
                }
            )

        return results
    

    def all_chunks(self):
        """
        Returns all chunks in a unified format expected by Ontology_Chunks_Retriever.
        """
        embeddings, ids, documents, metadatas = self.get_all()

        chunks = []
        for i in range(len(ids)):
            chunks.append({
                "chunk_id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
            })

        return chunks


    def get_for_clustering(self, metadata_keys):
        """
        Returns only the data needed for clustering:
            - embeddings
            - ids
            - filtered metadatas (only metadata_keys)
        Documents are NOT returned.
        """

        # Fetch only embeddings + metadatas + ids
        data = self.collection.get(
            include=["embeddings", "metadatas"]
        )

        embeddings = data["embeddings"]
        ids = data["ids"]
        metadatas = data["metadatas"]

        # Filter metadata to only the keys needed for clustering
        filtered_metadatas = []
        for meta in metadatas:
            filtered = {k: meta.get(k) for k in metadata_keys}
            filtered_metadatas.append(filtered)

        return embeddings, ids, filtered_metadatas



    def get_embedding(self, chunk_id: str):
        """
        Return the embedding vector for a given chunk_id.
        """
        result = self.collection.get(
            ids=[chunk_id],
            include=["embeddings"]
        )

        embs = result.get("embeddings")
        if embs is None or len(embs) == 0:
            return None

        return embs[0]



