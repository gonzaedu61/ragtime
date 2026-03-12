# embedding_backends.py

from typing import List, Protocol


# ------------------------------------------------------------
# Embedding Backend Interface (Protocol)
# ------------------------------------------------------------

class EmbeddingBackend(Protocol):
    """
    A protocol defining the interface for embedding backends.
    Any embedding backend must implement the embed() method.
    """
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...


# ------------------------------------------------------------
# HuggingFace / SentenceTransformers Backend
# ------------------------------------------------------------

class HFEmbeddingBackend:
    """
    Embedding backend using SentenceTransformers models.
    Supports local multilingual models such as:
    - intfloat/multilingual-e5-large
    - paraphrase-multilingual-mpnet-base-v2
    - BAAI/bge-m3
    """
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu", local_files_only=False)

    def embed(self, texts: List[str], progress_bar: bool = True) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=progress_bar,
            convert_to_numpy=False
        )

        # Convert PyTorch tensors → Python lists
        return [emb.tolist() for emb in embeddings]



# ------------------------------------------------------------
# OpenAI / Azure OpenAI Backend (Optional)
# ------------------------------------------------------------

class OpenAIEmbeddingBackend:
    """
    Embedding backend for OpenAI or Azure OpenAI.
    Useful if you later want to switch to cloud embeddings.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        from openai import OpenAI
        client = OpenAI()

        response = client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]
