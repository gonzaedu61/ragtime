import json
import os
import re
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any

from .embedding_backends import HFEmbeddingBackend
from Utilities.File_Utilities import expand_files_pattern


# ---------------------------------------------------------
# Helper: deterministic ID
# ---------------------------------------------------------
def deterministic_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------
# Dataclass for tokenized chunks
# ---------------------------------------------------------
@dataclass
class TokenizedChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------
# Main class
# ---------------------------------------------------------
class Tokenized_Chunks_Embedder:
    """
    Tokenized chunk embedder that:
    - Uses 'merged_original_texts' as the text source
    - Cleans numeric prefixes like '12;' or '7;'
    - Merges multiple source texts with newlines
    - Produces the same output structure as Structural_Chunks_Embedder
    """

    def __init__(
        self,
        chunks_path: str,
        chunk_files_pattern: str,
        embedding_backend: HFEmbeddingBackend,
        vectordb,
        collection_name: str,
        verbose: bool = False
    ):
        self.chunks_path = chunks_path
        self.chunk_files_pattern = chunk_files_pattern
        self.embedding_backend = embedding_backend
        self.vectordb = vectordb
        self.collection_name = collection_name
        self.verbose = verbose

    # -----------------------------------------------------
    # Clean merged_original_texts
    # -----------------------------------------------------
    def _extract_text(self, value):
        """Normalize merged_original_texts into a single string."""
        if isinstance(value, str):
            return value

        if isinstance(value, list):
            return "\n".join(str(v) for v in value)

        if isinstance(value, dict):
            parts = []
            for v in value.values():
                parts.append(str(v))
            return "\n".join(parts)

        return str(value)


    def _clean_merged_text(self, raw_text: str) -> str:
        """
        Removes leading numeric references like '12;' or '7;'
        and merges lines cleanly.
        """
        clean_lines = []

        for line in raw_text.split("\n"):
            # Remove prefixes like "12;" or "7;"
            cleaned = re.sub(r"^\s*\d+\s*;\s*", "", line).strip()
            if cleaned:
                clean_lines.append(cleaned)

        return "\n".join(clean_lines)


    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata contains only Chroma‑compatible types."""
        sanitized = {}

        for key, value in metadata.items():

            # Allowed types
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
                continue

            # Lists are allowed, but must contain only allowed types
            if isinstance(value, list):
                clean_list = []
                for item in value:
                    if isinstance(item, (str, int, float, bool)) or item is None:
                        clean_list.append(item)
                    else:
                        # Convert unsupported types inside lists to strings
                        clean_list.append(str(item))
                sanitized[key] = clean_list
                continue

            # Dicts → convert to JSON string
            if isinstance(value, dict):
                sanitized[key] = json.dumps(value, ensure_ascii=False)
                continue

            # Fallback: convert everything else to string
            sanitized[key] = str(value)

        return sanitized


    # -----------------------------------------------------
    # Load chunks
    # -----------------------------------------------------
    def load_tokenized_chunks(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []

        for item in data:
            raw_value = item.get("merged_original_texts", "")
            raw_text = self._extract_text(raw_value)
            text = self._clean_merged_text(raw_text)

            image_paths = item.get("image_paths", [])

            if not text and not image_paths:
                continue

            id_source = text if text else " ".join(image_paths)
            chunk_id = deterministic_id(id_source)

            metadata = {
                k: v for k, v in item.items()
                if k not in ("text", "merged_original_texts")
            }

            chunks.append(
                TokenizedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata
                )
            )

        texts = [c.text for c in chunks]
        ids = [str(c.chunk_id) for c in chunks]
        metadata = [self._sanitize_metadata(c.metadata) for c in chunks]
        for m in metadata:
            if "image_paths" in m and not m["image_paths"]:
                del m["image_paths"]

        return texts, ids, metadata

    # -----------------------------------------------------
    # Embed and store a single file
    # -----------------------------------------------------
    def embed_and_store_single_file_chunks(self, chunks_file):

        texts, ids, metadata = self.load_tokenized_chunks(chunks_file)

        if self.verbose:
            print(f"{os.path.basename(chunks_file)} --> {len(ids)} chunks")

        # Create embeddings
        embeddings = self.embedding_backend.embed(texts)
        embeddings = [e.detach().cpu().numpy() for e in embeddings]

        # Merge duplicates by TEXT
        merged = {}

        for i, text in enumerate(texts):
            img_paths = metadata[i].get("image_paths", [])

            if text not in merged:
                merged[text] = {
                    "id": ids[i],
                    "text": text,
                    "embedding": embeddings[i],
                    "metadata": metadata[i].copy()
                }

                if img_paths:
                    merged[text]["metadata"]["image_paths"] = list(img_paths)

            else:
                existing_paths = merged[text]["metadata"].get("image_paths", [])
                combined = set(existing_paths) | set(img_paths)

                if combined:
                    merged[text]["metadata"]["image_paths"] = list(combined)

        # Unpack merged results
        unique_ids = [v["id"] for v in merged.values()]
        unique_texts = [v["text"] for v in merged.values()]
        unique_embeddings = [v["embedding"] for v in merged.values()]
        unique_metadata = [v["metadata"] for v in merged.values()]

        # Store in vector DB
        self.vectordb.upsert(
            unique_ids,
            unique_texts,
            unique_embeddings,
            unique_metadata,
            scope={"document_name": os.path.basename(chunks_file)}
        )

        return len(unique_ids)

    # -----------------------------------------------------
    # Main entry point
    # -----------------------------------------------------
    def embed_and_store(self):
        chunk_files = expand_files_pattern(self.chunks_path, self.chunk_files_pattern)
        for chunks_file in chunk_files:
            self.embed_and_store_single_file_chunks(chunks_file)
