import json
from dataclasses import dataclass
from typing import List, Dict, Any
from .embedding_backends import HFEmbeddingBackend
from Utilities.File_Utilities import expand_files_pattern
import os
import numpy as np
import hashlib
import time
from Utilities import Simple_Progress_Bar


@dataclass
class StructuralChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


# ================================================================================================================================
def deterministic_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ================================================================================================================================
class Structural_Chunks_Embedder:
    def __init__(self,
                 chunks_path: str,
                 chunk_files_pattern: str,
                 embedding_backend: HFEmbeddingBackend,
                 vectordb,
                 collection_name: str,
                 verbose: bool = False,
                 llm=None,
                 show_progress: bool = True):

        self.chunks_path = chunks_path
        self.chunk_files_pattern = chunk_files_pattern
        self.embedding_backend = embedding_backend
        self.vectordb = vectordb
        self.collection_name = collection_name
        self.verbose = verbose

        # New parameters
        self.llm = llm
        self.show_progress = show_progress

    # -----------------------------------------------------------------------------------------------------------------------------
    def _build_classification_prompt(self, text: str) -> str:
        return f"""
You are a classification model.
Your task is to read the provided text chunk and assign exactly ONE category from the list below:

1. Product description
2. Process description
3. Usage instructions
4. Application description
5. Legal terms
6. Tale
7. Person description
8. User manual
9. Other

Rules:
- Choose the single best category.
- Do NOT explain your reasoning.
- Output ONLY the category name as plain text.

Chunk text:
<<<{text}>>>
""".strip()

    # -----------------------------------------------------------------------------------------------------------------------------
    def load_structural_chunks(self, json_path: str) -> List[StructuralChunk]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []

        # Progress bar for classification
        pbar = None
        if self.llm is not None and self.show_progress:
            pbar = Simple_Progress_Bar(total=len(data), enabled=True)

        for item in data:
            text = item.get("text", "").strip()
            image_paths = item.get("image_paths", [])

            # Skip only if both text AND images are missing
            if not text and not image_paths:
                if pbar:
                    pbar.update(label="Skipped empty")
                continue

            # Deterministic ID
            id_source = text if text else " ".join(image_paths)
            chunk_id = deterministic_id(id_source)

            # Copy metadata except text
            metadata = {k: v for k, v in item.items() if k != "text"}

            # Optional LLM classification
            if self.llm is not None and text:
                prompt = self._build_classification_prompt(text)
                category = self.llm.complete(prompt).strip()
                metadata["chunk_text_class"] = category

            if pbar:
                pbar.update(label="Classified")

            chunks.append(
                StructuralChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata
                )
            )

        if pbar:
            print()  # newline after progress bar

        # Extract lists
        texts = [c.text for c in chunks]
        ids = [str(c.chunk_id) for c in chunks]
        metadata = [c.metadata for c in chunks]

        # Remove empty list metadata fields
        for m in metadata:
            keys_to_delete = [k for k, v in m.items() if isinstance(v, list) and len(v) == 0]
            for k in keys_to_delete:
                del m[k]

        # Remove blocks to avoid storing nested structures
        for m in metadata:
            if "blocks" in m:
                del m["blocks"]

        return texts, ids, metadata

    # -----------------------------------------------------------------------------------------------------------------------------
    def embed_and_store_single_file_chunks(self, chunks_file):

        # 1. Load structural chunks
        texts, ids, metadata = self.load_structural_chunks(chunks_file)
        if self.verbose:
            print(f'{os.path.basename(chunks_file)} --> {len(ids)} chunks')

        # 2. Create embeddings
        embeddings = self.embedding_backend.embed(texts)
        processed = []
        for e in embeddings:
            if hasattr(e, "detach"):          # torch tensor
                e = e.detach().cpu().numpy()
            elif hasattr(e, "cpu"):           # numpy on device
                e = e.cpu().numpy()
            else:                             # already list or numpy
                e = np.array(e)
            processed.append(e)

        embeddings = processed

        # 3. Merge duplicates by TEXT while combining image_paths
        merged = {}  # key = text, value = merged record

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

        # 4. Store in vector DB
        self.vectordb.upsert(
            unique_ids,
            unique_texts,
            unique_embeddings,
            unique_metadata,
            scope={"document_name": os.path.basename(chunks_file)}
        )

        return len(unique_ids)

    # -----------------------------------------------------------------------------------------------------------------------------
    def embed_and_store(self):
        """ Main entry point. Processes one or multiple JSON chunk files. """
        chunk_files = expand_files_pattern(self.chunks_path, self.chunk_files_pattern)
        for chunks_file in chunk_files:
            self.embed_and_store_single_file_chunks(chunks_file)
