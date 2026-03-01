
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from .embedding_backends import HFEmbeddingBackend
from Utilities.File_Utilities import expand_files_pattern
import os
import numpy as np





@dataclass
class StructuralChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


#=================================================================================================================================
import hashlib
def deterministic_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
#=================================================================================================================================
class Structural_Chunks_Embedder:
    def __init__(self,
                 chunks_path: str, 
                 chunk_files_pattern: str,
                 embedding_backend: HFEmbeddingBackend, 
                 vectordb, 
                 collection_name: str,
                 verbose: bool = False):

        self.chunks_path = chunks_path
        self.chunk_files_pattern = chunk_files_pattern
        self.embedding_backend = embedding_backend
        self.vectordb = vectordb
        self.collection_name = collection_name
        self.verbose = verbose

    #-----------------------------------------------------------------------------------------------------------------------------

    def load_structural_chunks(self, json_path: str) -> List[StructuralChunk]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract chunks data
        chunks = []
        for item in data:
            text = item.get("text", "").strip()
            image_paths = item.get("image_paths", [])

            # Skip only if both text AND images are missing
            if not text and not image_paths:
                continue

            # Use text if available, otherwise use image paths to generate a stable ID
            id_source = text if text else " ".join(image_paths)
            chunk_id = deterministic_id(id_source)

            # Keep all metadata except text, and preserve original chunk_id
            metadata = {k: v for k, v in item.items() if k != "text"}

            chunks.append(
                StructuralChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata
                )
            )


        # Get texts for embedding
        texts = [c.text for c in chunks]

        # Get chunk IDs
        ids = [c.chunk_id for c in chunks]
        ids = [str(i) for i in ids]

        metadata = [c.metadata for c in chunks]
        # Remove ANY empty list metadata fields
        for m in metadata:
            keys_to_delete = [k for k, v in m.items() if isinstance(v, list) and len(v) == 0]
            for k in keys_to_delete:
                del m[k]

        # Remove metadata blocks
        for m in metadata:
            if "blocks" in m:
                del m["blocks"]  # Remove original blocks to avoid storing complex nested structures in vector DB

        return texts, ids, metadata

    #-----------------------------------------------------------------------------------------------------------------------------
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
                # First occurrence → initialize merged entry
                merged[text] = {
                    "id": ids[i],
                    "text": text,
                    "embedding": embeddings[i],
                    "metadata": metadata[i].copy()
                }

                # Ensure image_paths exists in metadata
                if img_paths:
                    merged[text]["metadata"]["image_paths"] = list(img_paths)

            else:
                # Duplicate text → merge image paths
                existing_paths = merged[text]["metadata"].get("image_paths", [])
                combined = set(existing_paths) | set(img_paths)

                if combined:
                    merged[text]["metadata"]["image_paths"] = list(combined)

        # Unpack merged results into final lists
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

    

    # -----------------------------------------------------
    # Main entry point: Embed multiple files
    # -----------------------------------------------------

    def embed_and_store(self):
        """ Main entry point. Processes one or multiple JSON chunk files. """
        chunk_files = expand_files_pattern(self.chunks_path, self.chunk_files_pattern)
        for chunks_file in chunk_files: 
            self.embed_and_store_single_file_chunks(chunks_file)



#=================================================================================================================================
