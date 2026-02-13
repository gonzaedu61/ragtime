
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from .embedding_backends import HFEmbeddingBackend
from Utilities.File_Utilities import expand_files_pattern
import os



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
class Embedder:
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
            metadata["chunk_number"] = item.get("chunk_id")

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

        # Adjust Metadata images: Remove empty image paths to avoid storing empty lists in vector DB
        metadata = [c.metadata for c in chunks]
        for m in metadata:
            if "image_paths" in m and not m["image_paths"]:
                del m["image_paths"]

        # Adjust Metadata blocks: Remove spans and convert to JSON string to avoid storing complex nested structures in vector DB
        for m in metadata:
            if "blocks" in m:
                cleaned_blocks = []
                for block in m["blocks"]:
                    cleaned_block = {
                        "kind": block.get("kind"),
                        "text": block.get("text"),
                        "page_num": block.get("page_num"),
                        "heading_level": block.get("heading_level"),
                        "is_process_step": block.get("is_process_step"),
                    }
                    cleaned_blocks.append(cleaned_block)

                # Convert list of dicts → JSON string
                m["blocks"] = json.dumps(cleaned_blocks)

        return texts, ids, metadata

    #-----------------------------------------------------------------------------------------------------------------------------

    def embed_and_store_single_file_chunks(self, chunks_file):


        # 1. Load structural chunks
        texts, ids, metadata = self.load_structural_chunks(chunks_file)
        if self.verbose: print(f'{os.path.basename(chunks_file)} --> {len(ids)} chunks')

        # 2. Create embeddings
        embeddings = self.embedding_backend.embed(texts)
        embeddings = [e.detach().cpu().numpy() for e in embeddings]

        # 3. Remove duplicates while keeping everything aligned
        unique = {}
        for i, id_ in enumerate(ids):
            unique[id_] = (embeddings[i], metadata[i])

        ids = list(unique.keys())
        embeddings = [v[0] for v in unique.values()]
        metadata = [v[1] for v in unique.values()]

        # 4. Store in vector DB
        self.vectordb.upsert(
            ids,
            embeddings,
            metadata,
            scope={"document_name": os.path.basename(chunks_file)}
        )

        return len(ids)
    

    # -----------------------------------------------------
    # Main entry point: Embed multiple files
    # -----------------------------------------------------

    def embed_and_store(self):
        """ Main entry point. Processes one or multiple JSON chunk files. """
        chunk_files = expand_files_pattern(self.chunks_path, self.chunk_files_pattern)
        for chunks_file in chunk_files: 
            self.embed_and_store_single_file_chunks(chunks_file)



#=================================================================================================================================
