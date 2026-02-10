
import json
import sys
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any
from .embedding_backends import HFEmbeddingBackend


@dataclass
class StructuralChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


class Semantic_Chunker:
    def __init__(self,
                 json_path: str, 
                 embedder: HFEmbeddingBackend, 
                 vectordb, 
                 collection_name: str, 
                 persist_dir: str):
        self.json_path = json_path
        self.embedder = embedder
        self.vectordb = vectordb
        self.collection_name = collection_name
        self.persist_dir = persist_dir


    def load_structural_chunks(self) -> List[StructuralChunk]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        for item in data:
            chunk_id = item.get("chunk_id") or str(uuid.uuid4())
            text = item.get("text", "").strip()
            metadata = {k: v for k, v in item.items() if k not in ("text", "chunk_id")}
            chunks.append(
                StructuralChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata
                )
            )

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

        # Get texts for embedding
        texts = [c.text for c in chunks]

        return texts, ids, metadata


    def embed_and_store(self):

        # 1. Load structural chunks
        texts, ids, metadata = self.load_structural_chunks()
        print('Chunks loaded ...')

        # 3. Create embeddings and convert to list of lists
        print('Starting embeddings ...')
        embeddings = self.embedder.embed(texts)
        embeddings = [e.detach().cpu().numpy() for e in embeddings]
        print('Embedding created ...')


        # 4. Store in vector DB
        print('Storing in vector DB ...')
        self.vectordb.add(ids, embeddings, metadata)
        print('Stored in vector DB')

        # Return stored chunks 
        return len(ids)

