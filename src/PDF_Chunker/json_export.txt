# json_export.py
"""
Utility functions for converting parser dataclasses into
JSON‑serializable dictionaries.

This module keeps the parser code clean and ensures that
all exported structures are stable, explicit, and safe.
"""

from typing import List, Dict, Any

# Import your dataclasses
from .models import TextSpan, LogicalBlock, StructuralChunk


# ---------------------------------------------------------
# TextSpan → JSON
# ---------------------------------------------------------
def span_to_json(s: TextSpan) -> Dict[str, Any]:
    return {
        "text": s.text,
        "font_size": s.font_size,
        "bold": s.bold,
        "bbox": list(s.bbox) if s.bbox else None,
        "block_no": s.block_no,
        "line_no": s.line_no,
        "page_num": s.page_num,
        "column_id": s.column_id,
    }


# ---------------------------------------------------------
# LogicalBlock → JSON
# ---------------------------------------------------------
def block_to_json(b: LogicalBlock) -> Dict[str, Any]:
    return {
        "kind": b.kind,
        "text": b.text,
        "page_num": b.page_num,
        "heading_level": b.heading_level,
        "is_process_step": b.is_process_step,
        "spans": [span_to_json(s) for s in b.spans],
    }


# ---------------------------------------------------------
# StructuralChunk → JSON
# ---------------------------------------------------------
def chunk_to_json(c: StructuralChunk) -> Dict[str, Any]:
    return {
        "document_name": c.document_name,
        "chunk_id": c.chunk_id,
        "pages": c.pages,
        "heading_path": c.heading_path,
        "text": c.text,
        "token_count": c.token_count,
        "chunk_type": c.chunk_type,
        "image_paths": c.image_paths,
        "blocks": [block_to_json(b) for b in c.blocks],
        "mixed_combo": c.mixed_combo
    }


# ---------------------------------------------------------
# Bulk converters
# ---------------------------------------------------------
def spans_to_json(spans: List[TextSpan]) -> List[Dict[str, Any]]:
    out = []
    for s in spans:
        item = {
            "text": s.text,
            "font_size": s.font_size,
            "bold": s.bold,
            "bbox": s.bbox,
            "block_no": s.block_no,
            "line_no": s.line_no,
            "page_num": s.page_num,
        }
        
        if getattr(s, "is_image", False):
            item["is_image"] = True
            item["image_path"] = s.image_path

        out.append(item)

    return out

def blocks_to_json(blocks: List[LogicalBlock]) -> List[Dict[str, Any]]:
    return [block_to_json(b) for b in blocks]


def chunks_to_json(chunks: List[StructuralChunk]) -> List[Dict[str, Any]]:
    return [chunk_to_json(c) for c in chunks]
