from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class TextSpan:
    text: str
    font_size: float
    bold: bool
    bbox: Any
    block_no: int
    line_no: int
    page_num: int
    column_id: int = 0
    is_image: bool = False 
    image_path: Optional[str] = None
    original_spans: List["TextSpan"] = field(default_factory=list, repr=False, compare=False)

@dataclass
class LogicalBlock:
    kind: str
    text: str
    spans: List[TextSpan]
    page_num: int
    heading_level: Optional[int] = None
    is_process_step: bool = False
    image_paths: List[str] = field(default_factory=list)


@dataclass
class StructuralChunk:
    document_name: str
    chunk_id: int
    pages: List[int]
    heading_path: List[str]
    text: str
    blocks: List[LogicalBlock]
    token_count: int
    chunk_type: str
    image_paths: List[str] = field(default_factory=list)
    mixed_combo: str = ""
