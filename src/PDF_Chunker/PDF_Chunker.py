import json
import fitz
import re
from numpy import block
import tiktoken
from typing import List, Dict, Any, Optional
import io
from PIL import Image
import os
from .models import TextSpan, LogicalBlock, StructuralChunk
from .json_export import spans_to_json, blocks_to_json, chunks_to_json
from collections import Counter
import glob


KBs_DIR = "DATA/KBs"
PDF_DIR="1_PDFs"
DOCUMENT_NAME="*.pdf"
OUTPUT_DIR="2_Structural_Chunks"
IMAGES_FOLDER = "3_Extracted_Images"


# ---------------------------------------------------------
# PDF_Chunker v1.1
# ---------------------------------------------------------
# This parser extracts structured content from PDFs and produces
# semantically meaningful JSON chunks suitable for RAG pipelines.
#
# The pipeline performs:
#   1. Span extraction (raw text + images)
#   2. Header/footer removal (optional)
#   3. TOC detection and removal (optional)
#   4. Column detection
#   5. Line merging into logical blocks
#   6. Block classification (heading, paragraph, list, table, caption…)
#   7. Heading-level assignment
#   8. Chunking with semantic boundaries
#   9. Removal of heading-only chunks
#
# The result is a list of StructuralChunk objects with:
#   - text
#   - heading_path
#   - page numbers
#   - block metadata
#   - image references
#
# This version includes:
#   - robust TOC detection (supports multiple TOC formats)
#   - robust header/footer removal
#   - suppression of heading-only chunks
#   - improved heading detection logic
# ---------------------------------------------------------


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Defines all tunable parameters for:
# - chunking
# - heading detection
# - TOC detection
# - header/footer removal
# - image extraction
# This makes the parser highly configurable.

MODEL_NAME = "gpt-4o-mini"
enc = tiktoken.encoding_for_model(MODEL_NAME)

# Chunk size constraints
MIN_TOKENS = 800
MAX_TOKENS = 1500

# Headings of level <= this threshold trigger a chunk break
MIN_HEADING_LEVEL_FOR_CHUNK_BREAK = 4

# Heuristics for heading detection
MAX_WORDS_NUMBERED_HEADING = 50
MAX_WORDS_NON_NUMBERED_HEADING = 50

# Regex patterns for structural detection
HEADING_REGEX = re.compile(r"^(?:[0-9]+(\.[0-9]+)*)?\s*[A-Z][A-Z0-9\s\-\.,]{3,}$")
LIST_BULLET_REGEX = re.compile(r"^(\s*[-*•]\s+|\s*[0-9]+\.\s+)")
PROCESS_STEP_REGEX = re.compile(r"^\s*(Step\s*[0-9]+:|[0-9]+\.)\s+")
CAPTION_REGEX = re.compile(r"^(Figure|Fig\.|Table)\s*\d+[:.\-]")
NUMBERED_HEADING_PREFIX = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[.)])?\s+")


# Header/footer removal settings
REMOVE_HEADERS = True
REMOVE_FOOTERS = True
HEADER_FOOTER_HEIGHT_RATIO = 0.08   # top/bottom 8% of page height
MIN_REPEAT_COUNT = 3                # must appear on ≥ 3 pages

# TOC detection settings
INCLUDE_TOC = False
MAX_TOC_PAGES = 6
MIN_TOC_RIGHT_NUMBERS = 4
MIN_TOC_HEADING_ENTRIES = 4
MAX_TOC_LONG_LINE_RATIO = 0.3

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def find_bytes(path, obj):  # Find bytes within a JSON structure for debugging
    if isinstance(obj, bytes):
        print("Found bytes at:", path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_bytes(f"{path}.{k}", v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_bytes(f"{path}[{i}]", v)

# ---------------------------------------------------------
def _resolve_input_files(dir, pattern) -> List[str]:
    """
    Resolves document_name into a list of PDF file paths.
    Supports:
    - exact filename
    - wildcard patterns (*.pdf)
    - regex (if document_name starts with 're:')
    """

    # Case 1: regex pattern
    if pattern.startswith("re:"):
        regex = re.compile(pattern[3:])
        files = [
            os.path.join(dir, f)
            for f in os.listdir(dir)
            if regex.match(f)
        ]
        return files

    # Case 2: wildcard pattern
    wildcard_path = os.path.join(dir, pattern)
    files = glob.glob(wildcard_path)
    if files:
        return files

    # Case 3: exact file
    exact_path = os.path.join(dir, pattern)
    if os.path.exists(exact_path):
        return [exact_path]

    raise FileNotFoundError(f"No PDF files match: {pattern}")

# ---------------------------------------------------------
# Parser Class
# ---------------------------------------------------------
class PDF_Chunker:

    def __init__(
            self,
            kb_name: str,
            kbs_dir: Optional[str] = KBs_DIR,
            pdf_dir: Optional[str] = PDF_DIR,
            document_name: Optional[str] = DOCUMENT_NAME,
            output_dir: Optional[str] = OUTPUT_DIR,
            export_spans: Optional[bool] = False,
            export_blocks: Optional[bool] = False,
            verbose: bool = False
        ):

        self.verbose = verbose

        # Store paths
        self.kb_path = os.path.abspath(kbs_dir + '/' + kb_name)
        self.pdf_path = os.path.abspath(self.kb_path + '/' + pdf_dir)

        # document_name may be a wildcard or regex
        self.document_name = document_name

        # Output directory
        self.output_dir = os.path.abspath(self.kb_path + '/' + output_dir)
        os.makedirs(self.output_dir, exist_ok=True)


        # Export flags
        self.export_spans = export_spans
        self.export_blocks = export_blocks

    # -----------------------------------------------------
    # Utility
    # -----------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(enc.encode(text))

    def normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def is_bold(self, font_name: str) -> bool:
        return "Bold" in font_name or "bold" in font_name
    
    def _span_signature(self, s: TextSpan):
        """
        Creates a signature for a span that allows detecting repeated
        header/footer elements across pages.

        Signature includes:
        - normalized text (page numbers normalized to PAGE_NUMBER)
        - font size
        - horizontal position (x0)
        """

        text = s.text.strip()

        # Detect page numbers
        if text.isdigit():
            text_key = "PAGE_NUMBER"
        else:
            text_key = text

        return (
            text_key,
            round(s.font_size, 1),
            round(s.bbox[0], 1),   # horizontal alignment
        )

    def detect_repeated_header_footer_spans(self, spans):
        """
        Identifies spans that appear on many pages with the same signature.
        These are likely headers or footers.
        """

        # Group signatures by page
        sig_pages = {}
        for s in spans:
            if getattr(s, "is_image", False):
                continue
            sig = self._span_signature(s)
            sig_pages.setdefault(sig, set()).add(s.page_num)

        # A signature is considered repeated if it appears on multiple pages
        repeated = {sig for sig, pages in sig_pages.items() if len(pages) >= MIN_REPEAT_COUNT}
        return repeated
    
    def filter_headers_footers(self, spans: List[TextSpan]) -> List[TextSpan]:
        """
        Removes spans that:
        - appear on many pages (repeated signature)
        - AND are located in the top/bottom X% of the page
        """

        if not (REMOVE_HEADERS or REMOVE_FOOTERS):
            return spans

        repeated_sigs = self.detect_repeated_header_footer_spans(spans)
        filtered = []

        for s in spans:
            if getattr(s, "is_image", False):
                filtered.append(s)
                continue

            sig = self._span_signature(s)
            if sig not in repeated_sigs:
                filtered.append(s)
                continue

            # Now check vertical position
            page = self.doc[s.page_num - 1]
            page_height = page.rect.height
            y0, y1 = s.bbox[1], s.bbox[3]

            in_header_zone = y1 < page_height * HEADER_FOOTER_HEIGHT_RATIO
            in_footer_zone = y0 > page_height * (1 - HEADER_FOOTER_HEIGHT_RATIO)

            if (REMOVE_HEADERS and in_header_zone) or (REMOVE_FOOTERS and in_footer_zone):
                # Remove it
                continue

            filtered.append(s)

        return filtered


    def detect_toc_pages(self, spans: List[TextSpan]):
        """
        Robust TOC page detector with:
        - line merging
        - indentation tolerance
        - multi-level numbering support
        - improved leader detection
        - improved right-number detection
        """

        # --- Group spans by page ---
        pages: Dict[int, List[TextSpan]] = {}
        for s in spans:
            pages.setdefault(s.page_num, []).append(s)

        toc_pages = set()

        for page_num, page_spans in pages.items():
            if page_num > MAX_TOC_PAGES:
                continue

            page = self.doc[page_num - 1]
            page_width = page.rect.width
            page_height = page.rect.height

            # --- 1. Merge spans into lines ---
            lines = {}
            for s in page_spans:
                # cluster by y-position (tolerance 3 px)
                key = round(s.bbox[1] / 3)
                lines.setdefault(key, []).append(s)

            merged_lines = []
            for spans_on_line in lines.values():
                spans_on_line.sort(key=lambda s: s.bbox[0])
                merged_text = " ".join(s.text.strip() for s in spans_on_line if s.text.strip())
                if not merged_text:
                    continue

                merged_bbox = (
                    spans_on_line[0].bbox[0],
                    spans_on_line[0].bbox[1],
                    spans_on_line[-1].bbox[2],
                    spans_on_line[-1].bbox[3],
                )

                merged_lines.append((merged_text, merged_bbox))

            toc_like = 0
            total = len(merged_lines)

            # --- 2. Evaluate each merged line ---
            for txt, bbox in merged_lines:
                txt = txt.strip()
                if not txt:
                    continue

                # --- Pattern A: Single-line TOC entry ---
                # Detect leaders
                has_leader = (
                    "..." in txt or
                    "___" in txt or
                    "···" in txt or
                    "—" in txt or
                    "–" in txt or
                    re.search(r"\.{5,}", txt) is not None  # long dot sequences
                )

                # Detect page number at end
                ends_with_number = bool(re.search(r"\d+$", txt))

                # Detect title text
                has_title_text = bool(re.search(r"[A-Za-zÄÖÜäöüß]", txt))

                # Detect multi-level numbering prefix
                multi_level_prefix = bool(re.match(r"^\d+(?:\.\d+){1,6}\b", txt))

                # Allow indentation (TOC entries often indent deeply)
                pattern_A = (
                    has_leader
                    and ends_with_number
                    and (has_title_text or multi_level_prefix)
                )

                # --- Pattern B: Right-aligned page number ---
                is_right_number = (
                    ends_with_number
                    and txt.isdigit()
                    and bbox[2] > page_width * 0.65
                    and page_height * 0.05 < bbox[1] < page_height * 0.95
                )

                pattern_B = False
                if is_right_number:
                    # Look for a left-side title on same line
                    for other_txt, other_bbox in merged_lines:
                        if other_txt == txt:
                            continue
                        if abs(other_bbox[1] - bbox[1]) < 3:
                            if len(other_txt.strip()) > 3:
                                pattern_B = True
                                break

                if pattern_A or pattern_B:
                    toc_like += 1

            # --- Decide if page is TOC ---
            if total > 0 and toc_like / total >= 0.20:
                toc_pages.add(page_num)

        return toc_pages


    def merge_cross_page_paragraphs(self, blocks):
        """
        Merge paragraph blocks that continue across page boundaries.
        A block at the top of a page is merged with the last block of the
        previous page if both are paragraphs and the second one does not
        start with a structural marker (heading, list, caption, etc.).
        """

        merged = []
        prev = None

        for b in blocks:
            if prev is None:
                prev = b
                continue

            # Conditions for merging
            continuation = (
                prev.kind == "paragraph"
                and b.kind == "paragraph"
                and b.page_num == prev.page_num + 1
                and b.spans[0].bbox[1] < 100  # near top of page
                and not HEADING_REGEX.match(b.text)
                and not LIST_BULLET_REGEX.match(b.text)
                and not CAPTION_REGEX.match(b.text)
            )

            if continuation:
                prev.text = prev.text.rstrip() + " " + b.text.lstrip()
                prev.spans.extend(b.spans)
                prev.image_paths.extend(b.image_paths)
            else:
                merged.append(prev)
                prev = b

        if prev:
            merged.append(prev)

        return merged
    

    def merge_images_into_previous_blocks(self, blocks):
        """
        Attach each image block to the nearest previous non-image block
        using vertical proximity, including images at the top of the next page.
        If the nearest previous block is a heading, attach to the next paragraph instead.
        """

        merged = []
        text_blocks = [b for b in blocks if b.kind != "image"]

        for block in blocks:
            if block.kind != "image":
                merged.append(block)
                continue

            img_span = block.spans[0]
            img_y = img_span.bbox[1]
            page_num = block.page_num

            page = self.doc[page_num - 1]
            page_height = page.rect.height

            SAME_PAGE_MAX_DIST = page_height * 0.20
            NEXT_PAGE_MAX_DIST = page_height * 0.10

            best_block = None
            best_distance = float("inf")

            # --- PASS 1: Try to attach to previous non-heading block ---
            for tb in text_blocks:
                if tb.page_num == page_num:
                    tb_y = tb.spans[-1].bbox[3]
                    distance = img_y - tb_y

                    if 0 <= distance < best_distance and distance <= SAME_PAGE_MAX_DIST:
                        if tb.kind != "heading":  # skip headings
                            best_distance = distance
                            best_block = tb

                elif tb.page_num == page_num - 1:
                    if img_y <= NEXT_PAGE_MAX_DIST:
                        if tb.kind != "heading":
                            best_distance = 0
                            best_block = tb

            # --- PASS 2: If previous block was a heading, attach to next paragraph ---
            if best_block is None:
                for tb in text_blocks:
                    if tb.page_num == page_num and tb.spans[0].bbox[1] > img_y:
                        if tb.kind == "paragraph":
                            best_block = tb
                            break

            # Attach image if a suitable block was found
            if best_block:
                best_block.image_paths.extend(
                    [s.image_path for s in block.spans if s.is_image and s.image_path]
                )

        return merged
    


    # -----------------------------------------------------
    # Step 1: Extract spans
    # -----------------------------------------------------

    def extract_spans(self) -> List[TextSpan]:
        """
        Extracts raw spans from the PDF:
        - text spans
        - image spans (saved to disk)
        - preserves block_no, line_no, bbox, page_num

        Produces a flat list of TextSpan objects.
        This is the lowest-level extraction step.
        No semantic interpretation happens here. 

        """

        spans: List[TextSpan] = []
        jsonBlocks = {}

        # Create a subfolder for this specific PDF
        pdf_base_name = self.document_base_name

        img_dir = os.path.join(
            os.path.dirname(self.pdf_path),
            IMAGES_FOLDER,
            pdf_base_name
        )
        os.makedirs(img_dir, exist_ok=True)

        def save_image_bytes(image_bytes, page_num, block_no):
            """Guarda bytes de imagen como JPG y devuelve la ruta."""
            try:
                img = Image.open(io.BytesIO(image_bytes))
                img = img.convert("RGB")
            except Exception as e:
                print(f"[WARN] Error decoding image on page {page_num}, block {block_no}: {e}")
                return None

            filename = f"page_{page_num}_block_{block_no}.jpg"
            filepath = os.path.join(img_dir, filename)
            img.save(filepath, "JPEG", quality=90)
            return filepath

        # ---------------------------------------------------------
        # Procesar cada página
        # ---------------------------------------------------------
        for page_num, page in enumerate(self.doc, start=1):
            page_dict = page.get_text("dict")
            blocks = page_dict["blocks"]

            cleaned_blocks = []

            for block_no, block in enumerate(blocks):

                # ---------------------------------------------
                # Caso 1: Bloque de imagen
                # ---------------------------------------------
                if block.get("type") == 1 and "image" in block:
                    img_bytes = block.get("image")

                    if isinstance(img_bytes, bytes):
                        img_path = save_image_bytes(img_bytes, page_num, block_no)
                    else:
                        img_path = None

                    # Create an image span
                    ts = TextSpan(
                        text="", 
                        font_size=0.0,
                        bold=False,
                        bbox=block.get("bbox"),
                        block_no=block_no,
                        line_no=0,
                        page_num=page_num,
                        is_image=True,
                        image_path=img_path
                    )
                    spans.append(ts)

                    cleaned_blocks.append({
                        "type": "image",
                        "bbox": block.get("bbox"),
                        "image_path": img_path
                    })
                    continue

                # ---------------------------------------------
                # Caso 2: Bloque de texto
                # ---------------------------------------------
                if "lines" in block:
                    block_lines = []
                    for line_no, line in enumerate(block["lines"]):
                        line_spans = []

                        for span in line["spans"]:
                            text = span.get("text", "")
                            if not text.strip():
                                continue

                            # Crear TextSpan
                            ts = TextSpan(
                                text=text,
                                font_size=span.get("size", 0.0),
                                bold=self.is_bold(span.get("font", "")),
                                bbox=span.get("bbox"),
                                block_no=block_no,
                                line_no=line_no,
                                page_num=page_num
                            )
                            spans.append(ts)

                            # Guardar versión JSON limpia
                            line_spans.append({
                                "text": text,
                                "font_size": ts.font_size,
                                "bold": ts.bold,
                                "bbox": ts.bbox,
                                "line_no": line_no
                            })

                        if line_spans:
                            block_lines.append(line_spans)

                    cleaned_blocks.append({
                        "type": "text",
                        "bbox": block.get("bbox"),
                        "lines": block_lines
                    })

                # ---------------------------------------------
                # Caso 3: Otros tipos de bloques (vector, etc.)
                # ---------------------------------------------
                else:
                    cleaned_blocks.append({
                        "type": block.get("type", "unknown"),
                        "bbox": block.get("bbox")
                    })

            jsonBlocks['page-' + str(page_num)] = cleaned_blocks

        # ---------------------------------------------------------
        # Guardar blocks JSON
        # ---------------------------------------------------------
        if self.export_blocks:
            out_path = os.path.join(
                self.output_dir,
                f"{self.document_base_name}_blocks.json"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(jsonBlocks, f, indent=4, ensure_ascii=False)

        return spans

    # -----------------------------------------------------
    # Step 2: Detect columns
    # -----------------------------------------------------

    def detect_columns(self, spans: List[TextSpan], tolerance=20):
        """
        Groups spans into columns by clustering x-center positions.
        Column ID is used only as a boundary, not for sorting.

        Prevents merging text across columns.
        (this column detection function is not properly tested yet, so use with caution and verify results on complex layouts)

        """

        spans_by_page = {}
        for s in spans:
            spans_by_page.setdefault(s.page_num, []).append(s)

        for page_num, page_spans in spans_by_page.items():
            x_centers = [(s, (s.bbox[0] + s.bbox[2]) / 2) for s in page_spans]
            x_centers_sorted = sorted(x_centers, key=lambda x: x[1])

            columns = []
            current_col = [x_centers_sorted[0][0]]
            last_x = x_centers_sorted[0][1]

            for span, xc in x_centers_sorted[1:]:
                if abs(xc - last_x) < tolerance:
                    current_col.append(span)
                else:
                    columns.append(current_col)
                    current_col = [span]
                last_x = xc

            columns.append(current_col)

            for col_id, col_spans in enumerate(columns):
                for s in col_spans:
                    s.column_id = col_id

    # -----------------------------------------------------
    # Step 3: Group spans into blocks
    # -----------------------------------------------------

    def group_spans_into_blocks(self, spans: List[TextSpan]) -> List[LogicalBlock]:
        """
        Merges spans into line-level units, then groups lines into blocks
        based on block_no, page_num, and column_id.

        Each block is classified as:
        - heading
        - paragraph
        - list_item
        - table
        - caption
        - sidebar
        - image

        Key features
        - merges spans into full lines
        - detects headings using multiple heuristics
        - detects list items
        - detects tables via column alignment
        - detects captions via font size + adjacency
        - detects sidebars via width + margin position

        """

        blocks = []

        # ---------------------------------------------------------
        # PREPROCESS: merge spans into line-level units
        # ---------------------------------------------------------
        merged = []

        # Group spans by (page, block_no, line_no)
        from collections import defaultdict
        groups = defaultdict(list)

        for s in spans:
            if getattr(s, "is_image", False):
                merged.append(s)
                continue
            key = (s.page_num, s.block_no, s.line_no)
            groups[key].append(s)

        for (page, block_no, line_no), group in groups.items():
            if any(getattr(s, "is_image", False) for s in group):
                # keep image spans as-is
                merged.extend(group)
                continue

            # Merge text spans into a single line-level span
            text = " ".join(s.text for s in group)
            font_sizes = [s.font_size for s in group]
            bold_flags = [s.bold for s in group]

            x0 = min(s.bbox[0] for s in group)
            y0 = min(s.bbox[1] for s in group)
            x1 = max(s.bbox[2] for s in group)
            y1 = max(s.bbox[3] for s in group)

            merged_span = TextSpan(
                text=text,
                font_size=sum(font_sizes) / len(font_sizes),
                bold=all(bold_flags),
                bbox=[x0, y0, x1, y1],
                block_no=block_no,
                line_no=line_no,
                page_num=page,
            )
            merged_span.original_spans = group  # optional metadata to not lose the original formatting of words (i.e. bold)

            merged.append(merged_span)

        # Replace spans with merged version
        spans = merged



        # Sort primarily by page and block/line order, not by column
        spans_sorted = sorted(
            spans,
            key=lambda s: (
                s.page_num,
                s.block_no,
                s.line_no,
                s.bbox[1],  # y0 as a tie-breaker if needed
            ),
        )

        current = []
        current_block_no = None
        current_page = None
        current_col = None

        def flush():
            nonlocal current
            if not current:
                return
            text = self.normalize(" ".join(s.text for s in current))
            first = current[0]


            # --------------------------------------------
            # Determine if this block is a heading
            # --------------------------------------------
            all_bold = all(s.bold for s in current)
            line_font = max(s.font_size for s in current)


            words = text.split()
            word_count = len(words)
            is_short_non_numbered = word_count <= MAX_WORDS_NON_NUMBERED_HEADING
            is_short_numbered = word_count <= MAX_WORDS_NUMBERED_HEADING
                        
            # Headings often end with ":" in German technical docs
            valid_heading_ending = (
                text.endswith(":") or
                not text.endswith((".", ";"))
            )

            same_block = len({s.block_no for s in current}) == 1
            few_lines = len({s.line_no for s in current}) <= 2


            # Compare to global body font size
            larger_than_body = line_font > self.body_font_size * 1.05

            is_heading_candidate = (
                HEADING_REGEX.match(text)
                or all_bold
                or larger_than_body
            )

            # --------------------------------------------
            # Numbered heading detection
            # --------------------------------------------
            m = NUMBERED_HEADING_PREFIX.match(text)
            if m:
                prefix = m.group(1)
                after_prefix = text[m.end():].strip()

                after_word_count = len(after_prefix.split())
                after_short = after_word_count <= MAX_WORDS_NUMBERED_HEADING
                
                after_no_period = not after_prefix.endswith(".")
                after_all_bold = all(s.bold for s in current)
                after_matches_heading_regex = bool(HEADING_REGEX.match(after_prefix))

                if (
                    same_block
                    and few_lines
                    and after_short
                    and (after_prefix.endswith(":") or not after_prefix.endswith((".", ";")))
                    and (after_all_bold or after_matches_heading_regex)
                ):
                    kind = "heading"
                    blocks.append(
                        LogicalBlock(
                            kind=kind,
                            text=text,
                            spans=current.copy(),
                            page_num=first.page_num,
                            is_process_step=False,
                            image_paths=getattr(first, "image_paths", [])
                        )
                    )
                    current = []
                    return  # early exit from flush()

            # --------------------------------------------
            # Existing heading classification (updated)
            # --------------------------------------------

            kind = ""

            # Numbered heading detection
            if m:
                if (
                    is_heading_candidate
                    and is_short_numbered
                    and few_lines
                    and valid_heading_ending
                    and same_block
                ):
                    kind = "heading"

            # Fallback classification ALWAYS runs if kind is still empty
            if kind == "":
                if (
                    is_heading_candidate
                    and is_short_non_numbered
                    and few_lines
                    and valid_heading_ending
                    and same_block
                ):
                    kind = "heading"
                elif LIST_BULLET_REGEX.match(text):
                    kind = "list_item"
                else:
                    kind = "paragraph"

            is_step = bool(PROCESS_STEP_REGEX.match(text))

            blocks.append(
                LogicalBlock(
                    kind=kind,
                    text=text,
                    spans=current.copy(),
                    page_num=first.page_num,
                    is_process_step=is_step,
                    image_paths=getattr(first, "image_paths", [])
                )
            )
            current = []

        for s in spans_sorted:

            # Image block
            if getattr(s, "is_image", False):
                blocks.append(
                    LogicalBlock(
                        kind="image",
                        text="",
                        spans=[s],
                        page_num=s.page_num,
                        is_process_step=False,
                        image_paths=[s.image_path] if s.image_path else []
                    )
                )
                continue

            # Text block: keep using column_id only as a *boundary*, not as a sort driver
            if (
                s.block_no != current_block_no
                or s.page_num != current_page
                #or getattr(s, "column_id", None) != current_col
            ):
                flush()
                current_block_no = s.block_no
                current_page = s.page_num
                #current_col = getattr(s, "column_id", None)

            current.append(s)

        flush()
        return blocks

    # -----------------------------------------------------
    # Step 4: Detect tables, captions, sidebars
    # -----------------------------------------------------

    def detect_tables(self, blocks: List[LogicalBlock]):
        for block in blocks:
            xs = [s.bbox[0] for s in block.spans]
            if len(xs) > 5:
                unique_xs = len(set(round(x, -1) for x in xs))
                if unique_xs >= 2:
                    block.kind = "table"

    def detect_captions(self, blocks: List[LogicalBlock], avg_font: float):
        for i, block in enumerate(blocks):
            if CAPTION_REGEX.match(block.text):
                if block.spans[0].font_size < avg_font:
                    block.kind = "caption"
                    continue

            if block.kind == "paragraph" and block.spans[0].font_size < avg_font * 0.9:
                if i > 0 and blocks[i-1].kind in ("table", "image"):
                    block.kind = "caption"
                if i < len(blocks)-1 and blocks[i+1].kind in ("table", "image"):
                    block.kind = "caption"

    def detect_sidebars(self, blocks: List[LogicalBlock], page_width: float):
        for block in blocks:

            if block.kind == "heading": 
                continue # headings must never become sidebars

            x0, y0, x1, y1 = block.spans[0].bbox
            width = x1 - x0
            if width < page_width * 0.4:
                if x0 < 50 or x1 > (page_width - 50):
                    block.kind = "sidebar"

    # -----------------------------------------------------
    # Step 5: Assign heading levels
    # -----------------------------------------------------

    def assign_heading_levels(self, blocks: List[LogicalBlock]):
        """
        Assign heading levels based on font size as a fallback.
        Numeric prefixes will override this later.
        """
        sizes = [b.spans[0].font_size for b in blocks if b.kind == "heading"]
        if not sizes:
            return

        unique = sorted(set(sizes), reverse=True)
        size_to_level = {size: i+1 for i, size in enumerate(unique)}

        for b in blocks:
            if b.kind == "heading":
                b.heading_level = size_to_level.get(b.spans[0].font_size)


    def assign_numeric_heading_levels(self, blocks: List[LogicalBlock]):
        """
        Assign heading levels based on numeric prefixes like:
        1 → level 1
        1.2 → level 2
        3.4.5 → level 3
        """
        for b in blocks:
            if b.kind != "heading":
                continue

            m = NUMBERED_HEADING_PREFIX.match(b.text)
            if m:
                # Count dots to determine level
                prefix = m.group(1)
                level = prefix.count(".") + 1
                b.heading_level = level

        for b in blocks:
            if b.kind == "heading" and b.heading_level is None:
                b.heading_level = 1




    # -----------------------------------------------------
    # Step 6: Chunking
    # -----------------------------------------------------

    def create_chunks(self, blocks: List[LogicalBlock]) -> List[StructuralChunk]:

        """
        Converts logical blocks into semantic chunks.

        Chunk boundaries occur at:
        - headings (depending on level)
        - page breaks
        - block-type changes (paragraph → table, etc.)
        - token overflow (MAX_TOKENS)

        Heading-only chunks are suppressed.
        """

        chunks: List[StructuralChunk] = []
        current_blocks: List[LogicalBlock] = []
        current_pages = set()
        heading_path: List[str] = []
        chunk_id = 1

        def flush_chunk():
            nonlocal chunk_id, current_blocks, current_pages

            # Skip heading-only chunks
            if not current_blocks:
                return

            # If the chunk contains only headings, skip it
            non_heading_blocks = [b for b in current_blocks if b.kind != "heading"]
            if len(non_heading_blocks) == 0:
                current_blocks = []
                current_pages = set()
                return

            text = self.normalize(" ".join(b.text for b in current_blocks))
            tokens = self.count_tokens(text)

            # Collect image paths from ALL blocks (images were merged into text blocks)
            image_paths = []
            for b in current_blocks:
                for p in b.image_paths:
                    rel_path = os.path.relpath(p, start=self.output_dir)
                    image_paths.append(rel_path)

            # --------------------------------------------
            # OPTION B: New chunk-type logic
            # --------------------------------------------
            block_kinds = {b.kind for b in current_blocks}

            # Always record the combination of block kinds
            mixed_combo = " + ".join(sorted(block_kinds))

            # Chunk type rules:
            # - If only one block kind → use that kind
            # - If multiple block kinds → "mixed"
            # - Special case: pure image chunks (rare now)
            if block_kinds == {"image"}:
                chunk_type = "image"
            elif len(block_kinds) == 1:
                # Single-type chunk → use the type name directly
                chunk_type = next(iter(block_kinds))
            else:
                # Multiple block types → mixed
                chunk_type = "mixed"

            # --------------------------------------------

            chunks.append(
                StructuralChunk(
                    document_name=self.document_base_name,
                    chunk_id=chunk_id,
                    pages=sorted(current_pages),
                    heading_path=heading_path.copy(),
                    text=text,
                    blocks=current_blocks.copy(),
                    token_count=tokens,
                    chunk_type=chunk_type,
                    image_paths=image_paths,
                    mixed_combo=mixed_combo
                )
            )

            chunk_id += 1
            current_blocks = []
            current_pages = set()


        for block in blocks:
            # -----------------------------
            # 1) Heading: hard semantic boundary
            # -----------------------------
            if block.kind == "heading":
                # Decide if this is a numbered heading
                m = NUMBERED_HEADING_PREFIX.match(block.text or "")
                level = block.heading_level or 1

                if m:
                    # Numbered heading → use numeric level threshold
                    break_here = level <= MIN_HEADING_LEVEL_FOR_CHUNK_BREAK
                else:
                    # Non-numbered heading → always break chunk
                    break_here = True

                # If we break here, close the current chunk BEFORE this heading
                if break_here:
                    flush_chunk()

                # Update heading path using the level
                heading_path = heading_path[:level - 1]
                heading_path.append(block.text)

                # Start a new chunk with the heading if we broke,
                # otherwise treat it as inline text in the current chunk
                if break_here:
                    current_blocks = [block]
                    current_pages = {block.page_num}
                else:
                    current_blocks.append(block)
                    current_pages.add(block.page_num)

                continue


            # -----------------------------
            # 2) Page break: optional semantic boundary
            # -----------------------------
            if current_blocks and block.page_num != current_blocks[-1].page_num:
                flush_chunk()

            # -----------------------------
            # 3) Block-type change: avoid mixing very different semantics
            #    (paragraph/list_item can be grouped together, others separated)
            # -----------------------------
            if current_blocks:
                last_kind = current_blocks[-1].kind
                new_kind = block.kind

                def is_textual(k: str) -> bool:
                    return k in ("paragraph", "list_item")

                if is_textual(last_kind) and is_textual(new_kind):
                    # ok to keep together
                    pass
                else:
                    # different semantic type → new chunk
                    if new_kind != last_kind:
                        flush_chunk()

            # -----------------------------
            # 4) Size guardrail: only split when adding this block would exceed MAX_TOKENS
            #    We do NOT merge across semantic boundaries just to reach a minimum size.
            # -----------------------------
            if current_blocks:
                candidate_text = self.normalize(
                    " ".join(b.text for b in current_blocks + [block])
                )
                candidate_tokens = self.count_tokens(candidate_text)

                if candidate_tokens > MAX_TOKENS:
                    # Close current chunk, start a new one with this block
                    flush_chunk()
                    current_blocks = [block]
                    current_pages = {block.page_num}
                    continue

            # -----------------------------
            # 5) Normal accumulation
            # -----------------------------
            current_blocks.append(block)
            current_pages.add(block.page_num)

        # Flush any remaining blocks
        flush_chunk()

        return chunks

    # -----------------------------------------------------
    # One file parser
    # -----------------------------------------------------

    def parse_document(self, pdf_pathName: str) -> List[StructuralChunk]:
        """
        Full pipeline:
        1. Extract spans
        2. Remove headers/footers
        3. Remove TOC pages (optional)
        4. Save spans JSON
        5. Detect columns
        6. Group spans into blocks
        7. Detect tables, captions, sidebars
        8. Assign heading levels
        9. Chunk blocks
        10. Save chunks JSON
        """


        # Open the PDF
        self.doc = fitz.open(pdf_pathName)
        self.document_base_name = os.path.splitext(os.path.basename(pdf_pathName))[0]

        spans = self.extract_spans()
        spans = self.filter_headers_footers(spans)
        if not INCLUDE_TOC:
            toc_pages = self.detect_toc_pages(spans)
            spans = [s for s in spans if s.page_num not in toc_pages]


        # Save spans JSON only if enabled
        if self.export_spans:
            json_spans = spans_to_json(spans)
            out_path = os.path.join(
                self.output_dir,
                f"{self.document_base_name}_spans.json"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(json_spans, f, indent=4, ensure_ascii=False)


        font_sizes = [s.font_size for s in spans if not getattr(s, "is_image", False)]
        size_counts = Counter(round(size, 1) for size in font_sizes)
        body_font_size = size_counts.most_common(1)[0][0]
        self.body_font_size = body_font_size

        self.detect_columns(spans)

        blocks = self.group_spans_into_blocks(spans)

        avg_font = sum(s.font_size for s in spans) / len(spans)
        page_width = self.doc[0].rect.width

        self.detect_tables(blocks)
        self.detect_captions(blocks, avg_font)
        self.detect_sidebars(blocks, page_width)
        blocks = self.merge_images_into_previous_blocks(blocks)
        blocks = self.merge_cross_page_paragraphs(blocks)
        self.assign_heading_levels(blocks)
        self.assign_numeric_heading_levels(blocks)

        chunks = self.create_chunks(blocks)

        json_chunks = chunks_to_json(chunks)
        chunks_path = os.path.join(
            self.output_dir,
            f"{self.document_base_name}_chunks.json"
        )
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(json_chunks, f, indent=4, ensure_ascii=False)
        
        return chunks
    

    # -----------------------------------------------------
    # Main entry point: Parse multiple files
    # -----------------------------------------------------

    def parse(self):
        """ Main entry point. Processes one or multiple PDF files. """
        
        pdf_files = _resolve_input_files(self.pdf_path, self.document_name)

        for pdf_pathName in pdf_files: 
            if self.verbose: print(f"Chunking {os.path.basename(pdf_pathName)} ... ", end="", flush=True)
            chunks = self.parse_document(pdf_pathName)
            if self.verbose: print(len(chunks))


