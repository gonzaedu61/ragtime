# **PDFStructuralParser v1.1 — README**

## **Overview**

`PDFStructuralParser` is a high‑precision, configurable PDF parsing pipeline designed for **semantic document understanding** and **RAG (Retrieval‑Augmented Generation)** workflows.  
It transforms raw PDF pages into **structured, hierarchical, semantically meaningful chunks**, preserving:

- heading hierarchy  
- paragraph boundaries  
- list structure  
- tables  
- captions  
- images  
- page metadata  

The parser is engineered to handle **real‑world, messy PDFs** including:

- multi‑column layouts  
- inconsistent formatting  
- noisy spans  
- repeated headers/footers  
- table of contents pages  
- embedded images  
- mixed font sizes and styles  

This v1.1 release adds:

- **custom output directory support**  
- **optional export of spans and blocks JSON**  
- **cleaner chunk output (heading‑only chunks suppressed)**  

---

## **Key Features**

### **✓ Accurate span extraction**
Extracts text and image spans from PDFs using PyMuPDF (`fitz`), preserving:

- text  
- font size  
- bold/regular  
- bounding boxes  
- block/line numbers  
- page numbers  

### **✓ Optional header/footer removal**
Detects repeated elements across pages using:

- signature matching (text + font + x‑position)  
- vertical position (top/bottom of page)  

This avoids removing real content while eliminating noise.

### **✓ Optional TOC removal**
Supports **multiple TOC formats**, including:

- classic two‑column TOC (title left, page number right)  
- single‑span TOC entries with dotted/underscore leaders  
- hybrid formats  

Only early pages are considered TOC candidates.

### **✓ Column detection**
Clusters spans by x‑center to prevent cross‑column merging.

### **✓ Logical block construction**
Merges spans into line‑level units, then groups lines into blocks classified as:

- heading  
- paragraph  
- list item  
- table  
- caption  
- sidebar  
- image  

### **✓ Robust heading detection**
Uses multiple signals:

- numeric prefixes (`1`, `1.2`, `3.4.5`)  
- bold text  
- larger font size  
- uppercase patterns  
- short line length  
- ending with `:` (common in German technical docs)  

### **✓ Hierarchical heading levels**
Heading levels are assigned using:

- numeric prefix depth  
- fallback: font size ranking  

### **✓ Semantic chunking**
Chunks are created based on:

- heading boundaries  
- page breaks  
- block-type changes  
- token limits  
- semantic continuity  

### **✓ Heading-only chunks are suppressed**
Headings populate the `heading_path` but do not form standalone chunks.

### **✓ JSON export**
Exports:

- chunks (always)  
- spans (optional)  
- blocks (optional)  

in clean, machine‑readable JSON.

---

## **New in v1.1**

### **1. Custom Output Directory**
You can now specify where all JSON outputs should be written:

```python
parser = PDFStructuralParser(
    "docs/manual.pdf",
    output_dir="output/json"
)
```

If omitted, the parser writes to the same directory as the input PDF.

### **2. Optional Export of Spans and Blocks**
By default, only the final **chunks JSON** is exported.

You can enable additional exports:

```python
parser = PDFStructuralParser(
    "docs/manual.pdf",
    export_spans=True,
    export_blocks=True
)
```

This produces:

```
manual_chunks.json
manual_spans.json
manual_blocks.json
```

### **3. Cleaner Chunk Output**
Heading‑only chunks are now automatically suppressed.  
Headings still update the `heading_path`, but no empty or redundant chunks are created.

---

## **Pipeline Architecture**

```
PDF → extract_spans()
      → filter_headers_footers()
      → detect_toc_pages() → remove TOC spans
      → detect_columns()
      → group_spans_into_blocks()
      → detect_tables()
      → detect_captions()
      → detect_sidebars()
      → assign_heading_levels()
      → assign_numeric_heading_levels()
      → create_chunks()
      → export JSON
```

Each stage is modular and can be extended or replaced.

---

## **Configuration Options**

### **Constructor Parameters (v1.1)**

| Parameter | Type | Default | Description |
|----------|------|----------|-------------|
| `pdf_path` | `str` | required | Path to the PDF file |
| `document_name` | `str` | filename | Name used in output JSON |
| `output_dir` | `str` | same folder as PDF | Directory where JSON files are written |
| `export_spans` | `bool` | `False` | Whether to export `_spans.json` |
| `export_blocks` | `bool` | `False` | Whether to export `_blocks.json` |

### **Header/Footer Removal**

```python
REMOVE_HEADERS = True
REMOVE_FOOTERS = True
HEADER_FOOTER_HEIGHT_RATIO = 0.08
MIN_REPEAT_COUNT = 3
```

### **TOC Filtering**

```python
INCLUDE_TOC = False
MAX_TOC_PAGES = 5
MIN_TOC_RIGHT_NUMBERS = 4
MIN_TOC_HEADING_ENTRIES = 4
MAX_TOC_LONG_LINE_RATIO = 0.3
```

### **Chunking**

```python
MIN_TOKENS = 800
MAX_TOKENS = 1500
MIN_HEADING_LEVEL_FOR_CHUNK_BREAK = 3
```

### **Heading Detection**

```python
MAX_WORDS_NUMBERED_HEADING = 50
MAX_WORDS_NON_NUMBERED_HEADING = 50
```

---

## **Output Format**

### **StructuralChunk**
Each chunk contains:

```json
{
  "document_name": "manual.pdf",
  "chunk_id": 12,
  "pages": [6],
  "heading_path": ["3 Overview", "3.1 Summary"],
  "text": "Full normalized text…",
  "token_count": 113,
  "chunk_type": "text",
  "image_paths": [],
  "blocks": [...]
}
```

### **Block Types**
- `heading`
- `paragraph`
- `list_item`
- `table`
- `caption`
- `sidebar`
- `image`

---

## **Usage Example**

```python
parser = PDFStructuralParser(
    "docs/auf_pos_vor103.pdf",
    output_dir="output/json",
    export_spans=True,
    export_blocks=False
)

chunks = parser.parse()

for chunk in chunks:
    print(chunk.heading_path, chunk.text[:200])
```

---

## **Design Philosophy**

This parser is built around three principles:

### **1. Preserve semantic structure**
Headings define hierarchy, not chunk content.  
Chunks represent meaningful units of knowledge.

### **2. Avoid noise**
Headers, footers, and TOC pages are optional but strongly recommended to remove.

### **3. Be token‑aware**
Chunks respect token limits without breaking semantic boundaries.

---

## **Roadmap (v1.2+)**

- OCR integration (Tesseract or Azure Cognitive Services)  
- Table reconstruction (grid extraction)  
- Improved caption association  
- Inline formatting preservation (bold/italic spans)  
- Optional chunk overlap for RAG  

---

If you want, I can also prepare a **v1.1 CHANGELOG**, a **sample output folder**, or a **developer guide** for extending heuristics.