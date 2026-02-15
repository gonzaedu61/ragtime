import json
import os
from glob import glob
from transformers import AutoTokenizer


class Tokens_Chunker:
    """
    Converts structural chunks into token-based chunks with full traceability:
    - merges small structural text chunks (heading-safe)
    - preserves image/mixed chunks as atomic units
    - token-splits merged text chunks
    - preserves all metadata for reconstruction
    - associates image_paths to related text chunks
    - writes <filename>_Token_Chunks.json
    """

    def __init__(self,
                 chunks_path: str,
                 chunk_files_pattern: str,
                 output_path: str,
                 model_name: str,
                 min_tokens=80,
                 max_tokens=384,
                 overlap=64,
                 use_block_proximity=False,
                 verbose=False):

        self.chunks_path = chunks_path
        self.chunk_files_pattern = chunk_files_pattern
        self.output_path = output_path
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.use_block_proximity = use_block_proximity
        self.verbose = verbose

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = 10_000_000  # disable HF max-length warnings

        os.makedirs(self.output_path, exist_ok=True)

    # ---------------------------------------------------------
    # Utility: count tokens
    # ---------------------------------------------------------
    def count_tokens(self, text):
        tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(tokens)

    # ---------------------------------------------------------
    # STEP 1: Merge small structural chunks (heading-safe)
    # ---------------------------------------------------------
    def merge_small_chunks(self, structural_chunks):
        merged_groups = []
        buffer_group = []

        for chunk in structural_chunks:

            # IMAGE or MIXED chunks → flush buffer and keep atomic
            if chunk["chunk_type"] in ("image", "mixed"):
                if buffer_group:
                    merged_groups.append(buffer_group)
                    buffer_group = []
                merged_groups.append([chunk])
                continue

            # TEXT chunks only
            if not buffer_group:
                buffer_group = [chunk]
                continue

            # Enforce top-level heading boundary
            current_heading = chunk["heading_path"][0]
            buffer_heading = buffer_group[0]["heading_path"][0]

            if current_heading != buffer_heading:
                merged_groups.append(buffer_group)
                buffer_group = [chunk]
                continue

            # Try merging buffer + current
            combined_text = " ".join([c["text"] for c in buffer_group] + [chunk["text"]])
            token_count = self.count_tokens(combined_text)

            if token_count < self.min_tokens:
                buffer_group.append(chunk)
            else:
                merged_groups.append(buffer_group)
                buffer_group = [chunk]

        if buffer_group:
            merged_groups.append(buffer_group)

        return merged_groups

    # ---------------------------------------------------------
    # STEP 2: Token-split merged text chunks
    # ---------------------------------------------------------
    def split_by_tokens(self, text):
        # Safe, warning-free tokenization
        tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.max_tokens
            chunk_tokens = tokens[start:end]

            # Fully warning-free decoding
            token_list = self.tokenizer.convert_ids_to_tokens(
                chunk_tokens,
                skip_special_tokens=True
            )
            chunk_text = self.tokenizer.convert_tokens_to_string(token_list)

            chunks.append({
                "token_start": start,
                "token_end": min(end, len(tokens)),
                "text": chunk_text
            })

            start += self.max_tokens - self.overlap

        return chunks

    # ---------------------------------------------------------
    # STEP 3: Build final token-based chunks with full traceability
    # ---------------------------------------------------------
    def build_token_chunks(self, merged_groups):
        final_chunks = []
        token_chunk_id = 1

        for group in merged_groups:

            # IMAGE or MIXED chunk → keep atomic
            if len(group) == 1 and group[0]["chunk_type"] in ("image", "mixed"):
                c = group[0]
                final_chunks.append({
                    "token_chunk_id": token_chunk_id,
                    "document_name": c["document_name"],
                    "heading_path": c["heading_path"],
                    "pages": c["pages"],
                    "chunk_type": c["chunk_type"],
                    "image_paths": c.get("image_paths", []),
                    "associated_images": [],

                    # Traceability
                    "merged_from": [c["chunk_id"]],
                    "merged_original_texts": {c["chunk_id"]: c.get("text", "")},
                    "merged_original_pages": {c["chunk_id"]: c["pages"]},
                    "merged_original_headings": {c["chunk_id"]: c["heading_path"]},
                    "merged_original_blocks": {c["chunk_id"]: c.get("blocks", [])},

                    "token_start": None,
                    "token_end": None,
                    "chunk_text": c.get("text", "")
                })
                token_chunk_id += 1
                continue

            # TEXT chunks: merge + token-split
            merged_text = " ".join([c["text"] for c in group])
            token_chunks = self.split_by_tokens(merged_text)

            merged_from_ids = [c["chunk_id"] for c in group]
            merged_original_texts = {c["chunk_id"]: c["text"] for c in group}
            merged_original_pages = {c["chunk_id"]: c["pages"] for c in group}
            merged_original_headings = {c["chunk_id"]: c["heading_path"] for c in group}
            merged_original_blocks = {c["chunk_id"]: c.get("blocks", []) for c in group}

            dominant_heading = group[0]["heading_path"]
            doc_name = group[0]["document_name"]
            pages = sorted({p for c in group for p in c["pages"]})

            for tc in token_chunks:
                final_chunks.append({
                    "token_chunk_id": token_chunk_id,
                    "document_name": doc_name,
                    "heading_path": dominant_heading,
                    "pages": pages,
                    "chunk_type": "text",
                    "image_paths": [],
                    "associated_images": [],

                    "merged_from": merged_from_ids,
                    "merged_original_texts": merged_original_texts,
                    "merged_original_pages": merged_original_pages,
                    "merged_original_headings": merged_original_headings,
                    "merged_original_blocks": merged_original_blocks,

                    "token_start": tc["token_start"],
                    "token_end": tc["token_end"],
                    "chunk_text": tc["text"]
                })
                token_chunk_id += 1

        return final_chunks

    # ---------------------------------------------------------
    # STEP 4: Block proximity helpers
    # ---------------------------------------------------------
    @staticmethod
    def _block_vertical_center(block):
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            return None
        return (bbox[1] + bbox[3]) / 2.0

    def _distance_image_to_text(self, image_blocks, text_blocks):
        img_centers = [
            self._block_vertical_center(b)
            for b in image_blocks
            if self._block_vertical_center(b) is not None
        ]
        txt_centers = [
            self._block_vertical_center(b)
            for b in text_blocks
            if self._block_vertical_center(b) is not None
        ]

        if not img_centers or not txt_centers:
            return None

        return min(abs(ic - tc) for ic in img_centers for tc in txt_centers)

    # ---------------------------------------------------------
    # STEP 5: Automatic image association
    # ---------------------------------------------------------
    def associate_images(self, token_chunks):
        image_chunks = [c for c in token_chunks if c["chunk_type"] in ("image", "mixed") and c.get("image_paths")]
        text_chunks = [c for c in token_chunks if c["chunk_type"] == "text"]

        for img in image_chunks:
            doc = img["document_name"]
            img_pages = set(img.get("pages", []))
            img_heading = img.get("heading_path")
            img_blocks = [b for blocks in img.get("merged_original_blocks", {}).values() for b in blocks]

            # Candidates: same doc + overlapping pages
            candidates = [
                t for t in text_chunks
                if t["document_name"] == doc and img_pages.intersection(t.get("pages", []))
            ]

            # Prefer same heading
            same_heading = [t for t in candidates if t.get("heading_path") == img_heading]
            if same_heading:
                candidates = same_heading

            # Block proximity
            if self.use_block_proximity and img_blocks:
                best = None
                best_dist = None
                for t in candidates:
                    txt_blocks = [b for blocks in t.get("merged_original_blocks", {}).values() for b in blocks]
                    dist = self._distance_image_to_text(img_blocks, txt_blocks)
                    if dist is not None and (best_dist is None or dist < best_dist):
                        best = t
                        best_dist = dist
                if best:
                    candidates = [best]

            # Attach image paths
            for t in candidates:
                for ip in img.get("image_paths", []):
                    if ip not in t["associated_images"]:
                        t["associated_images"].append(ip)

        return token_chunks

    # ---------------------------------------------------------
    # PROCESS A SINGLE FILE
    # ---------------------------------------------------------
    def process_single_file(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            structural_chunks = json.load(f)

        merged_groups = self.merge_small_chunks(structural_chunks)
        token_chunks = self.build_token_chunks(merged_groups)
        token_chunks = self.associate_images(token_chunks)

        out_name = os.path.basename(json_file).replace(".json", "_tokenized.json")
        out_path = os.path.join(self.output_path, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(token_chunks, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"Generated {out_name} with {len(token_chunks)} token chunks")

        return out_path

    # ---------------------------------------------------------
    # PROCESS ALL FILES
    # ---------------------------------------------------------
    def run(self):
        pattern = os.path.join(self.chunks_path, self.chunk_files_pattern)
        files = glob(pattern)

        outputs = []
        for f in files:
            outputs.append(self.process_single_file(f))

        return outputs
