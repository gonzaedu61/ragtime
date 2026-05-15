import argparse
import sys
import json
import numpy as np
from pathlib import Path
from VectorDB_Factory import create_vectordb
from Embedders import HFEmbeddingBackend


# -----------------------------
# Short-mode field definition
# -----------------------------
SHORT_FIELDS = [
    "document_name",
    "pages",
    "chunk_type",
    "heading_path",
    "chunk_text",
    "distance"
]


# -----------------------------
# Field extraction
# -----------------------------
def extract_fields(entry, short=False):
    if not short:
        return entry  # full entry

    meta = entry["metadata"]
    return {
        "document_name": meta.get("document_name"),
        "pages": meta.get("pages"),
        "chunk_type": meta.get("chunk_type"),
        "heading_path": meta.get("heading_path"),
        "chunk_text": entry["text"],
        "distance": entry.get("distance"),
    }


# -----------------------------
# Summary computation
# -----------------------------
def compute_summary(results):
    summary = {}
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        doc = meta.get("document_name", "Unknown Document")
        summary[doc] = summary.get(doc, 0) + 1
    total = sum(summary.values())
    return summary, total


# -----------------------------
# Markdown formatting
# -----------------------------
def format_markdown(entry):
    md = []

    for key, value in entry.items():
        if key == "chunk_text":
            md.append("\n### Text\n" + (value or "") + "\n")
        else:
            md.append(f"- **{key.replace('_', ' ').title()}:** {value}")

    md.append("\n---\n")
    return "\n".join(md)


# -----------------------------
# Exporters
# -----------------------------
def export_json(results, filepath, short=False):
    filepath = Path(filepath)
    data = []

    for i in range(len(results["ids"][0])):
        full_entry = {
            "chunk_number": i + 1,
            "id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
            "text": results["documents"][0][i],
        }
        data.append(extract_fields(full_entry, short))

    summary, total = compute_summary(results)

    output = {
        "summary": {
            "per_document": summary,
            "total_chunks": total
        },
        "chunks": data
    }

    filepath.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"JSON exported to {filepath}")


def export_markdown(results, filepath, short=False):
    filepath = Path(filepath)

    summary, total = compute_summary(results)
    blocks = ["# Summary\n"]
    blocks.append(f"- **Total chunks:** {total}")
    for doc, count in summary.items():
        blocks.append(f"- **{doc}**: {count} chunks")

    blocks.append("\n# Query Results\n")

    for i in range(len(results["ids"][0])):
        full_entry = {
            "chunk_number": i + 1,
            "id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
            "text": results["documents"][0][i],
        }
        entry = extract_fields(full_entry, short)
        blocks.append(f"## Chunk {i+1}\n")
        blocks.append(format_markdown(entry))

    filepath.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Markdown exported to {filepath}")


# -----------------------------
# Main query logic
# -----------------------------
def query_chroma(vectordb, embedder, query_text, n_results=None,
                 markdown=False, short=False,
                 export_json_path=None, export_md_path=None,
                 max_distance=None):

    query_emb = embedder.embed([query_text])[0]
    query_emb = query_emb.detach().cpu().numpy()

    # Determine retrieval count
    if max_distance is not None:
        if n_results is not None:
            retrieval_count = n_results
        else:
            retrieval_count = 500  # large number for distance filtering
    else:
        retrieval_count = n_results if n_results is not None else 5

    results = vectordb.query(
        query_embeddings=[query_emb],
        n_results=retrieval_count
    )

    # -----------------------------
    # Apply distance filtering
    # -----------------------------
    if max_distance is not None:
        filtered = {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            if dist <= max_distance:
                filtered["ids"][0].append(results["ids"][0][i])
                filtered["distances"][0].append(dist)
                filtered["metadatas"][0].append(results["metadatas"][0][i])
                filtered["documents"][0].append(results["documents"][0][i])

        results = filtered

    # --- EXPORTS ---
    if export_json_path:
        export_json(results, export_json_path, short)

    if export_md_path:
        export_markdown(results, export_md_path, short)

    # Only print if no export flags were used
    no_export = not export_json_path and not export_md_path

    # Markdown printing
    if markdown and no_export:
        summary, total = compute_summary(results)
        print("# Summary\n")
        print(f"- **Total chunks:** {total}")
        for doc, count in summary.items():
            print(f"- **{doc}**: {count} chunks")

        print("\n# Query Results\n")
        for i in range(len(results["ids"][0])):
            full_entry = {
                "chunk_number": i + 1,
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "text": results["documents"][0][i],
            }
            entry = extract_fields(full_entry, short)
            print(f"## Chunk {i+1}\n")
            print(format_markdown(entry))
        sys.exit()

    # Default pretty-print
    if no_export:
        print("\n=== Query Results ===")
        for i in range(len(results["ids"][0])):
            print(f"\nResult {i+1}")
            print(f"ID:       {results['ids'][0][i]}")
            print(f"Distance: {results['distances'][0][i]}")
            print(f"Metadata: {results['metadatas'][0][i]}")
            print(f"Text:     {results['documents'][0][i]}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Query the Chroma Vector Database using semantic search."
    )

    parser.add_argument("query", type=str, help="The text query to search for.")
    parser.add_argument("--top", type=int, default=None,
                        help="Number of top results to retrieve.")
    parser.add_argument("--persist", type=str, default="./DATA/KBs/Test_KB/5_Vector_DB")
    parser.add_argument("--collection", type=str, default="Structural_Chunks")

    parser.add_argument("--md", action="store_true",
                        help="Print results in Markdown format.")

    parser.add_argument("-s", "--short", action="store_true",
                        help="Export only a reduced set of fields.")

    parser.add_argument("--export-json", type=str,
                        help="Export results to a JSON file.")

    parser.add_argument("--export-md", type=str,
                        help="Export results to a Markdown (.md) file.")

    parser.add_argument("--max-distance", type=float,
                        help="Maximum distance threshold for retrieved chunks.")

    args = parser.parse_args()

    print("Initializing embedding backend...")
    embedder = HFEmbeddingBackend("C:/Models/multilingual-e5-large")

    print("Initializing vector DB...")
    vectordb = create_vectordb(
        backend="chroma",
        collection_name=args.collection,
        persist_dir=args.persist
    )

    print("Running query...")
    query_chroma(
        vectordb,
        embedder,
        args.query,
        args.top,
        markdown=args.md,
        short=args.short,
        export_json_path=args.export_json,
        export_md_path=args.export_md,
        max_distance=args.max_distance
    )


if __name__ == "__main__":
    main()
