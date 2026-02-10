import argparse
import numpy as np
from VectorDB_Factory import create_vectordb
from Semantic_Chunker import HFEmbeddingBackend



def query_chroma(vectordb, embedder, query_text: str, n_results: int = 5):
    # 1. Embed the query
    query_emb = embedder.embed([query_text])[0]
    query_emb = query_emb.detach().cpu().numpy()

    # 2. Query Chroma
    results = vectordb.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )

    # 3. Pretty-print results
    print("\n=== Query Results ===")
    for i in range(len(results["ids"][0])):
        print(f"\nResult {i+1}")
        print(f"ID:       {results['ids'][0][i]}")
        print(f"Distance: {results['distances'][0][i]}")
        print(f"Metadata: {results['metadatas'][0][i]}")
        print(f"Text:     {results['documents'][0][i]}")


def main():
    parser = argparse.ArgumentParser(
        description="Query the Chroma Vector Database using semantic search."
    )

    parser.add_argument(
        "query",
        type=str,
        help="The text query to search for in the vector database."
    )

    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)."
    )

    parser.add_argument(
        "--persist",
        type=str,
        default="./DATA/chroma_store",
        help="Path to Chroma persistence directory."
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="default_collection",
        help="Chroma collection name."
    )

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
    query_chroma(vectordb, embedder, args.query, args.top)


if __name__ == "__main__":
    main()
