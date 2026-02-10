from VectorDB_Factory import create_vectordb


# Initialize vector DB backend (Chroma or others)
print('Initializing vector DB ...')
vectordbName = "chroma"
collection_name="default_collection"
persist_dir="./DATA/chroma_store"
vectordb = create_vectordb(
    backend=vectordbName,
    collection_name=collection_name,
    persist_dir=persist_dir
)
print('Vector DB Object created ...')

# Dump records from vector DB
all_docs = vectordb.get()
for i in range(len(all_docs["ids"])):
    print("\n--- Document", i+1, "---")
    print("ID:", all_docs["ids"][i])
    print("Metadata:", all_docs["metadatas"][i])
    print("Text:", all_docs["documents"][i])

