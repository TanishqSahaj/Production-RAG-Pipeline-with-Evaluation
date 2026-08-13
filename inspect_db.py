import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("rag_docs")

print(f"Total chunks in DB: {col.count()}\n")

results = col.get(limit=10, include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"--- Chunk {i+1} ({len(doc)} chars) | {meta.get('source', '?')} ---")
    print(repr(doc[:200]))
    print()