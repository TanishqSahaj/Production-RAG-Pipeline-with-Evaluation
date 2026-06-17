import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_documents, chunk_documents

EMBED_MODEL = "all-MiniLM-L6-v2" # free, fast, good
COLLECTION = "rag_docs"

def build_vectorstore():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Loading & chunking documents...")
    docs = load_documents()
    chunks = chunk_documents(docs)

    print("Embedding chunks...")
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path="./chroma_db")
    col = client.get_or_create_collection(COLLECTION)
    col.add(documents=texts, embeddings=embeddings,
            metadatas=metadatas, ids=ids)
    print(f"Done. {col.count()} chunks in ChromaDB")

if __name__ == "__main__":
    build_vectorstore()