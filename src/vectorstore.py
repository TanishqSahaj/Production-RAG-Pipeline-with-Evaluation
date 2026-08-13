# src/vectorstore.py
import chromadb
from sentence_transformers import SentenceTransformer
from src.ingest import load_documents, chunk_documents

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION  = "rag_docs"
DB_PATH     = "./chroma_db"

# module-level singletons — loaded once, reused across calls
_model  = None
_client = None

def get_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
    return _client.get_or_create_collection(COLLECTION)

def build_vectorstore():
    model = get_model()
    col   = get_collection()

    if col.count() > 0:
        print(f"DB already has {col.count()} chunks. Skipping rebuild.")
        return col

    docs   = load_documents()
    chunks = chunk_documents(docs)
    texts  = [c.page_content for c in chunks]
    metas  = [c.metadata     for c in chunks]
    ids    = [f"chunk_{i}"   for i in range(len(chunks))]

    print(f"Embedding {len(texts)} chunks...")
    embs = model.encode(texts, show_progress_bar=True).tolist()

    col.add(documents=texts, embeddings=embs, metadatas=metas, ids=ids)
    print(f"Done. {col.count()} chunks stored.")
    return col

if __name__ == "__main__":
    build_vectorstore()