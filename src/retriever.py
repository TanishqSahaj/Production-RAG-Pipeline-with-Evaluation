# src/retriever.py
from src.vectorstore import get_model, get_collection

def retrieve(query: str, k: int = 5) -> list:
    """Return top-k most relevant text chunks for a query."""
    model = get_model()
    col   = get_collection()
    emb   = model.encode([query]).tolist()

    results = col.query(
        query_embeddings=emb,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "text"  : c,
            "source": m.get("source", "unknown"),
            "score" : round(1 - d, 3)
        }
        for c, m, d in zip(chunks, metadatas, distances)
    ]

if __name__ == "__main__":
    results = retrieve("bamboo physical properties density diameter", k=3)
    for i, r in enumerate(results):
        print(f"\n--- Chunk {i+1} (score: {r['score']}) ---")
        print(f"Source: {r['source']}")
        print(r['text'][:200])