import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("rag_docs")
llm = OllamaLLM(model="llama3.2:3b")

def retrieve(query: str, k: int = 5):
    emb = model.encode([query]).tolist()
    results = col.query(query_embeddings=emb, n_results=k)
    return results["documents"][0]

def ask(question: str) -> dict:
    chunks = retrieve(question)
    context = "\n\n".join(chunks)
    prompt = f"""Use the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
    answer = llm.invoke(prompt)
    return {"answer": answer, "sources": chunks}

if __name__ == "__main__":
    result = ask("What is the main topic of these documents?")
    print("Answer:", result["answer"])
    print("\nTop source chunk:")
    print(result["sources"][0][:300])