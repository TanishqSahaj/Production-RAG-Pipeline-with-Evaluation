# src/rag.py
from langchain_google_genai import ChatGoogleGenerativeAI
# To this
from langchain_core.messages import HumanMessage, SystemMessage
from src.retriever import retrieve
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions \
based strictly on the provided context. \
If the answer is not in the context, say "I don't know based on the provided documents." \
Do not make up information."""

def ask(question: str, k: int = 5) -> dict:
    chunks = retrieve(question, k=k)  # list of dicts: {text, source, score}
    
    context = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ])
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]
    
    response = llm.invoke(messages)
    
    # Fix: extract text from response correctly
    answer = response.content
    if isinstance(answer, list):          # newer LangChain returns a list of blocks
        answer = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in answer
        )
    
    return {
        "answer" : answer,
        "sources": [{"text": c["text"][:150], "score": c["score"]} for c in chunks],
        "model"  : "gemini-3.1-flash-lite"
    }

if __name__ == "__main__":
    q = "What are the physical properties of bamboo like density and diameter?"
    result = ask(q)
    print(f"\nQuestion: {q}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nTop source (score: {result['sources'][0]['score']}):")
    print(result['sources'][0]['text'])