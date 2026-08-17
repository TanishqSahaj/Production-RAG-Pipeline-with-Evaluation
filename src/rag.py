import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import DEFAULT_K, DEFAULT_MODE, LLM_MODEL
from src.llm import get_llm
from src.retriever import retrieve

SYSTEM_PROMPT = """You are a helpful assistant that answers questions \
based strictly on the provided context. \
If the answer is not in the context, say "I don't know based on the provided documents." \
Do not make up information. Cite source filenames when you use them."""


def _response_text(response) -> str:
    answer = response.content
    if isinstance(answer, list):
        parts = []
        for block in answer:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return " ".join(parts).strip()
    return str(answer).strip()


def ask(question: str, k: int = DEFAULT_K, mode: str = DEFAULT_MODE) -> dict:
    question = (question or "").strip()
    if len(question) < 3:
        raise ValueError("Question is too short.")

    t0 = time.perf_counter()
    chunks = retrieve(question, k=k, mode=mode)
    context = "\n\n".join(
        f"[Source: {c['source']}" + (f", p.{c['page']}" if c.get("page") else "") + f"]\n{c['text']}"
        for c in chunks
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    answer = _response_text(get_llm().invoke(messages))
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "answer": answer,
        "sources": [
            {
                "text": c["text"][:400],
                "source": c["source"],
                "page": c.get("page"),
                "score": c["score"],
            }
            for c in chunks
        ],
        "contexts": [c["text"] for c in chunks],
        "model": LLM_MODEL,
        "retrieval_mode": mode,
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    q = "What are the physical properties of bamboo like density and diameter?"
    result = ask(q)
    print(f"\nQuestion: {q}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nMode: {result['retrieval_mode']} | {result['latency_ms']} ms")
    print(f"Top source: {result['sources'][0]['source']} (score {result['sources'][0]['score']})")
