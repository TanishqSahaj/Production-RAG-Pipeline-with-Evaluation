from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import DEFAULT_K, DEFAULT_MODE, LLM_MODEL, RETRIEVAL_MODES
from src.rag import ask
from src.vectorstore import get_collection

app = FastAPI(
    title="Bamboo Structures RAG API",
    description="Hybrid retrieval (BM25 + dense + RRF + rerank) over bamboo design documents.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    k: int = Field(DEFAULT_K, ge=1, le=15)
    mode: str = Field(DEFAULT_MODE)


class Source(BaseModel):
    text: str
    source: str
    page: Optional[int] = None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    retrieval_mode: str
    latency_ms: int


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        n = get_collection().count()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "chunks": n, "model": LLM_MODEL, "modes": list(RETRIEVAL_MODES)}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    if body.mode not in RETRIEVAL_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {list(RETRIEVAL_MODES)}")
    try:
        result = ask(body.question, k=body.k, mode=body.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        model=result["model"],
        retrieval_mode=result["retrieval_mode"],
        latency_ms=result["latency_ms"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
