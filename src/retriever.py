"""Dense, BM25, hybrid (RRF), and cross-encoder reranked retrieval."""
from __future__ import annotations

import re
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import CANDIDATE_K, DEFAULT_K, DEFAULT_MODE, RERANK_MODEL, RRF_K
from src.vectorstore import get_collection, get_model

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_corpus = None
_reranker = None


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _source_name(path: str) -> str:
    if not path:
        return "unknown"
    return str(path).replace("\\", "/").split("/")[-1]


def _chunk_from_parts(doc_id: str, text: str, meta: dict | None, score: float) -> dict:
    meta = meta or {}
    page = meta.get("page")
    try:
        page = int(page) + 1 if page is not None else None
    except (TypeError, ValueError):
        page = meta.get("page")
    return {
        "id": doc_id,
        "text": text,
        "source": _source_name(meta.get("source", "unknown")),
        "source_path": meta.get("source", "unknown"),
        "page": page,
        "score": round(float(score), 4),
    }


class _Corpus:
    def __init__(self):
        col = get_collection()
        data = col.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.texts = data["documents"]
        self.metas = data["metadatas"]
        self.id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

    def chunk(self, doc_id: str, score: float) -> dict:
        idx = self.id_to_idx[doc_id]
        return _chunk_from_parts(doc_id, self.texts[idx], self.metas[idx], score)


def get_corpus() -> _Corpus:
    global _corpus
    if _corpus is None:
        _corpus = _Corpus()
    return _corpus


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rrf_fuse(rank_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _dense(query: str, k: int) -> list[dict]:
    col = get_collection()
    n = min(k, max(col.count(), 1))
    emb = get_model().encode([query]).tolist()
    results = col.query(
        query_embeddings=emb,
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for i, doc_id in enumerate(results["ids"][0]):
        chunks.append(
            _chunk_from_parts(
                doc_id,
                results["documents"][0][i],
                results["metadatas"][0][i],
                1 - float(results["distances"][0][i]),
            )
        )
    return chunks


def _bm25(query: str, k: int) -> list[dict]:
    corpus = get_corpus()
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = np.asarray(corpus.bm25.get_scores(tokens), dtype=float)
    top = np.argsort(scores)[::-1][:k]
    return [corpus.chunk(corpus.ids[int(i)], float(scores[int(i)])) for i in top]


def _rerank(query: str, chunks: list[dict], k: int) -> list[dict]:
    if not chunks:
        return []
    try:
        scores = get_reranker().predict([(query, c["text"]) for c in chunks])
    except Exception as exc:
        print(f"Reranker unavailable, falling back to hybrid order: {exc}")
        return chunks[:k]
    ranked = sorted(zip(chunks, scores), key=lambda pair: float(pair[1]), reverse=True)
    out = []
    for chunk, score in ranked[:k]:
        item = dict(chunk)
        item["score"] = round(float(score), 4)
        out.append(item)
    return out


def retrieve(
    query: str,
    k: int = DEFAULT_K,
    mode: str = DEFAULT_MODE,
    candidate_k: int = CANDIDATE_K,
) -> list[dict]:
    """Return top-k chunks. mode: dense | bm25 | hybrid | hybrid_rerank."""
    mode = (mode or DEFAULT_MODE).strip()
    k = max(1, int(k))
    pool = max(k, int(candidate_k))

    if mode == "dense":
        return _dense(query, k)
    if mode == "bm25":
        return _bm25(query, k)

    dense = _dense(query, pool)
    sparse = _bm25(query, pool)
    fused = rrf_fuse([[c["id"] for c in dense], [c["id"] for c in sparse]])
    corpus = get_corpus()
    merged = [corpus.chunk(doc_id, score) for doc_id, score in fused if doc_id in corpus.id_to_idx]

    if mode == "hybrid":
        return merged[:k]
    return _rerank(query, merged[:pool], k)


if __name__ == "__main__":
    for item in retrieve("bamboo physical properties density diameter", k=3):
        print(f"\n--- {item['source']} p.{item['page']} score={item['score']} ---")
        print(item["text"][:220])
