"""Run RAGAS (or a Gemini judge fallback) on the golden set.

Usage from repo root:
    .\\venv\\Scripts\\python.exe eval\\run_eval.py
    .\\venv\\Scripts\\python.exe eval\\run_eval.py --modes dense,hybrid_rerank
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import EVAL_DIR, LLM_MODEL
from src.llm import get_llm
from src.rag import ask

GOLDEN_PATH = EVAL_DIR / "golden_set.json"


def load_items() -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["items"]


def run_pipeline(items: list[dict], mode: str, k: int) -> list[dict]:
    rows = []
    for item in items:
        result = ask(item["question"], k=k, mode=mode)
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "sources": result["sources"],
                "latency_ms": result["latency_ms"],
            }
        )
        print(f"  {item['id']}: {result['latency_ms']} ms | {result['answer'][:80]!r}")
    return rows


def _try_ragas(rows: list[dict]) -> dict | None:
    try:
        from src.ragas_compat import apply as _patch_ragas

        _patch_ragas()
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
    except Exception as exc:
        print(f"RAGAS import failed ({exc}). Using Gemini judge fallback.")
        return None

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            retrieved_contexts=row["contexts"],
            response=row["answer"],
            reference=row["ground_truth"],
        )
        for row in rows
    ]
    dataset = EvaluationDataset(samples=samples)
    judge = LangchainLLMWrapper(get_llm(), bypass_n=True)
    metrics = [
        Faithfulness(llm=judge),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
        ResponseRelevancy(llm=judge),
    ]
    kwargs = {"dataset": dataset, "metrics": metrics, "llm": judge}
    if HuggingFaceEmbeddings is not None:
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
        kwargs["embeddings"] = embeddings

    print("Running RAGAS metrics (this can take several minutes)...")
    result = evaluate(**kwargs)
    df = result.to_pandas()
    summary = {}
    mapping = {
        "faithfulness": "faithfulness",
        "llm_context_precision_with_reference": "context_precision",
        "context_precision": "context_precision",
        "llm_context_recall": "context_recall",
        "context_recall": "context_recall",
        "answer_relevancy": "answer_relevancy",
        "response_relevancy": "answer_relevancy",
    }
    for col in df.columns:
        key = mapping.get(col)
        if key and df[col].dtype != object:
            summary[key] = round(float(df[col].mean()), 4)
    if "faithfulness" in summary:
        summary["hallucination_rate"] = round(1.0 - summary["faithfulness"], 4)
    return {
        "engine": "ragas",
        "summary": summary,
        "per_sample": df.to_dict(orient="records"),
    }


def _gemini_judge(rows: list[dict]) -> dict:
    llm = get_llm()
    per_sample = []
    for row in rows:
        prompt = f"""You are evaluating a RAG system. Score each metric from 0 to 1.
Return ONLY JSON with keys: faithfulness, context_precision, context_recall, answer_relevancy, notes.

Definitions:
- faithfulness: fraction of answer claims supported by the retrieved contexts (1 = no hallucination)
- context_precision: retrieved chunks that are useful are ranked near the top
- context_recall: retrieved contexts cover the ground-truth answer
- answer_relevancy: the answer addresses the question

Question: {row['question']}
Ground truth: {row['ground_truth']}
Answer: {row['answer']}
Contexts:
{chr(10).join(f'- {c[:500]}' for c in row['contexts'])}
"""
        raw = llm.invoke(prompt).content
        if isinstance(raw, list):
            raw = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
            )
        text = str(raw)
        start, end = text.find("{"), text.rfind("}")
        scores = {"faithfulness": 0.0, "context_precision": 0.0, "context_recall": 0.0, "answer_relevancy": 0.0}
        if start != -1 and end != -1:
            try:
                parsed = json.loads(text[start : end + 1])
                for key in scores:
                    scores[key] = float(parsed.get(key, 0))
            except Exception:
                pass
        scores["id"] = row["id"]
        per_sample.append(scores)
        print(f"  judged {row['id']}: faithfulness={scores['faithfulness']}")

    n = max(len(per_sample), 1)
    summary = {
        key: round(sum(s[key] for s in per_sample) / n, 4)
        for key in ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
    }
    summary["hallucination_rate"] = round(1.0 - summary["faithfulness"], 4)
    return {"engine": "gemini_judge_fallback", "summary": summary, "per_sample": per_sample}


def evaluate_mode(items: list[dict], mode: str, k: int) -> dict:
    print(f"\n=== Pipeline mode={mode} ===")
    rows = run_pipeline(items, mode=mode, k=k)
    ragas_result = _try_ragas(rows)
    judged = ragas_result if ragas_result and ragas_result.get("summary") else _gemini_judge(rows)
    return {
        "mode": mode,
        "k": k,
        "generator": LLM_MODEL,
        "engine": judged["engine"],
        "summary": judged["summary"],
        "samples": [
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "latency_ms": row["latency_ms"],
                "sources": [s["source"] for s in row["sources"]],
            }
            for row in rows
        ],
        "per_sample_scores": judged.get("per_sample", []),
    }


def write_report(payload: dict) -> None:
    lines = [
        "# RAG evaluation report",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Generator: `{payload['generator']}`",
        "",
        "## Summary",
        "",
        "| Mode | Engine | Faithfulness | Context precision | Context recall | Answer relevancy | Hallucination rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in payload["runs"]:
        s = run["summary"]
        lines.append(
            "| {mode} | {engine} | {faithfulness:.3f} | {context_precision:.3f} | {context_recall:.3f} | {answer_relevancy:.3f} | {hallucination_rate:.3f} |".format(
                mode=run["mode"],
                engine=run["engine"],
                faithfulness=s.get("faithfulness", 0),
                context_precision=s.get("context_precision", 0),
                context_recall=s.get("context_recall", 0),
                answer_relevancy=s.get("answer_relevancy", 0),
                hallucination_rate=s.get("hallucination_rate", 0),
            )
        )
    lines += ["", "These scores are computed on `eval/golden_set.json` (10 domain questions)."]
    (EVAL_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="hybrid_rerank", help="Comma-separated retrieval modes")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N golden questions")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    items = load_items()
    if args.limit:
        items = items[: args.limit]
    runs = [evaluate_mode(items, mode, args.k) for mode in modes]
    primary = runs[-1]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": LLM_MODEL,
        "summary": primary["summary"],
        "runs": runs,
    }
    EVAL_DIR.mkdir(exist_ok=True)
    (EVAL_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload)
    print("\nSaved eval/results.json and eval/report.md")
    print(json.dumps(primary["summary"], indent=2))


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Total eval time: {time.perf_counter() - t0:.1f}s")
