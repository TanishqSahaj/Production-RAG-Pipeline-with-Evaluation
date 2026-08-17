import json

import streamlit as st

from src.config import DEFAULT_K, DEFAULT_MODE, EVAL_DIR, LLM_MODEL, RETRIEVAL_MODES
from src.rag import ask
from src.vectorstore import get_collection

st.set_page_config(page_title="Bamboo Structures RAG", page_icon="🎋", layout="wide")

EXAMPLES = [
    ("Density & diameter", "What are the physical properties of bamboo like density and diameter?"),
    ("Culm grading", "What is the grading system for bamboo culms in IS 15912?"),
    ("Structural uses", "How is bamboo used in structural design and construction?"),
    ("vs steel & wood", "How does bamboo compare to steel and wood as a structural material?"),
    ("Taper limits", "What taper and curvature limits apply to structural bamboo?"),
]


def _eval_metrics() -> dict | None:
    path = EVAL_DIR / "results.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("summary") or data
    except Exception:
        return None


@st.cache_resource
def _warmup() -> int:
    from src.retriever import get_corpus
    from src.vectorstore import get_model

    get_model()
    n = get_collection().count()
    get_corpus()
    return n


with st.sidebar:
    st.header("Retrieval")
    mode = st.selectbox(
        "Mode",
        RETRIEVAL_MODES,
        index=RETRIEVAL_MODES.index(DEFAULT_MODE),
        help="hybrid_rerank = BM25 + dense fused with Reciprocal Rank Fusion, then a cross-encoder reranker.",
    )
    k = st.slider("Chunks (k)", 3, 10, DEFAULT_K)
    st.caption(f"Generator: `{LLM_MODEL}`")
    try:
        with st.spinner("Loading retrieval models (first launch only)…"):
            st.caption(f"Indexed chunks: **{_warmup()}**")
    except Exception:
        st.caption("Indexed chunks: unavailable")

    metrics = _eval_metrics()
    if metrics:
        st.divider()
        st.header("RAGAS (latest)")
        cols = st.columns(2)
        keys = [
            ("faithfulness", "Faithfulness"),
            ("context_precision", "Context precision"),
            ("context_recall", "Context recall"),
            ("answer_relevancy", "Answer relevancy"),
            ("hallucination_rate", "Hallucination rate"),
        ]
        shown = [(k, label) for k, label in keys if k in metrics]
        for i, (key, label) in enumerate(shown):
            with cols[i % 2]:
                val = metrics[key]
                st.metric(label, f"{val:.2f}" if isinstance(val, float) else val)

    with st.expander("How retrieval works"):
        st.markdown(
            """
1. **Dense** — MiniLM embeddings in ChromaDB  
2. **BM25** — keyword match over the same chunks  
3. **RRF** — merge the two ranked lists  
4. **Rerank** — `ms-marco-MiniLM` cross-encoder  
5. **Generate** — Gemini, grounded only on retrieved context
            """
        )

st.title("Bamboo Structures RAG")
st.caption(
    "Ask questions over IS 15912 and IIT Bombay TD 643 lecture notes. "
    "Answers are grounded in retrieved chunks — if it is not in the documents, the model should refuse. "
    "The first Gemini call can take a couple of minutes; later questions are faster."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

example_cols = st.columns(len(EXAMPLES))
clicked = None
for i, (label, q) in enumerate(EXAMPLES):
    if example_cols[i].button(label, use_container_width=True, help=q):
        clicked = q

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources · {msg.get('latency_ms', '?')} ms · {msg.get('mode', '')}"):
                for i, src in enumerate(msg["sources"], 1):
                    page = f", p.{src['page']}" if src.get("page") else ""
                    st.markdown(f"**{i}. {src['source']}{page}** · score `{src['score']}`")
                    st.write(src["text"])

prompt = st.chat_input("Ask about bamboo structures, grading, connections, durability…")
if clicked:
    prompt = clicked

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                result = ask(prompt, k=k, mode=mode)
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        st.markdown(result["answer"])
        with st.expander(f"Sources · {result['latency_ms']} ms · {result['retrieval_mode']}"):
            for i, src in enumerate(result["sources"], 1):
                page = f", p.{src['page']}" if src.get("page") else ""
                st.markdown(f"**{i}. {src['source']}{page}** · score `{src['score']}`")
                st.write(src["text"])
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "latency_ms": result["latency_ms"],
                "mode": result["retrieval_mode"],
            }
        )
