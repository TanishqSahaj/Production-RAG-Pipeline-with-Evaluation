from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

COLLECTION = "rag_docs"
DB_PATH = str(ROOT / "chroma_db")
DATA_DIR = str(ROOT / "data")
EVAL_DIR = ROOT / "eval"

DEFAULT_K = 5
CANDIDATE_K = 20
RRF_K = 60

RETRIEVAL_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
DEFAULT_MODE = "hybrid_rerank"
