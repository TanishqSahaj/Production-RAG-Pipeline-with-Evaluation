from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GOOGLE_API_KEY, LLM_MODEL


@lru_cache(maxsize=1)
def get_llm():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key.")
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
        timeout=180,
    )
