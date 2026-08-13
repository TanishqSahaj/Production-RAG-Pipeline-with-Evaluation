# notebooks/test_queries.py
from src.vectorstore import build_vectorstore
from src.rag import ask

build_vectorstore() # skips if already built

test_questions = [
    "What are the physical properties of bamboo like density and diameter?",
    "What are the chemical properties of bamboo?",
    "How is bamboo used in structural design and construction?",
    "What is the grading system for bamboo culms?",
    "How does bamboo compare to steel and wood as a structural material?",
]

for q in test_questions:
    print(f"\n{'='*50}")
    print(f"Q: {q}")
    result = ask(q)
    print(f"A: {result['answer']}")
    print(f"Top chunk score: {result['sources'][0]['score']}")