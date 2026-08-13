# src/ingest.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def load_documents(data_dir: str = "data"):
    all_docs = []
    pdf_files = list(Path(data_dir).rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            # Filter out slide pages with very little text (headings, page numbers)
            usable = [d for d in docs if len(d.page_content.strip()) > 150]
            all_docs.extend(usable)
            print(f"  ✓ {pdf_path.name}: {len(docs)} pages → {len(usable)} usable")
        except Exception as e:
            print(f"  ✗ {pdf_path.name} skipped: {e}")

    print(f"\nTotal usable pages: {len(all_docs)}")
    return all_docs

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # bigger chunks — slides need more context
        chunk_overlap=150,     # more overlap so ideas don't get cut off
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Drop chunks that are still too short after splitting
    chunks = [c for c in chunks if len(c.page_content.strip()) > 150]
    print(f"Created {len(chunks)} meaningful chunks")
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    print("\nSample chunk:")
    print(chunks[0].page_content)
    print(f"\nChunk length: {len(chunks[0].page_content)} chars")