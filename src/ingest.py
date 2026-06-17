from langchain_community.document_loaders import ( PyPDFLoader, DirectoryLoader, TextLoader )
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from pathlib import Path

def load_documents(data_dir: str = "data"):
    # Load PDFs
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    # Inspect first chunk
    print("\nFirst chunk preview:")
    print(chunks[0].page_content[:300])
    print("\nMetadata:", chunks[0].metadata)