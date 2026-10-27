import argparse
import hashlib
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_qa import config
from rag_qa.retrieval import get_embeddings


def load_documents(data_dir=None):
    data_dir = data_dir or config.DATA_DIR
    return PyPDFDirectoryLoader(str(data_dir), recursive=True).load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def chunk_id(chunk):
    key = "|".join(
        [
            str(chunk.metadata.get("source", "")),
            str(chunk.metadata.get("page", "")),
            chunk.page_content,
        ]
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def save_to_chroma(chunks, embeddings=None, persist_directory=None):
    embeddings = embeddings or get_embeddings()
    persist_directory = str(persist_directory or config.CHROMA_DIR)
    db = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    unique = {chunk_id(chunk): chunk for chunk in chunks}
    existing = set(db.get()["ids"])
    pending = [(cid, chunk) for cid, chunk in unique.items() if cid not in existing]

    print(f"Already stored: {len(existing)} | New chunks: {len(pending)}")
    if not pending:
        print("Nothing to add. The vector store is up to date.")
        return 0

    for start in range(0, len(pending), config.BATCH_SIZE):
        batch = pending[start : start + config.BATCH_SIZE]
        db.add_documents([chunk for _, chunk in batch], ids=[cid for cid, _ in batch])
        done = min(start + config.BATCH_SIZE, len(pending))
        print(f"  stored {done}/{len(pending)} new chunks")

    print("Vector store updated.")
    return len(pending)


def build_index(reset=False, embeddings=None):
    if reset and config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
        print(f"Removed existing index at '{config.CHROMA_DIR}'.")

    documents = load_documents()
    if not documents:
        print(f"No PDFs found in '{config.DATA_DIR}'. Add some and run again.")
        return 0
    print(f"Loaded {len(documents)} pages.")

    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    return save_to_chroma(chunks, embeddings=embeddings)


def main():
    parser = argparse.ArgumentParser(description="Build the ChromaDB index from PDFs in the data folder.")
    parser.add_argument("--reset", action="store_true", help="delete the existing index and rebuild it from scratch")
    args = parser.parse_args()
    build_index(reset=args.reset)


if __name__ == "__main__":
    main()
