from langchain_core.documents import Document

from rag_qa import config
from rag_qa.ingest import chunk_id, load_documents, split_documents


def test_chunk_id_is_deterministic():
    a = Document(page_content="hello", metadata={"source": "a.pdf", "page": 0})
    b = Document(page_content="hello", metadata={"source": "a.pdf", "page": 0})
    assert chunk_id(a) == chunk_id(b)


def test_chunk_id_depends_on_page():
    a = Document(page_content="hello", metadata={"source": "a.pdf", "page": 0})
    b = Document(page_content="hello", metadata={"source": "a.pdf", "page": 1})
    assert chunk_id(a) != chunk_id(b)


def test_split_respects_chunk_size():
    doc = Document(page_content="word " * 2000, metadata={"source": "a.pdf", "page": 0})
    chunks = split_documents([doc])
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= config.CHUNK_SIZE for chunk in chunks)


def test_sample_pdf_loads():
    docs = load_documents()
    assert docs
    assert all(doc.page_content.strip() for doc in docs)
