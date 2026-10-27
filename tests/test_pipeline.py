import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_qa import config
from rag_qa.ingest import chunk_id, save_to_chroma
from rag_qa.pipeline import RAGPipeline, format_source
from rag_qa.retrieval import dedupe, generate_queries, retrieve


@pytest.fixture()
def store(tmp_path):
    embeddings = DeterministicFakeEmbedding(size=32)
    chunks = [
        Document(page_content="Gradient descent minimizes a loss function.", metadata={"source": "/x/ml.pdf", "page": 2}),
        Document(page_content="Support vector machines find a maximum margin hyperplane.", metadata={"source": "/x/ml.pdf", "page": 7}),
        Document(page_content="Chroma stores embeddings for similarity search.", metadata={"source": "/x/db.pdf", "page": 0}),
    ]
    save_to_chroma(chunks, embeddings=embeddings, persist_directory=tmp_path)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(tmp_path),
        embedding_function=embeddings,
    )


def test_save_to_chroma_is_idempotent(tmp_path):
    embeddings = DeterministicFakeEmbedding(size=32)
    chunks = [Document(page_content="same text", metadata={"source": "a.pdf", "page": 0})]
    assert save_to_chroma(chunks, embeddings=embeddings, persist_directory=tmp_path) == 1
    assert save_to_chroma(chunks, embeddings=embeddings, persist_directory=tmp_path) == 0
    assert chunk_id(chunks[0])


@pytest.mark.parametrize("strategy", ["similarity", "mmr"])
def test_retrieve_returns_k_documents(store, strategy):
    docs = retrieve(store, "What is gradient descent?", strategy=strategy, k=2)
    assert len(docs) == 2


def test_multiquery_dedupes_and_uses_llm(store):
    llm = FakeListChatModel(responses=["define gradient descent\nloss minimization\noptimization method"])
    docs = retrieve(store, "gradient descent", strategy="multiquery", k=2, llm=llm)
    contents = [doc.page_content for doc in docs]
    assert len(contents) == len(set(contents))


def test_multiquery_requires_llm(store):
    with pytest.raises(ValueError):
        retrieve(store, "anything", strategy="multiquery")


def test_unknown_strategy_raises(store):
    with pytest.raises(ValueError):
        retrieve(store, "anything", strategy="nope")


def test_generate_queries_strips_numbering():
    llm = FakeListChatModel(responses=["1. first query\n2) second query\n- third query"])
    queries = generate_queries(llm, "original")
    assert queries == ["original", "first query", "second query", "third query"]


def test_dedupe_keeps_first_occurrence():
    a = Document(page_content="x", metadata={"page": 1})
    b = Document(page_content="x", metadata={"page": 2})
    assert dedupe([a, b]) == [a]


def test_format_source_uses_one_based_pages():
    doc = Document(page_content="x", metadata={"source": "/some/dir/book.pdf", "page": 0})
    assert format_source(doc) == "book.pdf, page 1"


def test_pipeline_returns_answer_with_unique_sources(store):
    llm = FakeListChatModel(responses=["Gradient descent minimizes the loss."])
    pipeline = RAGPipeline(store=store, llm=llm)
    answer = pipeline.ask("What is gradient descent?", k=3)
    assert answer.text == "Gradient descent minimizes the loss."
    assert len(answer.sources) == len(set(answer.sources))
    assert "ml.pdf, page 3" in answer.sources
