from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from rag_qa import config

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are helping a retrieval system search a document collection.

Write 3 different search queries for the user's question.
Each query should express the same information need in different words.

Return only the queries, one per line, with no numbering and no explanations.

User question: {question}"""
)


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)


def load_vectorstore(embeddings=None, persist_directory=None):
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(persist_directory or config.CHROMA_DIR),
        embedding_function=embeddings or get_embeddings(),
    )


def generate_queries(llm, question, limit=3):
    raw = (MULTI_QUERY_PROMPT | llm | StrOutputParser()).invoke({"question": question})
    queries = [line.strip().lstrip("0123456789.-) ") for line in raw.splitlines() if line.strip()]
    return [question] + queries[:limit]


def dedupe(documents):
    seen = {}
    for doc in documents:
        seen.setdefault(doc.page_content, doc)
    return list(seen.values())


def retrieve(store, question, strategy=config.DEFAULT_STRATEGY, k=config.DEFAULT_K, llm=None):
    if strategy == "similarity":
        return store.similarity_search(question, k=k)

    if strategy == "mmr":
        return store.max_marginal_relevance_search(question, k=k, fetch_k=max(k * 4, 20))

    if strategy == "multiquery":
        if llm is None:
            raise ValueError("The multiquery strategy needs an LLM.")
        results = []
        for query in generate_queries(llm, question):
            results.extend(store.similarity_search(query, k=k))
        return dedupe(results)

    raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {', '.join(config.STRATEGIES)}.")
