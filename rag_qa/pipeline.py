import os
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_qa import config
from rag_qa.llm import get_llm
from rag_qa.retrieval import load_vectorstore, retrieve

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful tutor. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "{not_found}"
Explain clearly and in simple language. Do not invent facts.

Context:
{context}

Question: {question}

Answer:"""
)


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)


def format_source(doc):
    name = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page")
    return f"{name}, page {int(page) + 1}" if page is not None else name


class RAGPipeline:
    def __init__(self, store=None, llm=None):
        self.store = store or load_vectorstore()
        self.llm = llm or get_llm()
        self.chain = ANSWER_PROMPT | self.llm | StrOutputParser()

    def ask(self, question, strategy=config.DEFAULT_STRATEGY, k=config.DEFAULT_K):
        documents = retrieve(self.store, question, strategy=strategy, k=k, llm=self.llm)
        if not documents:
            return Answer(text=config.NOT_FOUND_MESSAGE)

        context = "\n\n---\n\n".join(doc.page_content for doc in documents)
        text = self.chain.invoke(
            {"context": context, "question": question, "not_found": config.NOT_FOUND_MESSAGE}
        )

        sources = []
        for doc in documents:
            label = format_source(doc)
            if label not in sources:
                sources.append(label)
        return Answer(text=text, sources=sources)
