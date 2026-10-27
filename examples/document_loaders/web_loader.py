import os

os.environ.setdefault("USER_AGENT", "rag-pdf-qa-example/1.0")

from langchain_community.document_loaders import WebBaseLoader

docs = WebBaseLoader("https://www.example.com/").load()

print(len(docs))
print(docs[0].page_content)
