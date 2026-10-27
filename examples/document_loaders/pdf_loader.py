from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path(__file__).resolve().parents[2] / "data" / "AI_ML_Notes.pdf"

docs = PyPDFLoader(str(PDF_PATH)).load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunks = splitter.split_documents(docs)

print(f"{len(docs)} pages -> {len(chunks)} chunks")
print(chunks[0].page_content)
