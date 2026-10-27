from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

NOTES_PATH = Path(__file__).resolve().parent / "notes.txt"

docs = TextLoader(str(NOTES_PATH), encoding="utf-8").load()

splitter = CharacterTextSplitter(separator="", chunk_size=10, chunk_overlap=1)
chunks = splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.page_content)
    print()
