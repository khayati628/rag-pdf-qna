import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

def resolve_path(value, default):
    path = Path(value) if value else default
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


DATA_DIR = resolve_path(os.getenv("DATA_DIR"), BASE_DIR / "data")
CHROMA_DIR = resolve_path(os.getenv("CHROMA_DIR"), BASE_DIR / "chroma_db")
COLLECTION_NAME = "rag_collection"

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHAT_MODELS = [
    name.strip()
    for name in os.getenv("GEMINI_MODELS", "gemini-3.8-flash,gemini-3.5-flash-lite").split(",")
    if name.strip()
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 200

DEFAULT_K = 5
STRATEGIES = ("similarity", "mmr", "multiquery")
DEFAULT_STRATEGY = "similarity"
NOT_FOUND_MESSAGE = "I couldn't find this in the provided documents."
