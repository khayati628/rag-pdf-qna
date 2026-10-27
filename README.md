# RAG PDF Q&A

A retrieval-augmented generation (RAG) application that answers questions about PDF documents and shows the source file and page for retrieved context. It includes a Streamlit web interface, a terminal interface, an incremental ChromaDB index, and offline unit tests.

## Features

- PDF ingestion with page metadata and overlapping text chunks.
- Local sentence-transformer embeddings; document text and vectors are stored locally.
- ChromaDB vector search with similarity, maximal marginal relevance (MMR), and multi-query retrieval.
- Google Gemini generation with configurable model fallback.
- Source labels containing the PDF filename and one-based page number.
- Streamlit and command-line interfaces.
- Automated tests using fake embeddings and a fake chat model, without requiring an API key.

## How it works

1. PDFs are loaded from `data/` and split into overlapping chunks.
2. The embedding model converts each chunk into a vector.
3. Chunks and metadata are stored in `chroma_db/`.
4. A question retrieves relevant chunks using the selected retrieval strategy.
5. Gemini generates an answer constrained to the retrieved context, and the interface displays source pages.

The system can still return incorrect or incomplete answers. Treat source pages as a way to verify the answer rather than as a guarantee of correctness.

## Project structure

```text
rag-pdf-qa/
├── app.py
├── rag_qa/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── retrieval.py
│   ├── llm.py
│   ├── pipeline.py
│   └── cli.py
├── data/
│   └── AI_ML_Notes.pdf
├── examples/
│   ├── arxiv_search.py
│   └── document_loaders/
├── tests/
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── requirements-examples.txt
```

## Requirements

- Python 3.11 or newer
- A Google AI Studio API key for asking questions
- Internet access for installing packages, downloading the embedding model the first time, and calling Gemini

The included sample PDF can be used to try the project. Scanned PDFs without a text layer are not supported unless OCR is performed first.

## Setup

### 1. Clone the repository and create an environment

```bash
git clone https://github.com/<github-username>/rag-pdf-qa.git
cd rag-pdf-qa
python -m venv .venv
```

Activate the environment:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Gemini

Create a `.env` file from the example:

**macOS / Linux**

```bash
cp .env.example .env
```

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

Create an API key in [Google AI Studio](https://aistudio.google.com/apikey), then replace the placeholder value for `GOOGLE_API_KEY` in `.env`. Do not commit `.env` or share the key. The repository's `.gitignore` excludes it.

### 3. Build the document index

Put PDF files in `data/`. Nested folders are searched recursively. Then run:

```bash
python -m rag_qa.ingest
```

The first run downloads the embedding model, which can take time and disk space. The vector database is created in `chroma_db/`, which is excluded from Git.

Re-run ingestion after adding PDFs. If you edit or remove PDFs, rebuild the index so old chunks do not remain:

```bash
python -m rag_qa.ingest --reset
```

**Warning:** `--reset` deletes the existing local vector index before rebuilding it.

### 4. Start the app

Run the Streamlit interface:

```bash
streamlit run app.py
```

Or use the terminal interface:

```bash
python -m rag_qa.cli
```

Choose retrieval options in the Streamlit sidebar or pass them to the CLI:

```bash
python -m rag_qa.cli --strategy mmr -k 5
python -m rag_qa.cli --strategy multiquery -k 5
```

Available strategies are `similarity`, `mmr`, and `multiquery`. Multi-query retrieval makes an additional Gemini request to generate alternative search queries.

## Configuration

Configuration is read from `.env`. Defaults are shown below.

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | Not set | Required for Gemini responses |
| `GEMINI_MODELS` | `gemini-3.8-flash,gemini-3.5-flash-lite` | Comma-separated model IDs, tried in order |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `DATA_DIR` | `./data` | Directory scanned for PDFs |
| `CHROMA_DIR` | `./chroma_db` | Persistent vector database directory |

Chunk size, chunk overlap, batch size, and retrieval defaults are configured in `rag_qa/config.py`. Relative directory settings are resolved relative to the project root.

## Tests

Install the main dependencies, then run:

```bash
python -m pytest
```

Tests use temporary vector stores and fake models for the RAG pipeline. They do not require a Gemini API key, but they do require the project dependencies to be installed.

## Optional examples

The scripts in `examples/` demonstrate separate PDF, text, and web loaders and an arXiv search. Install their additional dependencies with:

```bash
pip install -r requirements-examples.txt
python examples/document_loaders/pdf_loader.py
python examples/document_loaders/text_loader.py
python examples/document_loaders/web_loader.py
python examples/arxiv_search.py
```

The web loader requires network access. The arXiv example queries the public arXiv API.

## Limitations

- Supports text-based PDFs only; OCR is not included.
- Answers are grounded in retrieved chunks, but retrieval and generation can still miss relevant information or produce errors.
- Source labels refer to retrieved chunks, not necessarily every source used implicitly by the model.
- The application does not maintain conversation memory between questions.
- The default embedding model is lightweight and may not perform optimally on every technical document.
- Rebuild the index with `--reset` after editing or deleting source PDFs to avoid retaining obsolete chunks.

## Technology stack

Python, LangChain, ChromaDB, Hugging Face sentence-transformers, Google Gemini, Streamlit, and pytest.
