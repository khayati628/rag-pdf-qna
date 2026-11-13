import os

import streamlit as st

from rag_qa import config
from rag_qa.pipeline import RAGPipeline

st.set_page_config(page_title="PDF Q&A", page_icon="📘")
st.title("📘 Ask your PDFs")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Copy .env.example to .env and add your key.")
    st.stop()

@st.cache_resource(show_spinner="Building the index from your PDFs (first run only)...")
def ensure_index():
    from rag_qa.ingest import build_index

    if not config.CHROMA_DIR.exists() or not any(config.CHROMA_DIR.iterdir()):
        build_index()
    return config.CHROMA_DIR.exists()


if not ensure_index():
    st.error(f"No PDFs found in `{config.DATA_DIR}`, so no index could be built.")
    st.stop()


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox(
        "Retrieval strategy",
        config.STRATEGIES,
        index=config.STRATEGIES.index(config.DEFAULT_STRATEGY),
    )
    k = st.slider("Chunks to retrieve", 2, 10, config.DEFAULT_K)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(source)

question = st.chat_input("Ask a question about your PDFs...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                answer = pipeline.ask(question, strategy=strategy, k=k)
            except Exception as error:
                st.error(f"Something went wrong: {error}")
                st.stop()

        st.markdown(answer.text)
        if answer.sources:
            with st.expander("Sources"):
                for source in answer.sources:
                    st.write(source)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "sources": answer.sources}
    )
