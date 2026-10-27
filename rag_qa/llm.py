from langchain_google_genai import ChatGoogleGenerativeAI

from rag_qa import config


def get_llm():
    if not config.CHAT_MODELS:
        raise ValueError("GEMINI_MODELS must list at least one model name.")
    models = [ChatGoogleGenerativeAI(model=name) for name in config.CHAT_MODELS]
    primary, *fallbacks = models
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
