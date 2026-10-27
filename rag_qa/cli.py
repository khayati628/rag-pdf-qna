import argparse
import os
import sys

from rag_qa import config
from rag_qa.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your PDFs from the terminal.")
    parser.add_argument("--strategy", choices=config.STRATEGIES, default=config.DEFAULT_STRATEGY)
    parser.add_argument("-k", type=int, default=config.DEFAULT_K, help="chunks to retrieve")
    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key.")
    if not config.CHROMA_DIR.exists():
        sys.exit("No index found. Run: python -m rag_qa.ingest")

    pipeline = RAGPipeline()
    print(f"RAG chatbot ready (strategy: {args.strategy}, k: {args.k}). Type 'exit' to quit.")

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            answer = pipeline.ask(question, strategy=args.strategy, k=args.k)
        except Exception as error:
            print(f"Error: {error}")
            continue

        print(f"\nAnswer: {answer.text}")
        if answer.sources:
            print("\nSources:")
            for source in answer.sources:
                print(f"  - {source}")


if __name__ == "__main__":
    main()
