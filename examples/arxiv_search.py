import arxiv

client = arxiv.Client()
search = arxiv.Search(query="Large Language Models", max_results=2)

for index, paper in enumerate(client.results(search), start=1):
    authors = ", ".join(author.name for author in paper.authors)
    print(f"\nResult {index}")
    print(f"Title: {paper.title}")
    print(f"Authors: {authors}")
    print(f"Summary: {paper.summary[:500]}")
