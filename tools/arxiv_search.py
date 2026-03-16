import arxiv


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    papers = []
    for result in client.results(search):
        papers.append(
            {
                "title": result.title,
                "abstract": result.summary.replace("\n", " ").strip(),
                "link": result.entry_id,
            }
        )
    return papers
