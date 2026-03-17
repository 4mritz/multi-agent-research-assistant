from typing import Any

import arxiv


def search_arxiv(query: str, max_results: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Return top arXiv papers for a query.

    Each paper contains:
    - title: paper title
    - authors: author names
    - year: publication year
    - summary: abstract text
    - categories: arXiv category tags
    - url: arXiv entry URL
    """
    try:
        search = arxiv.Search(
            query=query,
            max_results=min(max_results, 5),
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client()
        papers = []
        for result in client.results(search):
            papers.append(
                {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "year": result.published.year,
                    "summary": result.summary.replace("\n", " ").strip(),
                    "categories": list(result.categories),
                    "url": result.entry_id,
                }
            )
        return {"papers": papers}
    except Exception:
        return {"papers": []}
