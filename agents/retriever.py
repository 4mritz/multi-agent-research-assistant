from tools.arxiv_search import search_arxiv


class RetrieverAgent:
    def __init__(self):
        self.name = "RetrieverAgent"

    def run(self, input_text) -> dict:
        queries = [input_text] if isinstance(input_text, str) else list(input_text)

        results = []
        for query in queries:
            papers = search_arxiv(query, max_results=5)

            results.append(
                {
                    "query": query,
                    "papers": papers,
                }
            )

        return {"results": results}