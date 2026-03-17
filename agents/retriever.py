import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter

from schemas.research_schema import Paper
from tools.arxiv_search import search_arxiv


class RetrieverAgent:
    def __init__(self):
        self.name = "RetrieverAgent"
        self.model_name = "tool:arxiv"
        self.provider = "tool"

    def run(self, input_text: str | list[str]) -> dict:
        queries = [input_text] if isinstance(input_text, str) else [q for q in input_text if isinstance(q, str) and q.strip()]
        results = []
        for query in queries:
            papers = search_arxiv(query, max_results=5)["papers"]
            validated = [paper.model_dump() for paper in TypeAdapter(list[Paper]).validate_python(papers)]
            results.append({"query": query, "papers": validated})
        output = {"results": results}
        self._append_agent_log(queries, output)
        return output

    def _append_agent_log(self, input_text: list[str], output: dict) -> None:
        path = Path("logs/agent_logs.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "agent": self.__class__.__name__,
            "input": input_text,
            "output": output,
            "model": self.model_name,
            "provider": self.provider,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except json.JSONDecodeError:
            data = []
        if not isinstance(data, list):
            data = []
        data.append(entry)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
