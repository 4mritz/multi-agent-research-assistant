import json

from pydantic import TypeAdapter

from agents.base_agent import BaseAgent
from schemas.research_schema import Paper, ResearchAnalysis


class AnalystAgent(BaseAgent):
    def __init__(self, model_name: str):
        system_prompt = (
            "You are a machine learning researcher. Return structured JSON only. "
            "For each paper, extract the method, communication mechanism if any, "
            "environment, training paradigm, and key contribution. Be specific and "
            'do not generalize. If information is missing, say "not specified".'
        )
        super().__init__(model_name, system_prompt, expect_json=True)

    def run(self, paper_data: dict) -> dict:
        papers = paper_data.get("papers")
        if papers is None and "results" in paper_data:
            papers = [paper for item in paper_data["results"] for paper in item.get("papers", [])]
        validated_input = {"papers": [paper.model_dump() for paper in TypeAdapter(list[Paper]).validate_python(papers or [])]}
        prompt = (
            "Analyze the input papers and return valid JSON only in this format:\n"
            '{"paper_summaries":[{"title":"...","url":"...","method":"...","dataset":"...",'
            '"key_contribution":"...","limitations":"..."}],'
            '"methodological_patterns":[],"research_trends":[],'
            '"comparative_analysis":{"architectural_differences":[],"training_strategies":[],'
            '"communication_mechanisms":[],"performance_tradeoffs":[]}}\n\n'
            "Use the fields like this:\n"
            '- method: include method, communication mechanism, and training paradigm\n'
            '- dataset: use it to store the environment name such as Hanabi or StarCraft, or "not specified"\n'
            '- key_contribution: main paper contribution\n'
            '- limitations: if not discussed, return "not specified"\n'
            'If any requested detail is missing, write "not specified". After paper summaries, '
            "perform explicit cross-paper comparison.\n"
            "- Compare papers directly and highlight differences, not just similarities.\n"
            "- Identify which approaches appear more scalable, efficient, or generalizable.\n"
            "- Mention environments such as Hanabi or SMAC if present.\n"
            "- Avoid generic statements and keep the analysis precise and technical.\n\n"
            f"Input:\n{json.dumps(validated_input, indent=2)}"
        )
        return ResearchAnalysis.model_validate(super().run(prompt)).model_dump()
