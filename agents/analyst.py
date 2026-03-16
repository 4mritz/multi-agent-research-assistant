import json

from agents.base_agent import BaseAgent


class AnalystAgent(BaseAgent):
    def __init__(self, model_name: str):
        system_prompt = (
            "You are a research analyst. Review paper titles and abstracts and return "
            "only valid JSON with keys: key_ideas, methods, results, limitations. "
            "Each value must be a list of concise strings."
        )
        super().__init__(model_name, system_prompt)

    def run(self, papers: list[dict]) -> dict:
        prompt = (
            "Analyze these papers and summarize them into the required sections:\n"
            f"{json.dumps(papers, indent=2)}"
        )
        output = super().run(prompt)
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {
                "key_ideas": [],
                "methods": [],
                "results": [],
                "limitations": [output.strip()],
            }
