import json

from agents.base_agent import BaseAgent


class CriticAgent(BaseAgent):
    def __init__(self, model_name: str):
        system_prompt = (
            "You are a research critic. Review a research analysis and return only "
            "valid JSON with keys: weaknesses, missing_areas, research_gaps. "
            "Each value must be a list of concise strings."
        )
        super().__init__(model_name, system_prompt)

    def run(self, analysis: dict) -> dict:
        prompt = (
            "Critique this research analysis and identify weaknesses, missing areas, "
            f"and potential research gaps:\n{json.dumps(analysis, indent=2)}"
        )
        output = super().run(prompt)
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {
                "weaknesses": [],
                "missing_areas": [],
                "research_gaps": [output.strip()],
            }
