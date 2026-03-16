import json

from agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    def __init__(self, model_name: str):
        system_prompt = (
            "You are a research planning agent. Given a user research question, "
            "return only valid JSON with keys: topic, search_queries, analysis_tasks. "
            "search_queries and analysis_tasks must be lists of concise strings."
        )
        super().__init__(model_name, system_prompt)

    def run(self, input_text: str) -> dict:
        prompt = (
            "Create a research plan for this question:\n"
            f"{input_text}\n\n"
            'Output format: {"topic":"...","search_queries":["..."],"analysis_tasks":["..."]}'
        )
        output = super().run(prompt)
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {
                "topic": input_text,
                "search_queries": [input_text],
                "analysis_tasks": [output.strip()],
            }
