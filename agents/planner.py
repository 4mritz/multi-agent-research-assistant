from agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    def __init__(self, model_name: str):
        system_prompt = (
            "You are a research planning agent. Convert a research question into "
            "structured search queries and return only valid JSON."
        )
        super().__init__(model_name, system_prompt, expect_json=True)

    def run(self, input_text: str) -> dict:
        prompt = (
            "Convert this research question into a structured search plan.\n"
            f"Question: {input_text}\n\n"
            "Return valid JSON only in this format:\n"
            '{'
            '"topic":"...",'
            '"search_queries":["...","...","..."],'
            '"analysis_focus":["architectures","datasets","evaluation methods"]'
            "}"
        )
        return super().run(prompt)
