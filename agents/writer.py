import json

from agents.base_agent import BaseAgent


class WriterAgent(BaseAgent):
    def __init__(self, model_name: str):
        system_prompt = (
            "You are a research report writer. Write a clear markdown report using "
            "exactly these sections: 1 Introduction, 2 Key Papers, 3 Methods, "
            "4 Limitations, 5 Future Research Directions."
        )
        super().__init__(model_name, system_prompt)

    def run(self, analysis: dict, critique: dict) -> str:
        prompt = (
            "Write a structured markdown research report from this analysis and critique.\n\n"
            f"Analysis:\n{json.dumps(analysis, indent=2)}\n\n"
            f"Critique:\n{json.dumps(critique, indent=2)}"
        )
        return super().run(prompt)
