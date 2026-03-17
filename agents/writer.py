import json

from agents.base_agent import BaseAgent


class WriterAgent(BaseAgent):
    def __init__(self, model_name: str):
        system_prompt = (
            "You are an academic literature review writer. Write formal markdown with "
            "clear headings, real paper metadata, and grounded claims only from the provided JSON."
        )
        super().__init__(model_name, system_prompt)

    def run(self, payload: dict) -> str:
        analysis = payload["analysis"]
        critique = payload["critique"]
        prompt = (
            "Write a markdown research report in an academic literature review tone.\n"
            "Use markdown headings for exactly these sections:\n"
            "# 1 Introduction\n"
            "# 2 Key Papers\n"
            "# 3 Methodological Trends\n"
            "# 4 Limitations of Current Research\n"
            "# 5 Future Research Directions\n"
            "# 6 References\n\n"
            "Requirements:\n"
            "- Base every claim only on the provided analysis and critique JSON.\n"
            "- Do not hallucinate citations, paper titles, or links.\n"
            "- Use concrete paper metadata from paper_summaries when describing key papers.\n"
            "- Avoid vague phrases like 'several studies' unless you immediately name them.\n"
            "- In References, include only paper titles and links that are explicitly present "
            "in paper_summaries.\n"
            "- If a paper link is missing, do not invent one.\n\n"
            f"Analysis JSON:\n{json.dumps(analysis, indent=2)}\n\n"
            f"Critique JSON:\n{json.dumps(critique, indent=2)}"
        )
        return super().run(prompt)
