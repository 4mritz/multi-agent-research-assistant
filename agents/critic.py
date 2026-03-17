import json

from agents.base_agent import BaseAgent
from schemas.research_schema import CritiqueReport, ResearchAnalysis


class CriticAgent(BaseAgent):
    def __init__(self, model_name: str):
        system_prompt = (
            "You are a peer reviewer for a machine learning conference. Review structured "
            "analysis JSON and return only valid JSON with deterministic, concise findings."
        )
        super().__init__(model_name, system_prompt, expect_json=True)

    def run(self, analysis: dict) -> dict:
        validated_analysis = ResearchAnalysis.model_validate(analysis).model_dump()
        prompt = (
            "Evaluate this structured analysis as a peer reviewer and return valid JSON only "
            'in this format: {"methodological_issues":[],"dataset_bias_risks":[],'
            '"reproducibility_concerns":[],"future_research_opportunities":[]}\n\n'
            "Focus on dataset bias, domain shift issues, model generalization risks, "
            "hardware constraints, and experimental limitations.\n\n"
            f"Input:\n{json.dumps(validated_analysis, indent=2)}"
        )
        return CritiqueReport.model_validate(super().run(prompt)).model_dump()
