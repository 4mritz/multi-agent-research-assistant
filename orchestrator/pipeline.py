import json
import logging
from pathlib import Path

from agents.analyst import AnalystAgent
from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.writer import WriterAgent
from schemas.research_schema import CritiqueReport, ResearchAnalysis


logging.basicConfig(level=logging.INFO)


class ResearchPipeline:
    def __init__(self, planner_model: str, llm_model: str, debug: bool = False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        Path("logs").mkdir(parents=True, exist_ok=True)
        if not Path("logs/agent_logs.json").exists():
            Path("logs/agent_logs.json").write_text("[]", encoding="utf-8")
        self.planner = PlannerAgent(planner_model)
        self.retriever = RetrieverAgent()
        self.analyst = AnalystAgent(llm_model)
        self.critic = CriticAgent(llm_model)
        self.writer = WriterAgent(llm_model)

    def run(self, research_question: str) -> str:
        planner_output = self._run_planner(research_question)
        queries = self._extract_queries(planner_output, research_question)
        retrieved = self._run_retriever(queries)
        analysis = self._run_analyst(retrieved)
        critique = self._run_critic(analysis)
        final_report = self._run_writer(analysis, critique, research_question)
        self._write_trace(research_question, planner_output, retrieved, analysis, critique, final_report)
        return final_report

    def _run_planner(self, question: str) -> dict:
        try:
            output = self.planner.run(question)
            if not isinstance(output, dict):
                raise ValueError("Planner output must be a dict.")
            return output
        except Exception as exc:
            self.logger.error("Planner failed: %s", exc)
            return {"topic": question, "search_queries": [question], "analysis_focus": ["architectures", "datasets", "evaluation methods"]}

    def _extract_queries(self, planner_output: dict, question: str) -> list[str]:
        queries = planner_output.get("search_queries", [])
        if isinstance(queries, str):
            queries = [queries]
        queries = [query.strip() for query in queries if isinstance(query, str) and query.strip()]
        return queries or [question]

    def _run_retriever(self, queries: list[str]) -> dict:
        try:
            output = self.retriever.run(queries)
            return output if isinstance(output, dict) else {"results": []}
        except Exception as exc:
            self.logger.error("Retriever failed: %s", exc)
            return {"results": []}

    def _run_analyst(self, retrieved: dict) -> dict:
        try:
            return ResearchAnalysis.model_validate(self.analyst.run(retrieved)).model_dump()
        except Exception as exc:
            self.logger.error("Analyst failed: %s", exc)
            papers = [paper for item in retrieved.get("results", []) for paper in item.get("papers", [])]
            fallback_summaries = [
                {
                    "title": paper["title"],
                    "url": paper["url"],
                    "method": "Summary unavailable due to analysis failure.",
                    "dataset": "Not extracted.",
                    "key_contribution": paper["summary"][:200],
                    "limitations": "Detailed limitations unavailable.",
                }
                for paper in papers[:5]
            ]
            return ResearchAnalysis(
                paper_summaries=fallback_summaries,
                methodological_patterns=[],
                research_trends=[],
                comparative_analysis={
                    "architectural_differences": [],
                    "training_strategies": [],
                    "communication_mechanisms": [],
                    "performance_tradeoffs": [],
                },
            ).model_dump()

    def _run_critic(self, analysis: dict) -> dict:
        try:
            return CritiqueReport.model_validate(self.critic.run(analysis)).model_dump()
        except Exception as exc:
            self.logger.error("Critic failed: %s", exc)
            return CritiqueReport(
                methodological_issues=[],
                dataset_bias_risks=[],
                reproducibility_concerns=[],
                future_research_opportunities=[],
            ).model_dump()

    def _run_writer(self, analysis: dict, critique: dict, question: str) -> str:
        try:
            return self.writer.run({"analysis": analysis, "critique": critique})
        except Exception as exc:
            self.logger.error("Writer failed: %s", exc)
            refs = "\n".join(
                f"- [{paper['title']}]({paper['url']})" for paper in analysis.get("paper_summaries", []) if paper.get("url")
            ) or "- No references available."
            return (
                f"# 1 Introduction\n\nThis fallback report summarizes research on {question}.\n\n"
                f"# 2 Key Papers\n\nAutomatic report generation failed.\n\n"
                f"# 3 Methodological Trends\n\n{'; '.join(analysis.get('methodological_patterns', [])) or 'Not available.'}\n\n"
                f"# 4 Limitations of Current Research\n\n{'; '.join(critique.get('methodological_issues', [])) or 'Not available.'}\n\n"
                f"# 5 Future Research Directions\n\n{'; '.join(critique.get('future_research_opportunities', [])) or 'Not available.'}\n\n"
                f"# 6 References\n\n{refs}\n"
            )

    def _write_trace(self, question: str, planner_output: dict, retrieved: dict, analysis: dict, critique: dict, final_report: str) -> None:
        trace = {
            "research_question": question,
            "planner_output": planner_output,
            "retrieved_papers": retrieved,
            "analysis": analysis,
            "critique": critique,
            "final_report": final_report,
        }
        path = Path("logs/pipeline_trace.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace, indent=2), encoding="utf-8")


def run_research_pipeline(research_question: str, planner_model: str, llm_model: str, debug: bool = False) -> str:
    return ResearchPipeline(planner_model=planner_model, llm_model=llm_model, debug=debug).run(research_question)
