import logging

from agents.analyst import AnalystAgent
from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.writer import WriterAgent


logging.basicConfig(level=logging.INFO)


class ResearchPipeline:
    def __init__(self, planner_model: str, llm_model: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.planner = PlannerAgent(planner_model)
        self.retriever = RetrieverAgent()
        self.analyst = AnalystAgent(llm_model)
        self.critic = CriticAgent(llm_model)
        self.writer = WriterAgent(llm_model)

    def run(self, research_question: str) -> str:
        self.logger.info("Running planner")
        plan = self.planner.run(research_question)

        self.logger.info("Running retriever")
        retrieved = self.retriever.run(plan.get("search_queries", [research_question]))
        papers = [paper for item in retrieved["results"] for paper in item["papers"]]

        self.logger.info("Running analyst")
        analysis = self.analyst.run(papers)

        self.logger.info("Running critic")
        critique = self.critic.run(analysis)

        self.logger.info("Running writer")
        return self.writer.run(analysis, critique)


def run_research_pipeline(research_question: str, planner_model: str, llm_model: str) -> str:
    pipeline = ResearchPipeline(planner_model=planner_model, llm_model=llm_model)
    return pipeline.run(research_question)
