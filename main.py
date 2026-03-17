import argparse
from pathlib import Path

from orchestrator.pipeline import run_research_pipeline


def main():
    parser = argparse.ArgumentParser(description="Multi-agent research assistant")
    parser.add_argument("question", help="Research question")
    parser.add_argument("--planner-model", default="ollama:llama3.1:8b-instruct-q4_K_M")
    parser.add_argument("--llm-model", default="ollama:llama3.1:8b-instruct-q4_K_M")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    report = run_research_pipeline(
        research_question=args.question,
        planner_model=args.planner_model,
        llm_model=args.llm_model,
        debug=args.debug,
    )

    print(report)

    report_path = Path("reports/report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
