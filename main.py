import argparse
from pathlib import Path

from orchestrator.pipeline import run_research_pipeline


def main():
    parser = argparse.ArgumentParser(description="Multi-agent research assistant")
    parser.add_argument("question", help="Research question")
    parser.add_argument("--planner-model", default="gpt-4.1-mini")
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    args = parser.parse_args()

    report = run_research_pipeline(
        research_question=args.question,
        planner_model=args.planner_model,
        llm_model=args.llm_model,
    )

    print(report)

    report_path = Path("reports/report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
