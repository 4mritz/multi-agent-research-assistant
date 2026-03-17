from pydantic import BaseModel, Field


class Paper(BaseModel):
    title: str = Field(..., description="Paper title.")
    authors: list[str] = Field(..., description="List of author names.")
    year: int = Field(..., description="Publication year.")
    summary: str = Field(..., description="Paper abstract or summary.")
    categories: list[str] = Field(..., description="arXiv categories or tags.")
    url: str = Field(..., description="Paper URL.")


class PaperAnalysis(BaseModel):
    title: str = Field(..., description="Paper title.")
    url: str = Field(..., description="Paper URL.")
    method: str = Field(..., description="Method, model type, training strategy, and evaluation summary.")
    dataset: str = Field(..., description="Primary datasets used.")
    key_contribution: str = Field(..., description="Main innovation or contribution.")
    limitations: str = Field(..., description="Key limitations.")


class ComparativeAnalysis(BaseModel):
    architectural_differences: list[str] = Field(..., description="Explicit architectural differences across papers.")
    training_strategies: list[str] = Field(..., description="Contrasts in training strategies across papers.")
    communication_mechanisms: list[str] = Field(..., description="Differences in communication mechanisms across papers.")
    performance_tradeoffs: list[str] = Field(..., description="Tradeoffs in scalability, efficiency, and generalization.")


class ResearchAnalysis(BaseModel):
    paper_summaries: list[PaperAnalysis] = Field(..., description="Per-paper analysis summaries.")
    methodological_patterns: list[str] = Field(..., description="Shared methodological patterns.")
    research_trends: list[str] = Field(..., description="Observed research trends.")
    comparative_analysis: ComparativeAnalysis = Field(..., description="Cross-paper comparative analysis.")


class CritiqueReport(BaseModel):
    methodological_issues: list[str] = Field(..., description="Methodology concerns.")
    dataset_bias_risks: list[str] = Field(..., description="Bias and domain shift risks.")
    reproducibility_concerns: list[str] = Field(..., description="Reproducibility or resource concerns.")
    future_research_opportunities: list[str] = Field(..., description="Promising future work.")
