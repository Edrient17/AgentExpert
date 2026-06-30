from typing import List, Optional

from pydantic import BaseModel, Field

from src.schema.constants import TeamName


class QuestionProcessingResult(BaseModel):
    q_validity: bool = Field(description="Whether the user question is valid and answerable.")
    q_en_transformed: str = Field(description="A clear English rewrite of the user question.")
    rag_queries: List[str] = Field(
        description="2-4 English RAG search query candidates.",
        min_length=2,
        max_length=4,
    )
    output_format: List[str] = Field(
        description="Requested output format as [type, language].",
        min_length=2,
        max_length=2,
    )


class QuestionEvaluationResult(BaseModel):
    semantic_alignment: float = Field(ge=0.0, le=1.0)
    format_compliance: bool
    rag_query_scores: List[float] = Field(default_factory=list)
    error_message: str = ""


class DocEvaluationResult(BaseModel):
    semantic_relevance: float = Field(ge=0.0, le=1.0)
    is_detailed: float = Field(ge=0.0, le=1.0)
    error_message: str = ""


class AnswerEvaluationResult(BaseModel):
    rules_compliance: float = Field(ge=0.0, le=1.0)
    question_coverage: float = Field(ge=0.0, le=1.0)
    hallucination_score: float = Field(ge=0.0, le=1.0)
    error_message: str = ""


class ManagerDecision(BaseModel):
    next_team: TeamName = Field(
        description="The next team to call, or end if workflow should stop."
    )
    feedback: Optional[str] = Field(
        description="Concrete feedback for the team when revision is needed."
    )
    reason: str = Field(description="Short explanation of the decision.")
