from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """A planned retrieval query derived from the user's question."""

    query: str
    reason: str


class Evidence(BaseModel):
    """A retrieved evidence snippet."""

    source_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, str] = Field(default_factory=dict)


class ClaimCheck(BaseModel):
    """Support status for one answer claim."""

    claim: str
    status: str
    supporting_source_ids: list[str] = Field(default_factory=list)


class RagTrace(BaseModel):
    """A minimal trace shape for one RAG run."""

    question: str
    planned_queries: list[RetrievalQuery]
    evidence: list[Evidence]
    answer: str
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
