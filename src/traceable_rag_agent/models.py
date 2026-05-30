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


class EvidenceSufficiency(BaseModel):
    """Summary of whether retrieved evidence is enough to trust an answer."""

    status: str
    reason: str
    evidence_count: int = Field(ge=0)
    supported_claim_count: int = Field(ge=0)
    unsupported_claim_count: int = Field(ge=0)


class RagTrace(BaseModel):
    """A minimal trace shape for one RAG run."""

    question: str
    planned_queries: list[RetrievalQuery]
    evidence: list[Evidence]
    answer: str
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    evidence_sufficiency: EvidenceSufficiency | None = None
