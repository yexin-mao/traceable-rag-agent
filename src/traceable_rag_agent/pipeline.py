import re
from dataclasses import dataclass
from pathlib import Path

from traceable_rag_agent.models import (
    ClaimCheck,
    Evidence,
    EvidenceSufficiency,
    RagTrace,
    RetrievalQuery,
    RetrievalStep,
)


@dataclass(frozen=True)
class Document:
    """A raw source document before chunking."""

    source_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A source-preserving document chunk used for retrieval."""

    chunk_id: str
    source_id: str
    text: str


def chunk_documents(documents: list[Document], max_words: int = 120) -> list[Chunk]:
    """Split documents into deterministic word chunks while preserving source metadata."""

    if max_words <= 0:
        raise ValueError("max_words must be positive")

    chunks: list[Chunk] = []
    for document in documents:
        words = document.text.split()
        for chunk_index, start in enumerate(range(0, len(words), max_words)):
            chunk_words = words[start : start + max_words]
            if not chunk_words:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source_id}#{chunk_index}",
                    source_id=document.source_id,
                    text=" ".join(chunk_words),
                )
            )
    return chunks


def plan_retrieval_queries(question: str) -> list[RetrievalQuery]:
    """Create focused retrieval queries from a complex user question."""

    cleaned_question = question.strip()
    if " and " not in cleaned_question.lower():
        return [
            RetrievalQuery(
                query=cleaned_question,
                reason="Use the original user question for first-pass retrieval.",
            )
        ]

    prefix_match = re.match(r"^(?P<prefix>.+?\bRAG\b)\s+(?P<first>.+?)\s+and\s+(?P<second>.+?)\?$", cleaned_question, re.IGNORECASE)
    if not prefix_match:
        return [
            RetrievalQuery(
                query=cleaned_question,
                reason="Use the original user question for first-pass retrieval.",
            )
        ]

    prefix = prefix_match.group("prefix")
    return [
        RetrievalQuery(
            query=f"{prefix} {prefix_match.group(part)}?",
            reason="Retrieve evidence for one focused part of the complex question.",
        )
        for part in ("first", "second")
    ]


def retrieve(query: str, chunks: list[Chunk], top_k: int = 3) -> list[Evidence]:
    """Rank chunks by simple lexical overlap and return evidence objects."""

    if top_k <= 0:
        return []

    query_terms = _terms(query)
    scored: list[tuple[float, int, Chunk]] = []
    for index, chunk in enumerate(chunks):
        chunk_terms = _terms(chunk.text)
        if not query_terms or not chunk_terms:
            score = 0.0
        else:
            score = len(query_terms & chunk_terms) / len(query_terms)
        scored.append((score, index, chunk))

    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]
    return [
        Evidence(
            source_id=chunk.source_id,
            text=chunk.text,
            score=score,
            metadata={"chunk_id": chunk.chunk_id},
        )
        for score, _, chunk in ranked
        if score > 0
    ]


def answer_question(question: str, documents: list[Document], top_k: int = 3) -> RagTrace:
    """Run a minimal local RAG pipeline and return a traceable answer skeleton."""

    chunks = chunk_documents(documents)
    planned_queries = plan_retrieval_queries(question)
    retrieval_steps: list[RetrievalStep] = []
    retrieved_items: list[Evidence] = []
    for planned_query in planned_queries:
        query_evidence = retrieve(planned_query.query, chunks, top_k=top_k)
        retrieved_items.extend(query_evidence)
        retrieval_steps.append(
            RetrievalStep(
                query=planned_query.query,
                retrieved_source_ids=[item.source_id for item in query_evidence],
                retrieved_chunk_ids=[item.metadata["chunk_id"] for item in query_evidence],
            )
        )
    evidence = _deduplicate_evidence(retrieved_items)
    answer = _synthesize_answer(evidence)
    claim_checks = check_answer_claims(answer, evidence)
    evidence_sufficiency = check_evidence_sufficiency(evidence, claim_checks)
    return RagTrace(
        question=question,
        planned_queries=planned_queries,
        retrieval_steps=retrieval_steps,
        evidence=evidence,
        answer=answer,
        claim_checks=claim_checks,
        evidence_sufficiency=evidence_sufficiency,
    )




def build_evidence_table(trace: RagTrace) -> list[dict[str, object]]:
    """Return ranked evidence snippets for dashboard/report rendering."""

    return [
        {
            "rank": rank,
            "source_id": item.source_id,
            "chunk_id": item.metadata.get("chunk_id", ""),
            "score": item.score,
            "snippet": item.text,
        }
        for rank, item in enumerate(trace.evidence, start=1)
    ]


def save_trace_json(trace: RagTrace, output_path: str | Path) -> Path:
    """Persist one RAG run trace as readable JSON for later observability."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return path


def check_answer_claims(answer: str, evidence: list[Evidence]) -> list[ClaimCheck]:
    """Flag answer sentences that are not supported by retrieved evidence."""

    checks: list[ClaimCheck] = []
    for claim in _sentences(answer):
        claim_terms = _terms(claim)
        supporting_sources = [
            item.source_id
            for item in evidence
            if claim_terms and claim_terms.issubset(_terms(item.text))
        ]
        checks.append(
            ClaimCheck(
                claim=claim,
                status="supported" if supporting_sources else "unsupported",
                supporting_source_ids=supporting_sources,
            )
        )
    return checks


def check_evidence_sufficiency(
    evidence: list[Evidence], claim_checks: list[ClaimCheck]
) -> EvidenceSufficiency:
    """Summarize whether retrieved evidence is enough for the current answer."""

    supported_count = sum(check.status == "supported" for check in claim_checks)
    unsupported_count = sum(check.status == "unsupported" for check in claim_checks)
    if not evidence:
        status = "insufficient"
        reason = "No evidence was retrieved for this question."
    elif unsupported_count:
        status = "insufficient"
        reason = "Some answer claims are not supported by retrieved evidence."
    else:
        status = "sufficient"
        reason = "All answer claims are supported by retrieved evidence."

    return EvidenceSufficiency(
        status=status,
        reason=reason,
        evidence_count=len(evidence),
        supported_claim_count=supported_count,
        unsupported_claim_count=unsupported_count,
    )


def _deduplicate_evidence(evidence_items) -> list[Evidence]:
    unique_items: list[Evidence] = []
    seen_chunk_ids: set[str] = set()
    for item in evidence_items:
        dedupe_key = item.metadata.get("chunk_id", item.source_id)
        if dedupe_key in seen_chunk_ids:
            continue
        seen_chunk_ids.add(dedupe_key)
        unique_items.append(item)
    return unique_items


def _synthesize_answer(evidence: list[Evidence]) -> str:
    if not evidence:
        return "I do not have enough retrieved evidence to answer this question."

    cited_sentences = [f"{item.text} [{item.source_id}]" for item in evidence]
    return " ".join(cited_sentences)


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))
