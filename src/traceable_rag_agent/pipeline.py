import re
from dataclasses import dataclass

from traceable_rag_agent.models import ClaimCheck, Evidence, RagTrace, RetrievalQuery


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
    evidence = retrieve(question, chunks, top_k=top_k)
    answer = _synthesize_answer(evidence)
    claim_checks = check_answer_claims(answer, evidence)
    return RagTrace(
        question=question,
        planned_queries=[
            RetrievalQuery(query=question, reason="Use the original user question for first-pass retrieval.")
        ],
        evidence=evidence,
        answer=answer,
        claim_checks=claim_checks,
    )




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


def _synthesize_answer(evidence: list[Evidence]) -> str:
    if not evidence:
        return "I do not have enough retrieved evidence to answer this question."

    cited_sentences = [f"{item.text} [{item.source_id}]" for item in evidence]
    return " ".join(cited_sentences)


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))
