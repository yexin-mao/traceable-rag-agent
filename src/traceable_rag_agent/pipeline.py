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

@dataclass(frozen=True)
class BenchmarkQuestion:
    """A benchmark item with expected sources for retrieval evaluation."""

    question: str
    expected_source_ids: list[str]


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


def measure_retrieval_coverage(
    trace: RagTrace, expected_source_ids: list[str]
) -> dict[str, object]:
    """Compare retrieved evidence sources against expected sources for evaluation."""

    retrieved_source_ids = list(dict.fromkeys(item.source_id for item in trace.evidence))
    found_source_ids = [source_id for source_id in expected_source_ids if source_id in retrieved_source_ids]
    missing_source_ids = [
        source_id for source_id in expected_source_ids if source_id not in retrieved_source_ids
    ]
    coverage_ratio = len(found_source_ids) / len(expected_source_ids) if expected_source_ids else 1.0
    return {
        "expected_source_ids": expected_source_ids,
        "retrieved_source_ids": retrieved_source_ids,
        "found_source_ids": found_source_ids,
        "missing_source_ids": missing_source_ids,
        "coverage_ratio": coverage_ratio,
    }


def measure_citation_support(trace: RagTrace) -> dict[str, object]:
    """Measure whether answer citations point to retrieved evidence sources."""

    cited_source_ids = list(dict.fromkeys(re.findall(r"\[([^\]]+)\]", trace.answer)))
    evidence_source_ids = list(dict.fromkeys(item.source_id for item in trace.evidence))
    supported_cited_source_ids = [source_id for source_id in cited_source_ids if source_id in evidence_source_ids]
    unsupported_cited_source_ids = [source_id for source_id in cited_source_ids if source_id not in evidence_source_ids]
    citation_support_ratio = len(supported_cited_source_ids) / len(cited_source_ids) if cited_source_ids else 1.0
    return {
        "cited_source_ids": cited_source_ids,
        "evidence_source_ids": evidence_source_ids,
        "supported_cited_source_ids": supported_cited_source_ids,
        "unsupported_cited_source_ids": unsupported_cited_source_ids,
        "citation_support_ratio": citation_support_ratio,
    }


def run_retrieval_coverage_benchmark(
    benchmark: list[BenchmarkQuestion], documents: list[Document], top_k: int = 3
) -> dict[str, object]:
    """Run benchmark questions and summarize retrieval coverage quality."""

    results: list[dict[str, object]] = []
    for item in benchmark:
        trace = answer_question(item.question, documents, top_k=top_k)
        coverage = measure_retrieval_coverage(trace, item.expected_source_ids)
        results.append({"question": item.question, **coverage})

    average_coverage_ratio = (
        sum(float(result["coverage_ratio"]) for result in results) / len(results) if results else 1.0
    )
    return {
        "question_count": len(benchmark),
        "average_coverage_ratio": average_coverage_ratio,
        "results": results,
    }


def run_rag_quality_benchmark(
    benchmark: list[BenchmarkQuestion], documents: list[Document], top_k: int = 3
) -> dict[str, object]:
    """Run benchmark questions and report retrieval plus citation quality metrics."""

    results: list[dict[str, object]] = []
    for item in benchmark:
        trace = answer_question(item.question, documents, top_k=top_k)
        results.append(
            {
                "question": item.question,
                "retrieval_coverage": measure_retrieval_coverage(trace, item.expected_source_ids),
                "citation_support": measure_citation_support(trace),
            }
        )

    average_retrieval_coverage_ratio = (
        sum(float(result["retrieval_coverage"]["coverage_ratio"]) for result in results) / len(results)
        if results
        else 1.0
    )
    average_citation_support_ratio = (
        sum(float(result["citation_support"]["citation_support_ratio"]) for result in results)
        / len(results)
        if results
        else 1.0
    )
    return {
        "question_count": len(benchmark),
        "average_retrieval_coverage_ratio": average_retrieval_coverage_ratio,
        "average_citation_support_ratio": average_citation_support_ratio,
        "results": results,
    }


def diagnose_rag_quality_failures(quality_report: dict[str, object]) -> list[dict[str, str]]:
    """Label question-level RAG failure modes from a quality benchmark report."""

    diagnoses: list[dict[str, str]] = []
    for result in quality_report.get("results", []):
        retrieval_coverage = result["retrieval_coverage"]
        citation_support = result["citation_support"]
        missing_source_ids = retrieval_coverage["missing_source_ids"]
        unsupported_cited_source_ids = citation_support["unsupported_cited_source_ids"]

        if missing_source_ids:
            failure_mode = "retrieval_gap"
            reason = f"Missing expected sources: {', '.join(missing_source_ids)}."
        elif unsupported_cited_source_ids:
            failure_mode = "unsupported_citation"
            reason = f"Answer cited sources that were not retrieved: {', '.join(unsupported_cited_source_ids)}."
        else:
            failure_mode = "pass"
            reason = "Retrieval covered expected sources and all citations are supported."

        diagnoses.append(
            {
                "question": result["question"],
                "failure_mode": failure_mode,
                "reason": reason,
            }
        )
    return diagnoses


def summarize_failure_diagnoses(diagnoses: list[dict[str, str]]) -> dict[str, object]:
    """Count failure diagnosis modes for dashboard and progress reporting."""

    failure_mode_counts: dict[str, int] = {}
    for diagnosis in diagnoses:
        failure_mode = diagnosis["failure_mode"]
        failure_mode_counts[failure_mode] = failure_mode_counts.get(failure_mode, 0) + 1

    passed_questions = failure_mode_counts.get("pass", 0)
    failed_questions = len(diagnoses) - passed_questions
    non_pass_counts = {
        mode: count for mode, count in failure_mode_counts.items() if mode != "pass"
    }
    top_failure_mode = max(non_pass_counts, key=non_pass_counts.get) if non_pass_counts else "pass"

    return {
        "total_questions": len(diagnoses),
        "passed_questions": passed_questions,
        "failed_questions": failed_questions,
        "failure_mode_counts": failure_mode_counts,
        "top_failure_mode": top_failure_mode,
    }


def recommend_next_evaluation_action(summary: dict[str, object]) -> dict[str, str]:
    """Translate a failure summary into one next debugging priority."""

    top_failure_mode = str(summary["top_failure_mode"])
    total_questions = int(summary["total_questions"])
    failure_mode_counts = summary["failure_mode_counts"]
    top_failure_count = int(failure_mode_counts[top_failure_mode])

    if top_failure_mode == "retrieval_gap":
        return {
            "priority": "fix_retrieval",
            "message": (
                f"Top failure mode is retrieval_gap across {top_failure_count} of "
                f"{total_questions} questions; improve query rewriting, retrieval recall, "
                "or reranking before changing answer synthesis."
            ),
        }
    if top_failure_mode == "unsupported_citation":
        return {
            "priority": "fix_citations",
            "message": (
                f"Top failure mode is unsupported_citation across {top_failure_count} of "
                f"{total_questions} questions; tighten citation filtering and answer "
                "synthesis so every cited source is retrieved evidence."
            ),
        }
    return {
        "priority": "keep_benchmarking",
        "message": "All benchmark questions passed; add harder questions before changing the pipeline.",
    }


def build_quality_report_markdown(summary: dict[str, object], action: dict[str, str]) -> str:
    """Format benchmark status and next action as a dashboard-ready Markdown report."""

    failure_mode_counts = summary["failure_mode_counts"]
    lines = [
        "## RAG Quality Report",
        "",
        f"- Total questions: {summary['total_questions']}",
        f"- Passed questions: {summary['passed_questions']}",
        f"- Failed questions: {summary['failed_questions']}",
        f"- Top failure mode: {summary['top_failure_mode']}",
        f"- Next priority: {action['priority']}",
        f"- Recommendation: {action['message']}",
        "",
        "| Failure mode | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {mode} | {count} |" for mode, count in failure_mode_counts.items())
    return "\n".join(lines)


def build_trace_report_markdown(trace: RagTrace) -> str:
    """Format one RAG run trace as Markdown for observability dashboard rendering."""

    sufficiency = trace.evidence_sufficiency
    sufficiency_text = (
        f"{sufficiency.status} — {sufficiency.reason}"
        if sufficiency is not None
        else "not evaluated — No evidence sufficiency summary is attached."
    )
    lines = [
        "## Trace Report",
        "",
        f"- Question: {trace.question}",
        f"- Evidence sufficiency: {sufficiency_text}",
        f"- Planned queries: {len(trace.planned_queries)}",
        f"- Retrieved evidence items: {len(trace.evidence)}",
        "",
        "### Retrieval steps",
        "| Step | Query | Sources | Chunks |",
        "| ---: | --- | --- | --- |",
    ]
    for step_index, step in enumerate(trace.retrieval_steps, start=1):
        sources = ", ".join(step.retrieved_source_ids) if step.retrieved_source_ids else "none"
        chunks = ", ".join(step.retrieved_chunk_ids) if step.retrieved_chunk_ids else "none"
        lines.append(f"| {step_index} | {step.query} | {sources} | {chunks} |")

    lines.extend(
        [
            "",
            "### Evidence",
            "| Rank | Source | Chunk | Score | Snippet |",
            "| ---: | --- | --- | ---: | --- |",
        ]
    )
    for rank, item in enumerate(trace.evidence, start=1):
        chunk_id = item.metadata.get("chunk_id", "")
        lines.append(
            f"| {rank} | {item.source_id} | {chunk_id} | {item.score:.2f} | {item.text} |"
        )
    return "\n".join(lines)


def build_trace_status_summary(trace: RagTrace) -> dict[str, object]:
    """Return compact per-run metrics for dashboard status cards."""

    cited_source_ids = list(dict.fromkeys(re.findall(r"\[([^\]]+)\]", trace.answer)))
    sufficiency_status = (
        trace.evidence_sufficiency.status if trace.evidence_sufficiency is not None else "not_evaluated"
    )
    return {
        "sufficiency_status": sufficiency_status,
        "planned_query_count": len(trace.planned_queries),
        "retrieval_step_count": len(trace.retrieval_steps),
        "evidence_count": len(trace.evidence),
        "supported_claim_count": sum(check.status == "supported" for check in trace.claim_checks),
        "unsupported_claim_count": sum(check.status == "unsupported" for check in trace.claim_checks),
        "cited_source_count": len(cited_source_ids),
    }


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
        claim_terms = _terms(_strip_citations(claim))
        if not claim_terms:
            continue
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


def _strip_citations(text: str) -> str:
    return re.sub(r"\s*\[[^\]]+\]", "", text)


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))
