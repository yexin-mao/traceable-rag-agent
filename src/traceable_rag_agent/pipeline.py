import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

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
    query_index = 0
    while query_index < len(planned_queries):
        planned_query = planned_queries[query_index]
        retrieval_started_at = perf_counter()
        query_evidence = retrieve(planned_query.query, chunks, top_k=top_k)
        latency_ms = (perf_counter() - retrieval_started_at) * 1000
        retrieved_items.extend(query_evidence)
        retrieval_steps.append(
            RetrievalStep(
                query=planned_query.query,
                retrieved_source_ids=[item.source_id for item in query_evidence],
                retrieved_chunk_ids=[item.metadata["chunk_id"] for item in query_evidence],
                latency_ms=latency_ms,
            )
        )
        if not query_evidence:
            retry_query = _rewrite_weak_retrieval_query(planned_query.query)
            if retry_query is not None and retry_query not in {query.query for query in planned_queries}:
                planned_queries.append(
                    RetrievalQuery(
                        query=retry_query,
                        reason="Retry retrieval with a rewritten query after weak evidence.",
                    )
                )
        query_index += 1
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




def build_evidence_decision_markdown(trace_items: list[dict[str, object]]) -> str:
    """Format answer-readiness decisions from evidence sufficiency status."""

    lines = [
        "## Evidence Decision Log",
        "",
        "| Trace | Sufficiency | Decision | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for item in trace_items:
        trace = item["trace"]
        sufficiency = trace.evidence_sufficiency
        status = sufficiency.status if sufficiency is not None else "not_evaluated"
        reason = sufficiency.reason if sufficiency is not None else "No evidence sufficiency summary is attached."
        decision = "ready_for_answer" if status == "sufficient" else "retry_or_escalate"
        lines.append(
            f"| {_escape_markdown_table_cell(str(item['label']))} | "
            f"{_escape_markdown_table_cell(status)} | {decision} | "
            f"{_escape_markdown_table_cell(reason)} |"
        )
    return "\n".join(lines)


def build_recovery_action_plan_markdown(trace_items: list[dict[str, object]]) -> str:
    """Format concrete recovery actions for traces that are not ready to answer."""

    lines = [
        "## Recovery Action Plan",
        "",
        "| Trace | Failure signal | Recommended action | Why |",
        "| --- | --- | --- | --- |",
    ]
    for item in trace_items:
        trace = item["trace"]
        failure_signal, action, reason = _recommend_recovery_action(trace)
        lines.append(
            f"| {_escape_markdown_table_cell(str(item['label']))} | {failure_signal} | "
            f"{action} | {_escape_markdown_table_cell(reason)} |"
        )
    return "\n".join(lines)


def build_answer_approval_gate_markdown(trace_items: list[dict[str, object]]) -> str:
    """Format an evidence-based gate for approving or blocking answer delivery."""

    lines = [
        "## Answer Approval Gate",
        "",
        "| Trace | Gate decision | Evidence status | Reviewer note |",
        "| --- | --- | --- | --- |",
    ]
    for item in trace_items:
        trace = item["trace"]
        sufficiency = trace.evidence_sufficiency
        evidence_status = sufficiency.status if sufficiency is not None else "not_evaluated"
        if evidence_status == "sufficient":
            decision = "approve_for_delivery"
            reviewer_note = "Evidence is sufficient; answer can be delivered with citations."
        else:
            decision = "block_for_revision"
            reviewer_note = sufficiency.reason if sufficiency is not None else "Run evidence evaluation before delivery."
        lines.append(
            f"| {_escape_markdown_table_cell(str(item['label']))} | {decision} | "
            f"{evidence_status} | {_escape_markdown_table_cell(reviewer_note)} |"
        )
    return "\n".join(lines)


def build_human_review_queue_markdown(trace_items: list[dict[str, object]]) -> str:
    """Format blocked answer traces as a human-review queue."""

    lines = [
        "## Human Review Queue",
        "",
        "| Trace | Review trigger | Recommended reviewer action | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for item in trace_items:
        trace = item["trace"]
        sufficiency = trace.evidence_sufficiency
        if sufficiency is not None and sufficiency.status == "sufficient":
            continue
        review_trigger = "not_evaluated" if sufficiency is None else "insufficient_evidence"
        reviewer_action = (
            "run_evidence_evaluation" if sufficiency is None else "inspect_retrieval_and_revise_answer"
        )
        reason = sufficiency.reason if sufficiency is not None else "Run evidence evaluation before delivery."
        lines.append(
            f"| {_escape_markdown_table_cell(str(item['label']))} | {review_trigger} | "
            f"{reviewer_action} | {_escape_markdown_table_cell(reason)} |"
        )
    return "\n".join(lines)


def build_human_review_action_summary_markdown(trace_items: list[dict[str, object]]) -> str:
    """Format blocked traces by the next reviewer action needed."""

    action_labels: dict[str, list[str]] = {}
    for item in trace_items:
        trace = item["trace"]
        sufficiency = trace.evidence_sufficiency
        if sufficiency is not None and sufficiency.status == "sufficient":
            continue
        reviewer_action = (
            "run_evidence_evaluation" if sufficiency is None else "inspect_retrieval_and_revise_answer"
        )
        action_labels.setdefault(reviewer_action, []).append(str(item["label"]))

    lines = [
        "## Human Review Action Summary",
        "",
        "| Reviewer action | Trace count | Trace labels |",
        "| --- | ---: | --- |",
    ]
    for action, labels in action_labels.items():
        lines.append(
            f"| {action} | {len(labels)} | "
            f"{_escape_markdown_table_cell(', '.join(labels))} |"
        )
    return "\n".join(lines)



def build_human_review_checklist_markdown(trace_items: list[dict[str, object]]) -> str:
    """Format reviewer checklist tasks for blocked answer traces."""

    lines = ["## Human Review Checklist"]
    for item in trace_items:
        trace = item["trace"]
        sufficiency = trace.evidence_sufficiency
        if sufficiency is not None and sufficiency.status == "sufficient":
            continue
        lines.extend(["", f"### {_escape_markdown_table_cell(str(item['label']))}"])
        if sufficiency is None:
            lines.extend(
                [
                    "- [ ] Run evidence sufficiency evaluation before delivery.",
                    "- [ ] Check unsupported claims and citation support.",
                    "- [ ] Record approval or escalation decision.",
                ]
            )
        else:
            lines.extend(
                [
                    "- [ ] Inspect retrieval trace and retrieved evidence.",
                    "- [ ] Revise answer or rerun retrieval before delivery.",
                    f"- [ ] Confirm evidence sufficiency is no longer {sufficiency.status}.",
                ]
            )
    return "\n".join(lines)



def build_human_review_priority_board_markdown(trace_items: list[dict[str, object]]) -> str:
    """Format blocked traces as a risk-ordered human-review priority board."""

    rows: list[tuple[int, str, str, str, str, str]] = []
    for item in trace_items:
        trace = item["trace"]
        sufficiency = trace.evidence_sufficiency
        if sufficiency is not None and sufficiency.status == "sufficient":
            continue
        if sufficiency is None:
            rows.append(
                (
                    3,
                    "low",
                    str(item["label"]),
                    "not_evaluated",
                    "run_evidence_evaluation",
                    "Run evidence sufficiency evaluation before delivery.",
                )
            )
        elif sufficiency.evidence_count == 0:
            rows.append(
                (
                    2,
                    "medium",
                    str(item["label"]),
                    "no_evidence",
                    "retry_retrieval",
                    "No evidence was retrieved; rewrite or broaden retrieval.",
                )
            )
        elif sufficiency.unsupported_claim_count > 0:
            rows.append(
                (
                    1,
                    "high",
                    str(item["label"]),
                    "unsupported_claims",
                    "revise_or_escalate_answer",
                    f"{sufficiency.unsupported_claim_count} unsupported claims need review before delivery.",
                )
            )
        else:
            rows.append(
                (
                    2,
                    "medium",
                    str(item["label"]),
                    "insufficient_evidence",
                    "inspect_retrieval_and_revise_answer",
                    sufficiency.reason,
                )
            )

    lines = [
        "## Human Review Priority Board",
        "",
        "| Priority | Trace | Risk signal | Reviewer action | Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, priority, label, risk_signal, reviewer_action, reason in sorted(rows, key=lambda row: row[0]):
        lines.append(
            f"| {priority} | {_escape_markdown_table_cell(label)} | {risk_signal} | "
            f"{reviewer_action} | {_escape_markdown_table_cell(reason)} |"
        )
    return "\n".join(lines)


def build_human_review_decision_log_markdown(trace_items: list[dict[str, object]]) -> str:
    """Format reviewer approval outcomes as an auditable human-review decision log."""

    lines = [
        "## Human Review Decision Log",
        "",
        "| Trace | Evidence status | Gate recommendation | Reviewer decision | Audit note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in trace_items:
        trace = item["trace"]
        sufficiency = trace.evidence_sufficiency
        evidence_status = sufficiency.status if sufficiency is not None else "not_evaluated"
        gate_recommendation = "approve_for_delivery" if evidence_status == "sufficient" else "block_for_revision"
        reviewer_decision = str(item["reviewer_decision"])
        audit_note = (
            "Human reviewer accepted the citation-grounded answer."
            if reviewer_decision == "approved"
            else sufficiency.reason
            if sufficiency is not None
            else "Run evidence evaluation before recording approval."
        )
        lines.append(
            f"| {_escape_markdown_table_cell(str(item['label']))} | {evidence_status} | "
            f"{gate_recommendation} | {_escape_markdown_table_cell(reviewer_decision)} | "
            f"{_escape_markdown_table_cell(audit_note)} |"
        )
    return "\n".join(lines)


def build_human_review_decision_summary(trace_items: list[dict[str, object]]) -> dict[str, object]:
    """Count reviewer approval outcomes for human-review audit reporting."""

    decision_counts: dict[str, int] = {}
    blocked_trace_labels: list[str] = []
    for item in trace_items:
        reviewer_decision = str(item["reviewer_decision"])
        decision_counts[reviewer_decision] = decision_counts.get(reviewer_decision, 0) + 1
        if reviewer_decision != "approved":
            blocked_trace_labels.append(str(item["label"]))

    approved_decisions = decision_counts.get("approved", 0)
    total_decisions = len(trace_items)
    return {
        "total_decisions": total_decisions,
        "approved_decisions": approved_decisions,
        "revision_decisions": total_decisions - approved_decisions,
        "decision_counts": decision_counts,
        "blocked_trace_labels": blocked_trace_labels,
    }


def build_human_review_decision_summary_markdown(summary: dict[str, object]) -> str:
    """Format reviewer decision counts as Markdown for dashboard/report pages."""

    blocked_trace_labels = ", ".join(
        _escape_markdown_table_cell(str(label)) for label in summary["blocked_trace_labels"]
    )
    lines = [
        "## Human Review Decision Summary",
        "",
        f"- Total decisions: {summary['total_decisions']}",
        f"- Approved decisions: {summary['approved_decisions']}",
        f"- Revision decisions: {summary['revision_decisions']}",
        f"- Blocked trace labels: {blocked_trace_labels}",
        "",
        "| Reviewer decision | Count |",
        "| --- | ---: |",
    ]
    decision_counts = summary["decision_counts"]
    lines.extend(f"| {decision} | {count} |" for decision, count in decision_counts.items())
    return "\n".join(lines)


def build_human_review_revision_queue_markdown(trace_items: list[dict[str, object]]) -> str:
    """Format non-approved reviewer decisions as a follow-up revision queue."""

    lines = [
        "## Human Review Revision Queue",
        "",
        "| Trace | Reviewer decision | Evidence status | Follow-up action | Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in trace_items:
        reviewer_decision = str(item["reviewer_decision"])
        if reviewer_decision == "approved":
            continue
        trace = item["trace"]
        sufficiency = trace.evidence_sufficiency
        evidence_status = sufficiency.status if sufficiency is not None else "not_evaluated"
        reason = sufficiency.reason if sufficiency is not None else "Run evidence evaluation before revision."
        follow_up_action = (
            "human_escalation_review"
            if reviewer_decision == "escalated"
            else "rerun_retrieval_and_evaluation"
        )
        lines.append(
            f"| {_escape_markdown_table_cell(str(item['label']))} | "
            f"{_escape_markdown_table_cell(reviewer_decision)} | {evidence_status} | "
            f"{follow_up_action} | {_escape_markdown_table_cell(reason)} |"
        )
    return "\n".join(lines)


def build_human_review_workload_summary(trace_items: list[dict[str, object]]) -> dict[str, object]:
    """Count approved and blocked traces for human-review workload planning."""

    summary: dict[str, object] = {
        "total_traces": len(trace_items),
        "approved_traces": 0,
        "blocked_traces": 0,
        "not_evaluated_traces": 0,
        "insufficient_evidence_traces": 0,
        "total_evidence_items_needing_review": 0,
        "total_unsupported_claims": 0,
        "blocked_trace_labels": [],
    }
    blocked_trace_labels: list[str] = []
    for item in trace_items:
        trace = item["trace"]
        sufficiency = trace.evidence_sufficiency
        if sufficiency is not None and sufficiency.status == "sufficient":
            summary["approved_traces"] = int(summary["approved_traces"]) + 1
            continue

        summary["blocked_traces"] = int(summary["blocked_traces"]) + 1
        blocked_trace_labels.append(str(item["label"]))
        if sufficiency is None:
            summary["not_evaluated_traces"] = int(summary["not_evaluated_traces"]) + 1
            summary["total_evidence_items_needing_review"] = int(
                summary["total_evidence_items_needing_review"]
            ) + len(trace.evidence)
            summary["total_unsupported_claims"] = int(summary["total_unsupported_claims"]) + sum(
                check.status == "unsupported" for check in trace.claim_checks
            )
            continue

        summary["insufficient_evidence_traces"] = int(summary["insufficient_evidence_traces"]) + 1
        summary["total_evidence_items_needing_review"] = int(
            summary["total_evidence_items_needing_review"]
        ) + sufficiency.evidence_count
        summary["total_unsupported_claims"] = int(summary["total_unsupported_claims"]) + sufficiency.unsupported_claim_count

    summary["blocked_trace_labels"] = blocked_trace_labels
    return summary


def build_human_review_workload_markdown(summary: dict[str, object]) -> str:
    """Format human-review workload counts as Markdown for dashboard/report pages."""

    blocked_trace_labels = ", ".join(
        _escape_markdown_table_cell(str(label)) for label in summary["blocked_trace_labels"]
    )
    return "\n".join(
        [
            "## Human Review Workload",
            "",
            f"- Total traces: {summary['total_traces']}",
            f"- Approved traces: {summary['approved_traces']}",
            f"- Blocked traces: {summary['blocked_traces']}",
            f"- Not evaluated traces: {summary['not_evaluated_traces']}",
            f"- Insufficient-evidence traces: {summary['insufficient_evidence_traces']}",
            f"- Evidence items needing review: {summary['total_evidence_items_needing_review']}",
            f"- Unsupported claims needing review: {summary['total_unsupported_claims']}",
            f"- Blocked trace labels: {blocked_trace_labels}",
        ]
    )



def build_human_review_escalation_brief_markdown(trace_items: list[dict[str, object]]) -> str:
    """Summarize blocked answer traces for human reviewer escalation."""

    blocked_items = []
    for item in trace_items:
        trace = item["trace"]
        sufficiency = trace.evidence_sufficiency
        if sufficiency is not None and sufficiency.status == "sufficient":
            continue
        trigger = "not_evaluated" if sufficiency is None else "insufficient_evidence"
        reviewer_action = (
            "run_evidence_evaluation" if sufficiency is None else "inspect_retrieval_and_revise_answer"
        )
        evidence_count = sufficiency.evidence_count if sufficiency is not None else len(trace.evidence)
        unsupported_claim_count = (
            sufficiency.unsupported_claim_count
            if sufficiency is not None
            else sum(check.status == "unsupported" for check in trace.claim_checks)
        )
        blocked_items.append(
            {
                "label": str(item["label"]),
                "question": trace.question,
                "trigger": trigger,
                "evidence_count": evidence_count,
                "unsupported_claim_count": unsupported_claim_count,
                "reviewer_action": reviewer_action,
            }
        )

    lines = [
        "## Human Review Escalation Brief",
        "",
        f"- Blocked traces: {len(blocked_items)}",
        f"- Total evidence items needing review: {sum(item['evidence_count'] for item in blocked_items)}",
        f"- Total unsupported claims: {sum(item['unsupported_claim_count'] for item in blocked_items)}",
        "",
        "| Trace | Question | Trigger | Evidence items | Unsupported claims | Reviewer action |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in blocked_items:
        lines.append(
            f"| {_escape_markdown_table_cell(item['label'])} | "
            f"{_escape_markdown_table_cell(item['question'])} | {item['trigger']} | "
            f"{item['evidence_count']} | {item['unsupported_claim_count']} | {item['reviewer_action']} |"
        )
    return "\n".join(lines)


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


def build_evaluation_metric_cards(
    trace: RagTrace, expected_source_ids: list[str]
) -> list[dict[str, str]]:
    """Build dashboard-ready evaluation metric cards for one RAG trace."""

    retrieval_coverage = measure_retrieval_coverage(trace, expected_source_ids)
    citation_support = measure_citation_support(trace)
    unsupported_claim_count = sum(check.status == "unsupported" for check in trace.claim_checks)
    checked_claim_count = len(trace.claim_checks)

    found_count = len(retrieval_coverage["found_source_ids"])
    expected_count = len(retrieval_coverage["expected_source_ids"])
    supported_citation_count = len(citation_support["supported_cited_source_ids"])
    cited_count = len(citation_support["cited_source_ids"])

    return [
        {
            "label": "Retrieval coverage",
            "value": _format_percent(float(retrieval_coverage["coverage_ratio"])),
            "status": "pass" if retrieval_coverage["coverage_ratio"] == 1.0 else "fail",
            "detail": f"Found {found_count} of {expected_count} expected sources.",
        },
        {
            "label": "Citation support",
            "value": _format_percent(float(citation_support["citation_support_ratio"])),
            "status": "pass" if citation_support["citation_support_ratio"] == 1.0 else "fail",
            "detail": f"Supported {supported_citation_count} of {cited_count} cited sources.",
        },
        {
            "label": "Unsupported claims",
            "value": str(unsupported_claim_count),
            "status": "pass" if unsupported_claim_count == 0 else "fail",
            "detail": f"{unsupported_claim_count} unsupported claims out of {checked_claim_count} checked claims.",
        },
    ]


def build_latency_metric_cards(
    trace: RagTrace, slow_threshold_ms: float | None = None
) -> list[dict[str, str]]:
    """Build dashboard-ready latency cards for one RAG trace."""

    step_count = len(trace.retrieval_steps)
    total_latency_ms = sum(step.latency_ms for step in trace.retrieval_steps)
    average_latency_ms = total_latency_ms / step_count if step_count else 0.0
    is_slow = slow_threshold_ms is not None and total_latency_ms > slow_threshold_ms
    status = "warn" if is_slow else "pass"
    total_detail = (
        f"{step_count} retrieval steps exceeded the {slow_threshold_ms:.2f} ms slow threshold."
        if is_slow
        else f"{step_count} retrieval steps completed without recorded errors."
    )
    return [
        {
            "label": "Total retrieval latency",
            "value": f"{total_latency_ms:.2f} ms",
            "status": status,
            "detail": total_detail,
        },
        {
            "label": "Average retrieval latency",
            "value": f"{average_latency_ms:.2f} ms",
            "status": status,
            "detail": f"Average latency across {step_count} retrieval steps.",
        },
    ]


def build_retrieval_error_metric_cards(trace: RagTrace) -> list[dict[str, str]]:
    """Build dashboard-ready error cards for failed retrieval steps."""

    error_steps = [
        (step_index, step)
        for step_index, step in enumerate(trace.retrieval_steps, start=1)
        if step.error
    ]
    error_count = len(error_steps)
    step_count = len(trace.retrieval_steps)
    status = "fail" if error_count else "pass"
    cards = [
        {
            "label": "Retrieval errors",
            "value": str(error_count),
            "status": status,
            "detail": f"{error_count} of {step_count} retrieval steps recorded errors.",
        }
    ]
    if error_steps:
        first_error_index, first_error_step = error_steps[0]
        cards.append(
            {
                "label": "First retrieval error",
                "value": f"query {first_error_index}",
                "status": "fail",
                "detail": first_error_step.error or "",
            }
        )
    return cards


def build_observability_dashboard_sections(
    trace: RagTrace,
    expected_source_ids: list[str],
    slow_threshold_ms: float | None = None,
) -> list[dict[str, object]]:
    """Group one trace's dashboard cards into render-ready observability sections."""

    return [
        {
            "title": "Trace health",
            "cards": build_trace_health_metric_cards(trace),
        },
        {
            "title": "Evaluation quality",
            "cards": build_evaluation_metric_cards(trace, expected_source_ids),
        },
        {
            "title": "Retrieval latency",
            "cards": build_latency_metric_cards(trace, slow_threshold_ms=slow_threshold_ms),
        },
        {
            "title": "Retrieval errors",
            "cards": build_retrieval_error_metric_cards(trace),
        },
    ]


def build_observability_dashboard_markdown(sections: list[dict[str, object]]) -> str:
    """Format dashboard sections as Markdown tables for demo/report rendering."""

    lines = ["## Observability Dashboard"]
    for section in sections:
        lines.extend(
            [
                "",
                f"### {section['title']}",
                "| Metric | Value | Status | Detail |",
                "| --- | --- | --- | --- |",
            ]
        )
        for card in section["cards"]:
            lines.append(
                "| "
                + " | ".join(
                    _escape_markdown_table_cell(str(card[field]))
                    for field in ("label", "value", "status", "detail")
                )
                + " |"
            )
    return "\n".join(lines)



def build_trace_health_metric_cards(trace: RagTrace) -> list[dict[str, str]]:
    """Build dashboard-ready health cards that summarize one run's readiness."""

    sufficiency = trace.evidence_sufficiency
    sufficiency_status = sufficiency.status if sufficiency is not None else "not_evaluated"
    sufficiency_reason = (
        sufficiency.reason if sufficiency is not None else "No evidence sufficiency summary is attached."
    )
    planned_query_count = len(trace.planned_queries)
    retrieval_step_count = len(trace.retrieval_steps)
    evidence_count = len(trace.evidence)
    supported_claim_count = sum(check.status == "supported" for check in trace.claim_checks)
    unsupported_claim_count = sum(check.status == "unsupported" for check in trace.claim_checks)
    checked_claim_count = len(trace.claim_checks)
    trace_is_complete = planned_query_count == retrieval_step_count and evidence_count > 0

    return [
        {
            "label": "Evidence sufficiency",
            "value": sufficiency_status,
            "status": "pass" if sufficiency_status == "sufficient" else "fail",
            "detail": sufficiency_reason,
        },
        {
            "label": "Trace completeness",
            "value": f"{retrieval_step_count}/{planned_query_count} steps",
            "status": "pass" if trace_is_complete else "fail",
            "detail": (
                f"{planned_query_count} planned queries produced {retrieval_step_count} retrieval steps "
                f"and {evidence_count} evidence items."
            ),
        },
        {
            "label": "Claim support",
            "value": f"{supported_claim_count} supported / {unsupported_claim_count} unsupported",
            "status": "pass" if unsupported_claim_count == 0 else "fail",
            "detail": f"Checked {checked_claim_count} answer claims against retrieved evidence.",
        },
    ]



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


def build_failure_diagnosis_markdown(diagnoses: list[dict[str, str]]) -> str:
    """Format question-level RAG failure diagnoses as a Markdown debug table."""

    lines = [
        "## Failure Diagnosis Report",
        "",
        "| Question | Failure mode | Reason |",
        "| --- | --- | --- |",
    ]
    for diagnosis in diagnoses:
        lines.append(
            "| "
            + " | ".join(
                _escape_markdown_table_cell(diagnosis[field])
                for field in ("question", "failure_mode", "reason")
            )
            + " |"
        )
    return "\n".join(lines)


def build_retrieval_gap_report_markdown(quality_report: dict[str, object]) -> str:
    """Format retrieval-coverage gaps as a focused Markdown debug table."""

    lines = [
        "## Retrieval Gap Report",
        "",
        "| Question | Coverage | Missing sources | Retrieved sources |",
        "| --- | ---: | --- | --- |",
    ]
    for result in quality_report.get("results", []):
        retrieval_coverage = result["retrieval_coverage"]
        missing_source_ids = retrieval_coverage["missing_source_ids"]
        if not missing_source_ids:
            continue
        retrieved_source_ids = retrieval_coverage["retrieved_source_ids"]
        lines.append(
            f"| {_escape_markdown_table_cell(result['question'])} | "
            f"{_format_percent(float(retrieval_coverage['coverage_ratio']))} | "
            f"{_escape_markdown_table_cell(', '.join(missing_source_ids))} | "
            f"{_escape_markdown_table_cell(', '.join(retrieved_source_ids) or 'none')} |"
        )
    return "\n".join(lines)


def build_unsupported_citation_report_markdown(quality_report: dict[str, object]) -> str:
    """Format unsupported citations as a focused Markdown debug table."""

    lines = [
        "## Unsupported Citation Report",
        "",
        "| Question | Citation support | Unsupported citations | Retrieved evidence sources |",
        "| --- | ---: | --- | --- |",
    ]
    for result in quality_report.get("results", []):
        citation_support = result["citation_support"]
        unsupported_cited_source_ids = citation_support["unsupported_cited_source_ids"]
        if not unsupported_cited_source_ids:
            continue
        evidence_source_ids = citation_support["evidence_source_ids"]
        lines.append(
            f"| {_escape_markdown_table_cell(result['question'])} | "
            f"{_format_percent(float(citation_support['citation_support_ratio']))} | "
            f"{_escape_markdown_table_cell(', '.join(unsupported_cited_source_ids))} | "
            f"{_escape_markdown_table_cell(', '.join(evidence_source_ids) or 'none')} |"
        )
    return "\n".join(lines)


def build_evidence_sufficiency_gap_report_markdown(trace_items: list[dict[str, object]]) -> str:
    """Format insufficient traces as a focused Markdown debug table."""

    lines = [
        "## Evidence Sufficiency Gap Report",
        "",
        "| Trace | Status | Evidence items | Unsupported claims | Reason |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in trace_items:
        trace = item["trace"]
        sufficiency = trace.evidence_sufficiency
        if sufficiency is None or sufficiency.status != "insufficient":
            continue
        lines.append(
            f"| {_escape_markdown_table_cell(str(item['label']))} | "
            f"{_escape_markdown_table_cell(sufficiency.status)} | "
            f"{sufficiency.evidence_count} | "
            f"{sufficiency.unsupported_claim_count} | "
            f"{_escape_markdown_table_cell(sufficiency.reason)} |"
        )
    return "\n".join(lines)


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


def build_unsupported_claim_report_markdown(claim_checks: list[ClaimCheck]) -> str:
    """Format unsupported answer claims as a Markdown review table."""

    lines = [
        "## Unsupported Claim Review",
        "",
        "| Claim | Status | Supporting sources |",
        "| --- | --- | --- |",
    ]
    for check in claim_checks:
        if check.status != "unsupported":
            continue
        supporting_sources = ", ".join(check.supporting_source_ids) or "none"
        lines.append(
            f"| {_escape_markdown_table_cell(check.claim)} | {check.status} | "
            f"{_escape_markdown_table_cell(supporting_sources)} |"
        )
    return "\n".join(lines)


def build_retrieval_retry_report_markdown(trace: RagTrace) -> str:
    """Format weak-evidence retrieval retries as a Markdown recovery report."""

    lines = [
        "## Retrieval Retry Report",
        "",
        "| Retry | Original query | Retry query | Before sources | After sources | Status |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    retry_count = 0
    for query_index, planned_query in enumerate(trace.planned_queries):
        if planned_query.reason != "Retry retrieval with a rewritten query after weak evidence.":
            continue
        retry_count += 1
        previous_step = trace.retrieval_steps[query_index - 1]
        retry_step = trace.retrieval_steps[query_index]
        before_sources = ", ".join(previous_step.retrieved_source_ids) or "none"
        after_sources = ", ".join(retry_step.retrieved_source_ids) or "none"
        status = "recovered" if retry_step.retrieved_source_ids else "unresolved"
        lines.append(
            f"| {retry_count} | {_escape_markdown_table_cell(previous_step.query)} | "
            f"{_escape_markdown_table_cell(planned_query.query)} | "
            f"{_escape_markdown_table_cell(before_sources)} | "
            f"{_escape_markdown_table_cell(after_sources)} | {status} |"
        )
    return "\n".join(lines)


def build_retrieval_plan_markdown(trace: RagTrace) -> str:
    """Format planned retrieval queries and reasons as a Markdown plan table."""

    lines = [
        "## Retrieval Plan",
        "",
        "| Step | Query | Reason | Retrieved sources |",
        "| ---: | --- | --- | --- |",
    ]
    for query_index, planned_query in enumerate(trace.planned_queries, start=1):
        sources = "none"
        if query_index <= len(trace.retrieval_steps):
            sources = ", ".join(trace.retrieval_steps[query_index - 1].retrieved_source_ids) or "none"
        lines.append(
            f"| {query_index} | {_escape_markdown_table_cell(planned_query.query)} | "
            f"{_escape_markdown_table_cell(planned_query.reason)} | "
            f"{_escape_markdown_table_cell(sources)} |"
        )
    return "\n".join(lines)



def build_source_attribution_markdown(trace: RagTrace) -> str:
    """Format retrieved source attribution as a Markdown table."""

    cited_source_ids = set(re.findall(r"\[([^\]]+)\]", trace.answer))
    supported_claim_counts: dict[str, int] = {}
    for check in trace.claim_checks:
        if check.status != "supported":
            continue
        for source_id in check.supporting_source_ids:
            supported_claim_counts[source_id] = supported_claim_counts.get(source_id, 0) + 1

    lines = [
        "## Source Attribution",
        "",
        "| Source | Evidence ranks | Cited in answer | Supported claims | Top score |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    source_rows: dict[str, dict[str, object]] = {}
    for rank, item in enumerate(trace.evidence, start=1):
        row = source_rows.setdefault(
            item.source_id,
            {"ranks": [], "top_score": item.score},
        )
        row["ranks"].append(str(rank))
        row["top_score"] = max(float(row["top_score"]), item.score)

    for source_id, row in source_rows.items():
        ranks = ", ".join(row["ranks"])
        cited = "yes" if source_id in cited_source_ids else "no"
        supported_claim_count = supported_claim_counts.get(source_id, 0)
        lines.append(
            f"| {_escape_markdown_table_cell(source_id)} | {ranks} | {cited} | "
            f"{supported_claim_count} | {float(row['top_score']):.2f} |"
        )
    return "\n".join(lines)


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
        lines.append(
            f"| {step_index} | {_escape_markdown_table_cell(step.query)} | "
            f"{_escape_markdown_table_cell(sources)} | {_escape_markdown_table_cell(chunks)} |"
        )

    lines.extend(
        [
            "",
            "### Evidence",
            "| Rank | Source | Chunk | Score | Snippet |",
            "| ---: | --- | --- | ---: | --- |",
        ]
    )
    for rank, item in enumerate(trace.evidence, start=1):
        chunk_id = str(item.metadata.get("chunk_id", ""))
        lines.append(
            f"| {rank} | {_escape_markdown_table_cell(item.source_id)} | "
            f"{_escape_markdown_table_cell(chunk_id)} | {item.score:.2f} | "
            f"{_escape_markdown_table_cell(item.text)} |"
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
        "total_retrieval_latency_ms": sum(step.latency_ms for step in trace.retrieval_steps),
    }


def build_trace_timeline_events(trace: RagTrace) -> list[dict[str, object]]:
    """Return ordered timeline events for visualizing one RAG run."""

    events: list[dict[str, object]] = []
    order = 1
    for query_index, planned_query in enumerate(trace.planned_queries, start=1):
        events.append(
            {
                "order": order,
                "event_type": "plan_query",
                "title": f"Planned query {query_index}",
                "detail": planned_query.query,
                "source_ids": [],
                "chunk_ids": [],
            }
        )
        order += 1
        if query_index <= len(trace.retrieval_steps):
            retrieval_step = trace.retrieval_steps[query_index - 1]
            events.append(
                {
                    "order": order,
                    "event_type": "retrieve_evidence",
                    "title": f"Retrieved evidence for query {query_index}",
                    "detail": retrieval_step.query,
                    "source_ids": retrieval_step.retrieved_source_ids,
                    "chunk_ids": retrieval_step.retrieved_chunk_ids,
                }
            )
            order += 1

    evidence_source_ids = [item.source_id for item in trace.evidence]
    evidence_chunk_ids = [item.metadata.get("chunk_id", "") for item in trace.evidence]
    events.append(
        {
            "order": order,
            "event_type": "synthesize_answer",
            "title": "Synthesized citation-grounded answer",
            "detail": f"{len(trace.claim_checks)} answer claims checked against {len(trace.evidence)} evidence items.",
            "source_ids": evidence_source_ids,
            "chunk_ids": evidence_chunk_ids,
        }
    )
    order += 1

    if trace.evidence_sufficiency is not None:
        events.append(
            {
                "order": order,
                "event_type": "evaluate_sufficiency",
                "title": "Evaluated evidence sufficiency",
                "detail": f"{trace.evidence_sufficiency.status}: {trace.evidence_sufficiency.reason}",
                "source_ids": evidence_source_ids,
                "chunk_ids": evidence_chunk_ids,
            }
        )
    return events




def build_trace_timeline_markdown(events: list[dict[str, object]]) -> str:
    """Format trace timeline events as a Markdown table for demo/report rendering."""

    lines = [
        "## Trace Timeline",
        "",
        "| Order | Event | Detail | Sources | Chunks |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for event in events:
        sources = ", ".join(str(source_id) for source_id in event["source_ids"]) or "none"
        chunks = ", ".join(str(chunk_id) for chunk_id in event["chunk_ids"]) or "none"
        lines.append(
            f"| {event['order']} | {_escape_markdown_table_cell(str(event['title']))} | "
            f"{_escape_markdown_table_cell(str(event['detail']))} | "
            f"{_escape_markdown_table_cell(sources)} | {_escape_markdown_table_cell(chunks)} |"
        )
    return "\n".join(lines)


def build_trace_interview_summary_markdown(
    trace: RagTrace, expected_source_ids: list[str]
) -> str:
    """Format one trace as an interviewer-friendly Agentic RAG summary."""

    retrieval_coverage = measure_retrieval_coverage(trace, expected_source_ids)
    citation_support = measure_citation_support(trace)
    unsupported_claim_count = sum(check.status == "unsupported" for check in trace.claim_checks)
    sufficiency = trace.evidence_sufficiency
    sufficiency_text = (
        f"{sufficiency.status} — {sufficiency.reason}"
        if sufficiency is not None
        else "not evaluated — No evidence sufficiency summary is attached."
    )
    return "\n".join(
        [
            "## Interview Trace Summary",
            "",
            "- Agent concept: query decomposition + citation-grounded synthesis + faithfulness evaluation",
            f"- Question: {trace.question}",
            f"- Planned retrieval queries: {len(trace.planned_queries)}",
            f"- Retrieved evidence items: {len(trace.evidence)}",
            f"- Retrieval coverage: {_format_percent(float(retrieval_coverage['coverage_ratio']))}",
            f"- Citation support: {_format_percent(float(citation_support['citation_support_ratio']))}",
            f"- Unsupported claims: {unsupported_claim_count}",
            f"- Evidence sufficiency: {sufficiency_text}",
        ]
    )



def build_interview_evidence_packet_markdown(trace: RagTrace) -> str:
    """Format a concise interview/demo packet for one traceable RAG run."""

    sufficiency = trace.evidence_sufficiency
    answer_status = sufficiency.status if sufficiency is not None else "not_evaluated"
    evidence_sources = ", ".join(dict.fromkeys(item.source_id for item in trace.evidence)) or "none"
    supported_claim_count = sum(check.status == "supported" for check in trace.claim_checks)
    unsupported_claim_count = sum(check.status == "unsupported" for check in trace.claim_checks)
    source_count = len(dict.fromkeys(item.source_id for item in trace.evidence))

    return "\n".join(
        [
            "## Interview Evidence Packet",
            "",
            f"**Question:** {trace.question}",
            f"**Answer status:** {answer_status}",
            f"**Evidence sources:** {evidence_sources}",
            f"**Unsupported claims:** {unsupported_claim_count}",
            "",
            "### Agentic RAG concepts demonstrated",
            f"- Query planning: {len(trace.planned_queries)} retrieval query planned before answering.",
            f"- Evidence grounding: {len(trace.evidence)} evidence item selected and cited from {source_count} source.",
            f"- Faithfulness check: {supported_claim_count} supported claims and {unsupported_claim_count} unsupported claims.",
            "",
            "### Demo talking point",
            "This trace shows a RAG agent that does not just answer; it records retrieval, citations, and evidence sufficiency so a reviewer can verify why the answer is safe to deliver.",
        ]
    )


def build_interview_walkthrough_markdown(trace: RagTrace) -> str:
    """Format a trace as a short interview walkthrough story."""

    sufficiency = trace.evidence_sufficiency
    sufficiency_status = sufficiency.status if sufficiency is not None else "not_evaluated"
    sufficiency_reason = (
        sufficiency.reason if sufficiency is not None else "No evidence sufficiency summary is attached."
    )
    supported_claim_count = sum(check.status == "supported" for check in trace.claim_checks)
    unsupported_claim_count = sum(check.status == "unsupported" for check in trace.claim_checks)
    cited_sources = ", ".join(dict.fromkeys(re.findall(r"\[([^\]]+)\]", trace.answer))) or "none"
    delivery_status = "ready_for_demo" if sufficiency_status == "sufficient" else "needs_review"

    return "\n".join(
        [
            "## Interview Walkthrough",
            "",
            f"**User question:** {trace.question}",
            f"**Delivery status:** {delivery_status}",
            "",
            "### What the agent did",
            f"1. Planned {len(trace.planned_queries)} retrieval queries before answering.",
            f"2. Ran {len(trace.retrieval_steps)} retrieval steps and selected {len(trace.evidence)} evidence items.",
            f"3. Generated a citation-grounded answer and checked {len(trace.claim_checks)} claims.",
            "",
            "### Evidence and evaluation",
            f"- Evidence sufficiency: {sufficiency_status} — {sufficiency_reason}",
            f"- Supported claims: {supported_claim_count}",
            f"- Unsupported claims: {unsupported_claim_count}",
            f"- Cited sources: {cited_sources}",
            "",
            "### Interview framing",
            "This is an Agentic RAG trace: the system plans retrieval, records evidence, checks faithfulness, and exposes whether the answer is safe to deliver.",
        ]
    )



def build_citation_evidence_map(trace: RagTrace) -> list[dict[str, object]]:
    """Link answer citations to retrieved evidence for citation/evidence visualization."""

    cited_source_ids = list(dict.fromkeys(re.findall(r"\[([^\]]+)\]", trace.answer)))
    evidence_by_source_id = {item.source_id: (rank, item) for rank, item in enumerate(trace.evidence, start=1)}
    return [
        {
            "citation": source_id,
            "is_retrieved": source_id in evidence_by_source_id,
            "evidence_rank": evidence_by_source_id[source_id][0] if source_id in evidence_by_source_id else None,
            "chunk_id": evidence_by_source_id[source_id][1].metadata.get("chunk_id", "")
            if source_id in evidence_by_source_id
            else "",
            "snippet": evidence_by_source_id[source_id][1].text if source_id in evidence_by_source_id else "",
            "supporting_claim_count": sum(
                source_id in check.supporting_source_ids for check in trace.claim_checks
            ),
        }
        for source_id in cited_source_ids
    ]


def build_citation_evidence_map_markdown(citation_map: list[dict[str, object]]) -> str:
    """Format citation-to-evidence links as a Markdown table for demo review."""

    lines = [
        "## Citation Evidence Map",
        "",
        "| Citation | Retrieved? | Evidence rank | Chunk | Supporting claims | Snippet |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for item in citation_map:
        evidence_rank = item["evidence_rank"] if item["evidence_rank"] is not None else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown_table_cell(str(item["citation"])),
                    "yes" if item["is_retrieved"] else "no",
                    str(evidence_rank),
                    _escape_markdown_table_cell(str(item["chunk_id"])),
                    str(item["supporting_claim_count"]),
                    _escape_markdown_table_cell(str(item["snippet"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


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


def _format_percent(ratio: float) -> str:
    return f"{ratio:.0%}"


def _recommend_recovery_action(trace: RagTrace) -> tuple[str, str, str]:
    if trace.evidence_sufficiency is None:
        return (
            "not_evaluated",
            "run_evidence_evaluation",
            "No evidence sufficiency summary exists; run faithfulness evaluation before delivery.",
        )
    if trace.evidence_sufficiency.evidence_count == 0:
        return (
            "no_evidence",
            "retry_retrieval",
            "No evidence was retrieved; rewrite the query or broaden retrieval before answering.",
        )
    if trace.evidence_sufficiency.unsupported_claim_count > 0:
        count = trace.evidence_sufficiency.unsupported_claim_count
        return (
            "unsupported_claims",
            "revise_or_escalate_answer",
            f"{count} answer claims are unsupported; revise synthesis or escalate for review before delivery.",
        )
    return (
        "ready",
        "deliver_answer",
        "Evidence is sufficient and answer claims are supported.",
    )


def _escape_markdown_table_cell(value: str) -> str:
    return value.replace("|", r"\|")



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


def _rewrite_weak_retrieval_query(query: str) -> str | None:
    query_terms = _terms(query)
    if query_terms & {"hallucination", "hallucinations", "hallucinate", "hallucinated"}:
        return "unsupported claims evidence"
    return None


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _strip_citations(text: str) -> str:
    return re.sub(r"\s*\[[^\]]+\]", "", text)


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))
