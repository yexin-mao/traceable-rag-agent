import json

from traceable_rag_agent.models import (
    ClaimCheck,
    Evidence,
    EvidenceSufficiency,
    RagTrace,
    RetrievalQuery,
    RetrievalStep,
)

from traceable_rag_agent.pipeline import (
    BenchmarkQuestion,
    Document,
    answer_question,
    build_evidence_table,
    build_evidence_decision_markdown,
    build_human_review_escalation_brief_markdown,
    build_human_review_queue_markdown,
    build_human_review_action_summary_markdown,
    build_human_review_decision_summary_markdown,
    build_human_review_revision_queue_markdown,
    build_human_review_workload_markdown,
    build_human_review_workload_summary,
    build_interview_evidence_packet_markdown,
    build_interview_followup_questions_markdown,
    build_interview_demo_script_markdown,
    build_interview_objection_handling_markdown,
    build_interview_concept_map_markdown,
    build_interview_readiness_scorecard_markdown,
    build_interview_walkthrough_markdown,
    build_quality_report_markdown,
    build_recovery_action_plan_markdown,
    build_retrieval_error_metric_cards,
    build_retrieval_plan_markdown,
    build_source_attribution_markdown,
    build_trace_report_markdown,
    build_trace_replay_plan_markdown,
    build_trace_status_summary,
    build_trace_timeline_events,
    build_trace_timeline_markdown,
    build_citation_evidence_map,
    build_citation_evidence_map_markdown,
    build_evaluation_metric_cards,
    build_latency_metric_cards,
    build_observability_dashboard_markdown,
    build_observability_dashboard_sections,
    build_trace_health_metric_cards,
    build_trace_interview_summary_markdown,
    check_answer_claims,
    check_evidence_sufficiency,
    chunk_documents,
    diagnose_rag_quality_failures,
    build_failure_diagnosis_markdown,
    build_retrieval_gap_report_markdown,
    measure_citation_support,
    measure_retrieval_coverage,
    run_rag_quality_benchmark,
    run_retrieval_coverage_benchmark,
    recommend_next_evaluation_action,
    summarize_failure_diagnoses,
    plan_retrieval_queries,
    retrieve,
    save_trace_json,
)


def test_chunk_documents_preserves_source_ids_and_ordered_chunk_ids() -> None:
    documents = [
        Document(source_id="langgraph", text="LangGraph stores agent state across steps."),
        Document(source_id="rag", text="RAG retrieves evidence before synthesis."),
    ]

    chunks = chunk_documents(documents, max_words=4)

    assert [chunk.chunk_id for chunk in chunks] == ["langgraph#0", "langgraph#1", "rag#0", "rag#1"]
    assert chunks[0].text == "LangGraph stores agent state"
    assert chunks[1].text == "across steps."
    assert chunks[2].source_id == "rag"


def test_retrieve_ranks_chunks_by_query_term_overlap() -> None:
    chunks = chunk_documents(
        [
            Document(source_id="simple", text="A simple chain answers after one retrieval."),
            Document(
                source_id="agentic",
                text="Agentic RAG decomposes questions and checks evidence sufficiency.",
            ),
        ],
        max_words=20,
    )

    evidence = retrieve("How does agentic RAG check evidence?", chunks, top_k=1)

    assert len(evidence) == 1
    assert evidence[0].source_id == "agentic"
    assert evidence[0].metadata["chunk_id"] == "agentic#0"
    assert evidence[0].score > 0


def test_plan_retrieval_queries_splits_complex_and_question_into_focused_subqueries() -> None:
    queries = plan_retrieval_queries("How does Agentic RAG decompose questions and check evidence?")

    assert [query.query for query in queries] == [
        "How does Agentic RAG decompose questions?",
        "How does Agentic RAG check evidence?",
    ]
    assert queries[0].reason == "Retrieve evidence for one focused part of the complex question."
    assert queries[1].reason == "Retrieve evidence for one focused part of the complex question."


def test_answer_question_returns_citation_grounded_skeleton_trace() -> None:
    documents = [
        Document(source_id="agentic-rag", text="Agentic RAG checks evidence before answering."),
        Document(source_id="observability", text="Trace logs explain each retrieval step."),
    ]

    trace = answer_question("What does Agentic RAG check before answering?", documents, top_k=1)

    assert trace.question == "What does Agentic RAG check before answering?"
    assert trace.planned_queries[0].query == "What does Agentic RAG check before answering?"
    assert trace.evidence[0].source_id == "agentic-rag"
    assert "Agentic RAG checks evidence before answering." in trace.answer
    assert "[agentic-rag]" in trace.answer
    assert trace.claim_checks[0].status == "supported"
    assert trace.claim_checks[0].supporting_source_ids == ["agentic-rag"]


def test_answer_question_retrieves_evidence_for_each_planned_subquery() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]

    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )

    assert [query.query for query in trace.planned_queries] == [
        "How does Agentic RAG decompose questions?",
        "How does Agentic RAG check evidence sufficiency?",
    ]
    assert [item.source_id for item in trace.evidence] == ["decomposition", "sufficiency"]


def test_answer_question_records_retrieval_steps_for_each_planned_query() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]

    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )

    assert [step.query for step in trace.retrieval_steps] == [
        "How does Agentic RAG decompose questions?",
        "How does Agentic RAG check evidence sufficiency?",
    ]
    assert [step.retrieved_source_ids for step in trace.retrieval_steps] == [
        ["decomposition"],
        ["sufficiency"],
    ]
    assert [step.retrieved_chunk_ids for step in trace.retrieval_steps] == [
        ["decomposition#0"],
        ["sufficiency#0"],
    ]
    assert all(step.latency_ms >= 0 for step in trace.retrieval_steps)


def test_check_answer_claims_flags_claims_without_supporting_evidence() -> None:
    evidence = [
        Evidence(
            source_id="agentic-rag",
            text="Agentic RAG checks retrieved evidence before answering.",
            score=1.0,
        )
    ]

    checks = check_answer_claims(
        "Agentic RAG checks retrieved evidence before answering. It guarantees perfect answers.",
        evidence,
    )

    assert [check.status for check in checks] == ["supported", "unsupported"]
    assert checks[0].supporting_source_ids == ["agentic-rag"]
    assert checks[1].supporting_source_ids == []


def test_check_evidence_sufficiency_marks_supported_answers_sufficient() -> None:
    evidence = [
        Evidence(
            source_id="agentic-rag",
            text="Agentic RAG checks retrieved evidence before answering.",
            score=1.0,
        )
    ]
    claim_checks = check_answer_claims(
        "Agentic RAG checks retrieved evidence before answering.",
        evidence,
    )

    result = check_evidence_sufficiency(evidence, claim_checks)

    assert result.status == "sufficient"
    assert result.supported_claim_count == 1
    assert result.unsupported_claim_count == 0
    assert result.evidence_count == 1
    assert result.reason == "All answer claims are supported by retrieved evidence."




def test_build_evidence_decision_markdown_recommends_publish_or_recovery_action() -> None:
    supported_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    insufficient_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )

    markdown = build_evidence_decision_markdown(
        [
            {"label": "supported demo", "trace": supported_trace},
            {"label": "missing | evidence demo", "trace": insufficient_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Evidence Decision Log",
            "",
            "| Trace | Sufficiency | Decision | Reason |",
            "| --- | --- | --- | --- |",
            "| supported demo | sufficient | ready_for_answer | All answer claims are supported by retrieved evidence. |",
            "| missing \\| evidence demo | insufficient | retry_or_escalate | No evidence was retrieved for this question. |",
        ]
    )



def test_build_recovery_action_plan_markdown_maps_insufficient_traces_to_next_steps() -> None:
    no_evidence_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )
    unsupported_claim_trace = RagTrace(
        question="How should the answer be reviewed?",
        planned_queries=[
            RetrievalQuery(
                query="How should the answer be reviewed?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[
            RetrievalStep(
                query="How should the answer be reviewed?",
                retrieved_source_ids=["review"],
                retrieved_chunk_ids=["review#0"],
            )
        ],
        evidence=[Evidence(source_id="review", text="Review claims against evidence.", score=0.9)],
        answer="Review claims against evidence. It is always perfect.",
        claim_checks=[
            ClaimCheck(claim="Review claims against evidence.", status="supported", supporting_source_ids=["review"]),
            ClaimCheck(claim="It is always perfect.", status="unsupported", supporting_source_ids=[]),
        ],
        evidence_sufficiency=check_evidence_sufficiency(
            [Evidence(source_id="review", text="Review claims against evidence.", score=0.9)],
            [
                ClaimCheck(claim="Review claims against evidence.", status="supported", supporting_source_ids=["review"]),
                ClaimCheck(claim="It is always perfect.", status="unsupported", supporting_source_ids=[]),
            ],
        ),
    )

    markdown = build_recovery_action_plan_markdown(
        [
            {"label": "missing | evidence", "trace": no_evidence_trace},
            {"label": "unsupported claim", "trace": unsupported_claim_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Recovery Action Plan",
            "",
            "| Trace | Failure signal | Recommended action | Why |",
            "| --- | --- | --- | --- |",
            "| missing \\| evidence | no_evidence | retry_retrieval | No evidence was retrieved; rewrite the query or broaden retrieval before answering. |",
            "| unsupported claim | unsupported_claims | revise_or_escalate_answer | 1 answer claims are unsupported; revise synthesis or escalate for review before delivery. |",
        ]
    )


def test_build_human_review_queue_markdown_lists_only_blocked_answer_traces() -> None:
    approved_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    blocked_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )

    markdown = build_human_review_queue_markdown(
        [
            {"label": "approved trace", "trace": approved_trace},
            {"label": "missing | evidence", "trace": blocked_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Human Review Queue",
            "",
            "| Trace | Review trigger | Recommended reviewer action | Reason |",
            "| --- | --- | --- | --- |",
            "| missing \\| evidence | insufficient_evidence | inspect_retrieval_and_revise_answer | No evidence was retrieved for this question. |",
        ]
    )


def test_build_human_review_action_summary_markdown_counts_reviewer_next_actions() -> None:
    blocked_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )
    not_evaluated_trace = RagTrace(
        question="How should unevaluated answers be handled?",
        planned_queries=[
            RetrievalQuery(
                query="How should unevaluated answers be handled?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[],
        evidence=[],
        answer="Do not deliver without checks.",
        claim_checks=[],
        evidence_sufficiency=None,
    )
    approved_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )

    markdown = build_human_review_action_summary_markdown(
        [
            {"label": "missing | evidence", "trace": blocked_trace},
            {"label": "not evaluated", "trace": not_evaluated_trace},
            {"label": "approved", "trace": approved_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Human Review Action Summary",
            "",
            "| Reviewer action | Trace count | Trace labels |",
            "| --- | ---: | --- |",
            "| inspect_retrieval_and_revise_answer | 1 | missing \\| evidence |",
            "| run_evidence_evaluation | 1 | not evaluated |",
        ]
    )



def test_build_human_review_checklist_markdown_formats_operator_tasks_for_blocked_traces() -> None:
    from traceable_rag_agent import pipeline

    blocked_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )
    not_evaluated_trace = RagTrace(
        question="How should unevaluated answers be handled?",
        planned_queries=[
            RetrievalQuery(
                query="How should unevaluated answers be handled?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[],
        evidence=[],
        answer="Do not deliver without checks.",
        claim_checks=[],
        evidence_sufficiency=None,
    )
    approved_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )

    markdown = pipeline.build_human_review_checklist_markdown(
        [
            {"label": "missing | evidence", "trace": blocked_trace},
            {"label": "not evaluated", "trace": not_evaluated_trace},
            {"label": "approved", "trace": approved_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Human Review Checklist",
            "",
            "### missing \\| evidence",
            "- [ ] Inspect retrieval trace and retrieved evidence.",
            "- [ ] Revise answer or rerun retrieval before delivery.",
            "- [ ] Confirm evidence sufficiency is no longer insufficient.",
            "",
            "### not evaluated",
            "- [ ] Run evidence sufficiency evaluation before delivery.",
            "- [ ] Check unsupported claims and citation support.",
            "- [ ] Record approval or escalation decision.",
        ]
    )



def test_build_human_review_priority_board_markdown_orders_blocked_traces_by_risk() -> None:
    from traceable_rag_agent import pipeline

    unsupported_claim_trace = RagTrace(
        question="How should unsupported answers be reviewed?",
        planned_queries=[
            RetrievalQuery(
                query="How should unsupported answers be reviewed?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[
            RetrievalStep(
                query="How should unsupported answers be reviewed?",
                retrieved_source_ids=["review"],
                retrieved_chunk_ids=["review#0"],
            )
        ],
        evidence=[Evidence(source_id="review", text="Review claims against evidence.", score=0.9)],
        answer="Review claims against evidence. It is always perfect.",
        claim_checks=[
            ClaimCheck(claim="Review claims against evidence.", status="supported", supporting_source_ids=["review"]),
            ClaimCheck(claim="It is always perfect.", status="unsupported", supporting_source_ids=[]),
        ],
        evidence_sufficiency=check_evidence_sufficiency(
            [Evidence(source_id="review", text="Review claims against evidence.", score=0.9)],
            [
                ClaimCheck(claim="Review claims against evidence.", status="supported", supporting_source_ids=["review"]),
                ClaimCheck(claim="It is always perfect.", status="unsupported", supporting_source_ids=[]),
            ],
        ),
    )
    not_evaluated_trace = RagTrace(
        question="How should unevaluated answers be handled?",
        planned_queries=[
            RetrievalQuery(
                query="How should unevaluated answers be handled?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[],
        evidence=[],
        answer="Do not deliver without checks.",
        claim_checks=[],
        evidence_sufficiency=None,
    )
    no_evidence_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )
    approved_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )

    markdown = pipeline.build_human_review_priority_board_markdown(
        [
            {"label": "not evaluated", "trace": not_evaluated_trace},
            {"label": "approved", "trace": approved_trace},
            {"label": "missing | evidence", "trace": no_evidence_trace},
            {"label": "unsupported claim", "trace": unsupported_claim_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Human Review Priority Board",
            "",
            "| Priority | Trace | Risk signal | Reviewer action | Reason |",
            "| --- | --- | --- | --- | --- |",
            "| high | unsupported claim | unsupported_claims | revise_or_escalate_answer | 1 unsupported claims need review before delivery. |",
            "| medium | missing \\| evidence | no_evidence | retry_retrieval | No evidence was retrieved; rewrite or broaden retrieval. |",
            "| low | not evaluated | not_evaluated | run_evidence_evaluation | Run evidence sufficiency evaluation before delivery. |",
        ]
    )


def test_build_human_review_decision_log_markdown_records_approval_outcomes() -> None:
    from traceable_rag_agent import pipeline

    approved_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    blocked_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )

    markdown = pipeline.build_human_review_decision_log_markdown(
        [
            {"label": "approved answer", "trace": approved_trace, "reviewer_decision": "approved"},
            {"label": "missing | evidence", "trace": blocked_trace, "reviewer_decision": "retry_retrieval"},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Human Review Decision Log",
            "",
            "| Trace | Evidence status | Gate recommendation | Reviewer decision | Audit note |",
            "| --- | --- | --- | --- | --- |",
            "| approved answer | sufficient | approve_for_delivery | approved | Human reviewer accepted the citation-grounded answer. |",
            "| missing \\| evidence | insufficient | block_for_revision | retry_retrieval | No evidence was retrieved for this question. |",
        ]
    )


def test_build_human_review_decision_summary_counts_reviewer_outcomes() -> None:
    from traceable_rag_agent import pipeline

    approved_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    retry_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )
    escalate_trace = RagTrace(
        question="How should unsupported answers be reviewed?",
        planned_queries=[
            RetrievalQuery(
                query="How should unsupported answers be reviewed?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[
            RetrievalStep(
                query="How should unsupported answers be reviewed?",
                retrieved_source_ids=["review"],
                retrieved_chunk_ids=["review#0"],
            )
        ],
        evidence=[Evidence(source_id="review", text="Review claims against evidence.", score=0.9)],
        answer="Review claims against evidence. It is always perfect.",
        claim_checks=[
            ClaimCheck(claim="Review claims against evidence.", status="supported", supporting_source_ids=["review"]),
            ClaimCheck(claim="It is always perfect.", status="unsupported", supporting_source_ids=[]),
        ],
        evidence_sufficiency=check_evidence_sufficiency(
            [Evidence(source_id="review", text="Review claims against evidence.", score=0.9)],
            [
                ClaimCheck(claim="Review claims against evidence.", status="supported", supporting_source_ids=["review"]),
                ClaimCheck(claim="It is always perfect.", status="unsupported", supporting_source_ids=[]),
            ],
        ),
    )

    summary = pipeline.build_human_review_decision_summary(
        [
            {"label": "approved answer", "trace": approved_trace, "reviewer_decision": "approved"},
            {"label": "missing | evidence", "trace": retry_trace, "reviewer_decision": "retry_retrieval"},
            {"label": "unsupported claim", "trace": escalate_trace, "reviewer_decision": "escalated"},
        ]
    )

    assert summary == {
        "total_decisions": 3,
        "approved_decisions": 1,
        "revision_decisions": 2,
        "decision_counts": {"approved": 1, "retry_retrieval": 1, "escalated": 1},
        "blocked_trace_labels": ["missing | evidence", "unsupported claim"],
    }



def test_build_human_review_decision_summary_markdown_formats_review_outcomes_for_dashboard() -> None:
    summary = {
        "total_decisions": 4,
        "approved_decisions": 1,
        "revision_decisions": 3,
        "decision_counts": {"approved": 1, "retry_retrieval": 2, "escalated": 1},
        "blocked_trace_labels": ["missing | evidence", "unsupported claim", "stale citation"],
    }

    markdown = build_human_review_decision_summary_markdown(summary)

    assert markdown == "\n".join(
        [
            "## Human Review Decision Summary",
            "",
            "- Total decisions: 4",
            "- Approved decisions: 1",
            "- Revision decisions: 3",
            "- Blocked trace labels: missing \\| evidence, unsupported claim, stale citation",
            "",
            "| Reviewer decision | Count |",
            "| --- | ---: |",
            "| approved | 1 |",
            "| retry_retrieval | 2 |",
            "| escalated | 1 |",
        ]
    )


def test_build_interview_evidence_packet_markdown_summarizes_trace_for_demo() -> None:
    trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )

    markdown = build_interview_evidence_packet_markdown(trace)

    assert markdown == "\n".join(
        [
            "## Interview Evidence Packet",
            "",
            "**Question:** How does Agentic RAG check evidence sufficiency?",
            "**Answer status:** sufficient",
            "**Evidence sources:** sufficiency",
            "**Unsupported claims:** 0",
            "",
            "### Agentic RAG concepts demonstrated",
            "- Query planning: 1 retrieval query planned before answering.",
            "- Evidence grounding: 1 evidence item selected and cited from 1 source.",
            "- Faithfulness check: 1 supported claims and 0 unsupported claims.",
            "",
            "### Demo talking point",
            "This trace shows a RAG agent that does not just answer; it records retrieval, citations, and evidence sufficiency so a reviewer can verify why the answer is safe to deliver.",
        ]
    )



def test_build_interview_walkthrough_markdown_formats_trace_as_interview_story() -> None:
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence?",
        [
            Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
            Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
        ],
        top_k=1,
    )

    markdown = build_interview_walkthrough_markdown(trace)

    assert markdown == "\n".join(
        [
            "## Interview Walkthrough",
            "",
            "**User question:** How does Agentic RAG decompose questions and check evidence?",
            "**Delivery status:** ready_for_demo",
            "",
            "### What the agent did",
            "1. Planned 2 retrieval queries before answering.",
            "2. Ran 2 retrieval steps and selected 2 evidence items.",
            "3. Generated a citation-grounded answer and checked 2 claims.",
            "",
            "### Evidence and evaluation",
            "- Evidence sufficiency: sufficient — All answer claims are supported by retrieved evidence.",
            "- Supported claims: 2",
            "- Unsupported claims: 0",
            "- Cited sources: decomposition, sufficiency",
            "",
            "### Interview framing",
            "This is an Agentic RAG trace: the system plans retrieval, records evidence, checks faithfulness, and exposes whether the answer is safe to deliver.",
        ]
    )



def test_build_interview_readiness_scorecard_markdown_summarizes_demo_batch() -> None:
    ready_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    blocked_trace = answer_question(
        "How does Agentic RAG handle missing evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )

    markdown = build_interview_readiness_scorecard_markdown(
        [
            {"label": "ready demo", "trace": ready_trace},
            {"label": "blocked demo", "trace": blocked_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Interview Readiness Scorecard",
            "",
            "- Demo traces reviewed: 2",
            "- Ready for demo: 1",
            "- Needs review: 1",
            "- Evidence items reviewed: 1",
            "- Unsupported claims found: 1",
            "",
            "| Trace | Status | Evidence items | Unsupported claims | Interview angle |",
            "| --- | --- | ---: | ---: | --- |",
            "| ready demo | ready_for_demo | 1 | 0 | Show citation-grounded answer delivery. |",
            "| blocked demo | needs_review | 0 | 1 | Show evidence gate blocking weak answers. |",
        ]
    )



def test_build_interview_followup_questions_markdown_prepares_trace_specific_answers() -> None:
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence?",
        [
            Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
            Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
        ],
        top_k=1,
    )

    markdown = build_interview_followup_questions_markdown(trace)

    assert markdown == "\n".join(
        [
            "## Interview Follow-up Questions",
            "",
            "| Question | Evidence-backed answer angle |",
            "| --- | --- |",
            "| How is this different from a simple ChatPDF app? | It plans 2 retrieval queries, records 2 retrieval steps, and checks evidence before delivery. |",
            "| How do you know the answer is grounded? | The trace selected 2 evidence items, cited decomposition, sufficiency, and marked sufficiency as sufficient. |",
            "| What happens when evidence is weak? | The same trace schema exposes sufficiency failures so the agent can retry retrieval or escalate to human review. |",
        ]
    )



def test_build_interview_concept_map_markdown_summarizes_agentic_rag_capabilities() -> None:
    ready_trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        [
            Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
            Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
        ],
        top_k=1,
    )
    blocked_trace = RagTrace(
        question="How should unsupported answers be handled?",
        planned_queries=[
            RetrievalQuery(
                query="How should unsupported answers be handled?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[
            RetrievalStep(
                query="How should unsupported answers be handled?",
                retrieved_source_ids=["review"],
                retrieved_chunk_ids=["review#0"],
            )
        ],
        evidence=[Evidence(source_id="review", text="Review answers against retrieved evidence.", score=0.9)],
        answer="Review answers against retrieved evidence. The answer is always perfect.",
        claim_checks=[
            ClaimCheck(
                claim="Review answers against retrieved evidence.", status="supported", supporting_source_ids=["review"]
            ),
            ClaimCheck(claim="The answer is always perfect.", status="unsupported", supporting_source_ids=[]),
        ],
        evidence_sufficiency=EvidenceSufficiency(
            status="insufficient",
            reason="Some answer claims are not supported by retrieved evidence.",
            evidence_count=1,
            supported_claim_count=1,
            unsupported_claim_count=1,
        ),
    )

    markdown = build_interview_concept_map_markdown(
        [
            {"label": "multi-query ready", "trace": ready_trace},
            {"label": "unsupported claim", "trace": blocked_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Agentic RAG Interview Concept Map",
            "",
            "| Concept | Concrete proof in this project | Interview explanation |",
            "| --- | --- | --- |",
            "| Query decomposition | 3 planned queries across 2 traces | The agent plans focused searches before answering instead of sending one broad prompt. |",
            "| Evidence grounding | 3 retrieved evidence items | Answers are built from retrieved snippets and source citations, not unsupported text generation. |",
            "| Faithfulness evaluation | 1 unsupported claims caught | The system checks whether answer claims are supported before delivery. |",
            "| Answer safety gate | 1 ready traces / 1 blocked traces | Good answers can be delivered, while weak or risky answers are routed to retry or human review. |",
        ]
    )



def test_build_interview_demo_script_markdown_sequences_happy_path_and_risk_trace() -> None:
    ready_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    blocked_trace = RagTrace(
        question="How should a hallucinated answer be handled?",
        planned_queries=[
            RetrievalQuery(
                query="How should a hallucinated answer be handled?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[
            RetrievalStep(
                query="How should a hallucinated answer be handled?",
                retrieved_source_ids=[],
                retrieved_chunk_ids=[],
            )
        ],
        evidence=[],
        answer="The system should not answer without evidence.",
        claim_checks=[
            ClaimCheck(
                claim="The system should not answer without evidence.",
                status="unsupported",
                supporting_source_ids=[],
            )
        ],
        evidence_sufficiency=EvidenceSufficiency(
            status="insufficient",
            reason="No evidence was retrieved for this question.",
            evidence_count=0,
            supported_claim_count=0,
            unsupported_claim_count=1,
        ),
    )

    markdown = build_interview_demo_script_markdown(
        [
            {"label": "grounded answer", "trace": ready_trace},
            {"label": "blocked hallucination", "trace": blocked_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Interview Demo Script",
            "",
            "- Demo goal: prove this is an Agentic RAG system that plans retrieval, grounds answers in evidence, and blocks weak answers before delivery.",
            "- Recommended flow: start with a ready trace, then show a blocked or risky trace to demonstrate failure handling.",
            "",
            "| Step | Trace | Demo status | What to show | Interview talking point |",
            "| ---: | --- | --- | --- | --- |",
            "| 1 | grounded answer | ready_for_demo | 1 planned queries, 1 retrieval steps, 1 evidence items, 0 unsupported claims | Shows citation-grounded answer delivery with traceable evidence. |",
            "| 2 | blocked hallucination | needs_review | 1 planned queries, 1 retrieval steps, 0 evidence items, 1 unsupported claims | Shows the evidence gate catching weak answers before users see them. |",
        ]
    )



def test_build_interview_objection_handling_markdown_prepares_trace_backed_responses() -> None:
    ready_trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence?",
        [
            Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
            Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
        ],
        top_k=1,
    )
    blocked_trace = RagTrace(
        question="How should unsupported answers be handled?",
        planned_queries=[
            RetrievalQuery(
                query="How should unsupported answers be handled?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[
            RetrievalStep(
                query="How should unsupported answers be handled?",
                retrieved_source_ids=["review"],
                retrieved_chunk_ids=["review#0"],
            )
        ],
        evidence=[Evidence(source_id="review", text="Review answers against retrieved evidence.", score=0.9)],
        answer="Review answers against retrieved evidence. The answer is always perfect.",
        claim_checks=[
            ClaimCheck(
                claim="Review answers against retrieved evidence.", status="supported", supporting_source_ids=["review"]
            ),
            ClaimCheck(claim="The answer is always perfect.", status="unsupported", supporting_source_ids=[]),
        ],
        evidence_sufficiency=EvidenceSufficiency(
            status="insufficient",
            reason="Some answer claims are not supported by retrieved evidence.",
            evidence_count=1,
            supported_claim_count=1,
            unsupported_claim_count=1,
        ),
    )

    markdown = build_interview_objection_handling_markdown(
        [
            {"label": "ready multi-query", "trace": ready_trace},
            {"label": "blocked unsupported", "trace": blocked_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Interview Objection Handling",
            "",
            "| Interviewer concern | Trace-backed response |",
            "| --- | --- |",
            "| Is this just keyword search or ChatPDF? | The demo batch includes 3 planned retrieval queries across 2 traces, so the agent exposes planning before answering instead of hiding one broad prompt. |",
            "| How do you know answers are faithful? | The traces checked 4 answer claims and caught 1 unsupported claims before delivery. |",
            "| What happens when evidence is weak? | The answer gate marks 1 traces ready and 1 traces needing review, so weak answers are routed to retry or human review instead of being shown as final. |",
            "| Can an interviewer inspect the reasoning path? | The batch records 3 retrieval steps and 3 evidence items, giving a concrete audit trail from query plan to cited evidence. |",
        ]
    )



def test_build_human_review_revision_queue_markdown_lists_only_non_approved_decisions() -> None:
    approved_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    retry_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )
    escalate_trace = RagTrace(
        question="How should unsupported answers be reviewed?",
        planned_queries=[
            RetrievalQuery(
                query="How should unsupported answers be reviewed?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[
            RetrievalStep(
                query="How should unsupported answers be reviewed?",
                retrieved_source_ids=["review"],
                retrieved_chunk_ids=["review#0"],
            )
        ],
        evidence=[Evidence(source_id="review", text="Review claims against evidence.", score=0.9)],
        answer="Review claims against evidence. It is always perfect.",
        claim_checks=[
            ClaimCheck(claim="Review claims against evidence.", status="supported", supporting_source_ids=["review"]),
            ClaimCheck(claim="It is always perfect.", status="unsupported", supporting_source_ids=[]),
        ],
        evidence_sufficiency=check_evidence_sufficiency(
            [Evidence(source_id="review", text="Review claims against evidence.", score=0.9)],
            [
                ClaimCheck(claim="Review claims against evidence.", status="supported", supporting_source_ids=["review"]),
                ClaimCheck(claim="It is always perfect.", status="unsupported", supporting_source_ids=[]),
            ],
        ),
    )

    markdown = build_human_review_revision_queue_markdown(
        [
            {"label": "approved answer", "trace": approved_trace, "reviewer_decision": "approved"},
            {"label": "missing | evidence", "trace": retry_trace, "reviewer_decision": "retry_retrieval"},
            {"label": "unsupported claim", "trace": escalate_trace, "reviewer_decision": "escalated"},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Human Review Revision Queue",
            "",
            "| Trace | Reviewer decision | Evidence status | Follow-up action | Reason |",
            "| --- | --- | --- | --- | --- |",
            "| missing \\| evidence | retry_retrieval | insufficient | rerun_retrieval_and_evaluation | No evidence was retrieved for this question. |",
            "| unsupported claim | escalated | insufficient | human_escalation_review | Some answer claims are not supported by retrieved evidence. |",
        ]
    )


def test_build_human_review_workload_summary_counts_review_outcomes() -> None:
    blocked_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )
    approved_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    not_evaluated_trace = RagTrace(
        question="How should unevaluated answers be handled?",
        planned_queries=[
            RetrievalQuery(
                query="How should unevaluated answers be handled?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[],
        evidence=[],
        answer="Do not deliver without checks.",
        claim_checks=[],
        evidence_sufficiency=None,
    )

    summary = build_human_review_workload_summary(
        [
            {"label": "missing | evidence", "trace": blocked_trace},
            {"label": "approved", "trace": approved_trace},
            {"label": "not evaluated", "trace": not_evaluated_trace},
        ]
    )

    assert summary == {
        "total_traces": 3,
        "approved_traces": 1,
        "blocked_traces": 2,
        "not_evaluated_traces": 1,
        "insufficient_evidence_traces": 1,
        "total_evidence_items_needing_review": 0,
        "total_unsupported_claims": 1,
        "blocked_trace_labels": ["missing | evidence", "not evaluated"],
    }



def test_build_human_review_workload_markdown_formats_dashboard_summary() -> None:
    summary = {
        "total_traces": 4,
        "approved_traces": 1,
        "blocked_traces": 3,
        "not_evaluated_traces": 1,
        "insufficient_evidence_traces": 2,
        "total_evidence_items_needing_review": 5,
        "total_unsupported_claims": 2,
        "blocked_trace_labels": ["missing | evidence", "not evaluated", "unsupported claim"],
    }

    markdown = build_human_review_workload_markdown(summary)

    assert markdown == "\n".join(
        [
            "## Human Review Workload",
            "",
            "- Total traces: 4",
            "- Approved traces: 1",
            "- Blocked traces: 3",
            "- Not evaluated traces: 1",
            "- Insufficient-evidence traces: 2",
            "- Evidence items needing review: 5",
            "- Unsupported claims needing review: 2",
            "- Blocked trace labels: missing \\| evidence, not evaluated, unsupported claim",
        ]
    )



def test_build_human_review_escalation_brief_markdown_summarizes_blocked_traces_for_reviewers() -> None:
    blocked_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )
    not_evaluated_trace = RagTrace(
        question="How should unevaluated answers be handled?",
        planned_queries=[
            RetrievalQuery(
                query="How should unevaluated answers be handled?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[],
        evidence=[],
        answer="Do not deliver without checks.",
        claim_checks=[],
        evidence_sufficiency=None,
    )
    approved_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )

    markdown = build_human_review_escalation_brief_markdown(
        [
            {"label": "missing | evidence", "trace": blocked_trace},
            {"label": "not evaluated", "trace": not_evaluated_trace},
            {"label": "approved", "trace": approved_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Human Review Escalation Brief",
            "",
            "- Blocked traces: 2",
            "- Total evidence items needing review: 0",
            "- Total unsupported claims: 1",
            "",
            "| Trace | Question | Trigger | Evidence items | Unsupported claims | Reviewer action |",
            "| --- | --- | --- | ---: | ---: | --- |",
            "| missing \\| evidence | How does Agentic RAG handle missing \\| evidence? | insufficient_evidence | 0 | 1 | inspect_retrieval_and_revise_answer |",
            "| not evaluated | How should unevaluated answers be handled? | not_evaluated | 0 | 0 | run_evidence_evaluation |",
        ]
    )


def test_build_evidence_table_exposes_ranked_snippets_for_dashboard() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )

    table = build_evidence_table(trace)

    assert table == [
        {
            "rank": 1,
            "source_id": "decomposition",
            "chunk_id": "decomposition#0",
            "score": trace.evidence[0].score,
            "snippet": "Agentic RAG decomposes questions into focused retrieval queries.",
        },
        {
            "rank": 2,
            "source_id": "sufficiency",
            "chunk_id": "sufficiency#0",
            "score": trace.evidence[1].score,
            "snippet": "Agentic RAG checks evidence sufficiency before final synthesis.",
        },
    ]


def test_answer_question_marks_trace_insufficient_when_no_evidence_is_retrieved() -> None:
    trace = answer_question(
        "What does Agentic RAG check before answering?",
        [Document(source_id="unrelated", text="Workflow agents call tools and inspect results.")],
        top_k=1,
    )

    assert trace.evidence == []
    assert trace.evidence_sufficiency.status == "insufficient"
    assert trace.evidence_sufficiency.reason == "No evidence was retrieved for this question."


def test_answer_question_retries_with_rewritten_query_when_initial_retrieval_is_weak() -> None:
    documents = [
        Document(
            source_id="faithfulness",
            text="Unsupported claim checks flag claims without evidence before final answers.",
        )
    ]

    trace = answer_question("How are hallucinations prevented?", documents, top_k=1)

    assert [query.query for query in trace.planned_queries] == [
        "How are hallucinations prevented?",
        "unsupported claims evidence",
    ]
    assert trace.planned_queries[1].reason == "Retry retrieval with a rewritten query after weak evidence."
    assert [step.query for step in trace.retrieval_steps] == [
        "How are hallucinations prevented?",
        "unsupported claims evidence",
    ]
    assert trace.retrieval_steps[0].retrieved_source_ids == []
    assert trace.retrieval_steps[1].retrieved_source_ids == ["faithfulness"]
    assert [item.source_id for item in trace.evidence] == ["faithfulness"]
    assert trace.evidence_sufficiency.status == "sufficient"


def test_save_trace_json_persists_run_for_later_observability(tmp_path) -> None:
    trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    output_path = tmp_path / "trace.json"

    saved_path = save_trace_json(trace, output_path)

    assert saved_path == output_path
    saved_trace = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_trace["question"] == "How does Agentic RAG check evidence sufficiency?"
    assert saved_trace["retrieval_steps"][0]["retrieved_chunk_ids"] == ["sufficiency#0"]
    assert saved_trace["evidence_sufficiency"]["status"] == "sufficient"


def test_build_trace_report_markdown_formats_one_run_for_observability_dashboard() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )

    markdown = build_trace_report_markdown(trace)

    assert markdown == "\n".join(
        [
            "## Trace Report",
            "",
            "- Question: How does Agentic RAG decompose questions and check evidence sufficiency?",
            "- Evidence sufficiency: sufficient — All answer claims are supported by retrieved evidence.",
            "- Planned queries: 2",
            "- Retrieved evidence items: 2",
            "",
            "### Retrieval steps",
            "| Step | Query | Sources | Chunks |",
            "| ---: | --- | --- | --- |",
            "| 1 | How does Agentic RAG decompose questions? | decomposition | decomposition#0 |",
            "| 2 | How does Agentic RAG check evidence sufficiency? | sufficiency | sufficiency#0 |",
            "",
            "### Evidence",
            "| Rank | Source | Chunk | Score | Snippet |",
            "| ---: | --- | --- | ---: | --- |",
            f"| 1 | decomposition | decomposition#0 | {trace.evidence[0].score:.2f} | Agentic RAG decomposes questions into focused retrieval queries. |",
            f"| 2 | sufficiency | sufficiency#0 | {trace.evidence[1].score:.2f} | Agentic RAG checks evidence sufficiency before final synthesis. |",
        ]
    )


def test_build_trace_report_markdown_escapes_table_pipes_in_trace_fields() -> None:
    trace = answer_question(
        "How does Agentic RAG compare query planning | retrieval?",
        [
            Document(
                source_id="demo|source",
                text="Agentic RAG compares query planning | retrieval before synthesis.",
            )
        ],
        top_k=1,
    )

    markdown = build_trace_report_markdown(trace)

    assert "| 1 | How does Agentic RAG compare query planning \\| retrieval? | demo\\|source | demo\\|source#0 |" in markdown
    assert f"| 1 | demo\\|source | demo\\|source#0 | {trace.evidence[0].score:.2f} | Agentic RAG compares query planning \\| retrieval before synthesis. |" in markdown


def test_build_trace_status_summary_returns_dashboard_card_metrics() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )

    summary = build_trace_status_summary(trace)

    assert summary == {
        "sufficiency_status": "sufficient",
        "planned_query_count": 2,
        "retrieval_step_count": 2,
        "evidence_count": 2,
        "supported_claim_count": 2,
        "unsupported_claim_count": 0,
        "cited_source_count": 2,
        "total_retrieval_latency_ms": summary["total_retrieval_latency_ms"],
    }
    assert summary["total_retrieval_latency_ms"] >= 0


def test_build_trace_timeline_events_includes_answer_synthesis_before_evaluation() -> None:
    trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )

    events = build_trace_timeline_events(trace)

    assert [event["event_type"] for event in events] == [
        "plan_query",
        "retrieve_evidence",
        "synthesize_answer",
        "evaluate_sufficiency",
    ]
    assert events[2] == {
        "order": 3,
        "event_type": "synthesize_answer",
        "title": "Synthesized citation-grounded answer",
        "detail": "1 answer claims checked against 1 evidence items.",
        "source_ids": ["sufficiency"],
        "chunk_ids": ["sufficiency#0"],
    }


def test_build_trace_timeline_events_returns_ordered_observability_steps() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )

    events = build_trace_timeline_events(trace)

    assert events == [
        {
            "order": 1,
            "event_type": "plan_query",
            "title": "Planned query 1",
            "detail": "How does Agentic RAG decompose questions?",
            "source_ids": [],
            "chunk_ids": [],
        },
        {
            "order": 2,
            "event_type": "retrieve_evidence",
            "title": "Retrieved evidence for query 1",
            "detail": "How does Agentic RAG decompose questions?",
            "source_ids": ["decomposition"],
            "chunk_ids": ["decomposition#0"],
        },
        {
            "order": 3,
            "event_type": "plan_query",
            "title": "Planned query 2",
            "detail": "How does Agentic RAG check evidence sufficiency?",
            "source_ids": [],
            "chunk_ids": [],
        },
        {
            "order": 4,
            "event_type": "retrieve_evidence",
            "title": "Retrieved evidence for query 2",
            "detail": "How does Agentic RAG check evidence sufficiency?",
            "source_ids": ["sufficiency"],
            "chunk_ids": ["sufficiency#0"],
        },
        {
            "order": 5,
            "event_type": "synthesize_answer",
            "title": "Synthesized citation-grounded answer",
            "detail": "2 answer claims checked against 2 evidence items.",
            "source_ids": ["decomposition", "sufficiency"],
            "chunk_ids": ["decomposition#0", "sufficiency#0"],
        },
        {
            "order": 6,
            "event_type": "evaluate_sufficiency",
            "title": "Evaluated evidence sufficiency",
            "detail": "sufficient: All answer claims are supported by retrieved evidence.",
            "source_ids": ["decomposition", "sufficiency"],
            "chunk_ids": ["decomposition#0", "sufficiency#0"],
        },
    ]


def test_build_trace_timeline_markdown_formats_events_for_demo_rendering() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )
    events = build_trace_timeline_events(trace)

    markdown = build_trace_timeline_markdown(events)

    assert markdown == "\n".join(
        [
            "## Trace Timeline",
            "",
            "| Order | Event | Detail | Sources | Chunks |",
            "| ---: | --- | --- | --- | --- |",
            "| 1 | Planned query 1 | How does Agentic RAG decompose questions? | none | none |",
            "| 2 | Retrieved evidence for query 1 | How does Agentic RAG decompose questions? | decomposition | decomposition#0 |",
            "| 3 | Planned query 2 | How does Agentic RAG check evidence sufficiency? | none | none |",
            "| 4 | Retrieved evidence for query 2 | How does Agentic RAG check evidence sufficiency? | sufficiency | sufficiency#0 |",
            "| 5 | Synthesized citation-grounded answer | 2 answer claims checked against 2 evidence items. | decomposition, sufficiency | decomposition#0, sufficiency#0 |",
            "| 6 | Evaluated evidence sufficiency | sufficient: All answer claims are supported by retrieved evidence. | decomposition, sufficiency | decomposition#0, sufficiency#0 |",
        ]
    )


def test_build_trace_interview_summary_markdown_highlights_agent_concepts() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )

    markdown = build_trace_interview_summary_markdown(
        trace,
        expected_source_ids=["decomposition", "sufficiency"],
    )

    assert markdown == "\n".join(
        [
            "## Interview Trace Summary",
            "",
            "- Agent concept: query decomposition + citation-grounded synthesis + faithfulness evaluation",
            "- Question: How does Agentic RAG decompose questions and check evidence sufficiency?",
            "- Planned retrieval queries: 2",
            "- Retrieved evidence items: 2",
            "- Retrieval coverage: 100%",
            "- Citation support: 100%",
            "- Unsupported claims: 0",
            "- Evidence sufficiency: sufficient — All answer claims are supported by retrieved evidence.",
        ]
    )


def test_build_citation_evidence_map_links_answer_citations_to_retrieved_evidence() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )

    citation_map = build_citation_evidence_map(trace)

    assert citation_map == [
        {
            "citation": "decomposition",
            "is_retrieved": True,
            "evidence_rank": 1,
            "chunk_id": "decomposition#0",
            "snippet": "Agentic RAG decomposes questions into focused retrieval queries.",
            "supporting_claim_count": 1,
        },
        {
            "citation": "sufficiency",
            "is_retrieved": True,
            "evidence_rank": 2,
            "chunk_id": "sufficiency#0",
            "snippet": "Agentic RAG checks evidence sufficiency before final synthesis.",
            "supporting_claim_count": 1,
        },
    ]


def test_build_citation_evidence_map_markdown_formats_citation_links_for_demo_review() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )
    citation_map = build_citation_evidence_map(trace)

    markdown = build_citation_evidence_map_markdown(citation_map)

    assert markdown == "\n".join(
        [
            "## Citation Evidence Map",
            "",
            "| Citation | Retrieved? | Evidence rank | Chunk | Supporting claims | Snippet |",
            "| --- | --- | ---: | --- | ---: | --- |",
            "| decomposition | yes | 1 | decomposition#0 | 1 | Agentic RAG decomposes questions into focused retrieval queries. |",
            "| sufficiency | yes | 2 | sufficiency#0 | 1 | Agentic RAG checks evidence sufficiency before final synthesis. |",
        ]
    )


def test_measure_retrieval_coverage_reports_found_missing_and_ratio() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions?",
        documents,
        top_k=1,
    )

    coverage = measure_retrieval_coverage(
        trace,
        expected_source_ids=["decomposition", "sufficiency"],
    )

    assert coverage == {
        "expected_source_ids": ["decomposition", "sufficiency"],
        "retrieved_source_ids": ["decomposition"],
        "found_source_ids": ["decomposition"],
        "missing_source_ids": ["sufficiency"],
        "coverage_ratio": 0.5,
    }


def test_measure_citation_support_reports_unsupported_answer_citations() -> None:
    trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    trace.answer = f"{trace.answer} Unsupported extra claim [missing-source]"

    support = measure_citation_support(trace)

    assert support == {
        "cited_source_ids": ["sufficiency", "missing-source"],
        "evidence_source_ids": ["sufficiency"],
        "supported_cited_source_ids": ["sufficiency"],
        "unsupported_cited_source_ids": ["missing-source"],
        "citation_support_ratio": 0.5,
    }


def test_run_retrieval_coverage_benchmark_summarizes_question_level_results() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    benchmark = [
        BenchmarkQuestion(
            question="How does Agentic RAG decompose questions?",
            expected_source_ids=["decomposition"],
        ),
        BenchmarkQuestion(
            question="How does Agentic RAG check evidence sufficiency?",
            expected_source_ids=["sufficiency", "decomposition"],
        ),
    ]

    report = run_retrieval_coverage_benchmark(benchmark, documents, top_k=1)

    assert report == {
        "question_count": 2,
        "average_coverage_ratio": 0.75,
        "results": [
            {
                "question": "How does Agentic RAG decompose questions?",
                "expected_source_ids": ["decomposition"],
                "retrieved_source_ids": ["decomposition"],
                "found_source_ids": ["decomposition"],
                "missing_source_ids": [],
                "coverage_ratio": 1.0,
            },
            {
                "question": "How does Agentic RAG check evidence sufficiency?",
                "expected_source_ids": ["sufficiency", "decomposition"],
                "retrieved_source_ids": ["sufficiency"],
                "found_source_ids": ["sufficiency"],
                "missing_source_ids": ["decomposition"],
                "coverage_ratio": 0.5,
            },
        ],
    }


def test_run_rag_quality_benchmark_combines_retrieval_and_citation_metrics() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    benchmark = [
        BenchmarkQuestion(
            question="How does Agentic RAG decompose questions?",
            expected_source_ids=["decomposition"],
        ),
        BenchmarkQuestion(
            question="How does Agentic RAG check evidence sufficiency?",
            expected_source_ids=["sufficiency", "decomposition"],
        ),
    ]

    report = run_rag_quality_benchmark(benchmark, documents, top_k=1)

    assert report["question_count"] == 2
    assert report["average_retrieval_coverage_ratio"] == 0.75
    assert report["average_citation_support_ratio"] == 1.0
    assert report["results"][0]["citation_support"]["supported_cited_source_ids"] == ["decomposition"]
    assert report["results"][1]["retrieval_coverage"]["missing_source_ids"] == ["decomposition"]


def test_diagnose_rag_quality_failures_labels_question_level_failure_modes() -> None:
    quality_report = {
        "results": [
            {
                "question": "How does Agentic RAG decompose questions?",
                "retrieval_coverage": {
                    "missing_source_ids": [],
                    "coverage_ratio": 1.0,
                },
                "citation_support": {
                    "unsupported_cited_source_ids": [],
                    "citation_support_ratio": 1.0,
                },
            },
            {
                "question": "How does Agentic RAG check evidence sufficiency?",
                "retrieval_coverage": {
                    "missing_source_ids": ["decomposition"],
                    "coverage_ratio": 0.5,
                },
                "citation_support": {
                    "unsupported_cited_source_ids": [],
                    "citation_support_ratio": 1.0,
                },
            },
            {
                "question": "How does Agentic RAG cite evidence?",
                "retrieval_coverage": {
                    "missing_source_ids": [],
                    "coverage_ratio": 1.0,
                },
                "citation_support": {
                    "unsupported_cited_source_ids": ["missing-source"],
                    "citation_support_ratio": 0.5,
                },
            },
        ]
    }

    diagnoses = diagnose_rag_quality_failures(quality_report)

    assert diagnoses == [
        {
            "question": "How does Agentic RAG decompose questions?",
            "failure_mode": "pass",
            "reason": "Retrieval covered expected sources and all citations are supported.",
        },
        {
            "question": "How does Agentic RAG check evidence sufficiency?",
            "failure_mode": "retrieval_gap",
            "reason": "Missing expected sources: decomposition.",
        },
        {
            "question": "How does Agentic RAG cite evidence?",
            "failure_mode": "unsupported_citation",
            "reason": "Answer cited sources that were not retrieved: missing-source.",
        },
    ]


def test_build_failure_diagnosis_markdown_formats_question_level_debug_table() -> None:
    diagnoses = [
        {
            "question": "How does Agentic RAG decompose questions?",
            "failure_mode": "pass",
            "reason": "Retrieval covered expected sources and all citations are supported.",
        },
        {
            "question": "How does Agentic RAG check evidence | citations?",
            "failure_mode": "retrieval_gap",
            "reason": "Missing expected sources: source-b | source-c.",
        },
    ]

    markdown = build_failure_diagnosis_markdown(diagnoses)

    assert markdown == "\n".join(
        [
            "## Failure Diagnosis Report",
            "",
            "| Question | Failure mode | Reason |",
            "| --- | --- | --- |",
            "| How does Agentic RAG decompose questions? | pass | Retrieval covered expected sources and all citations are supported. |",
            "| How does Agentic RAG check evidence \\| citations? | retrieval_gap | Missing expected sources: source-b \\| source-c. |",
        ]
    )


def test_build_retrieval_gap_report_markdown_lists_missing_sources_for_debugging() -> None:
    quality_report = {
        "results": [
            {
                "question": "How does Agentic RAG decompose questions?",
                "retrieval_coverage": {
                    "expected_source_ids": ["decomposition"],
                    "retrieved_source_ids": ["decomposition"],
                    "found_source_ids": ["decomposition"],
                    "missing_source_ids": [],
                    "coverage_ratio": 1.0,
                },
            },
            {
                "question": "How does Agentic RAG check evidence | citations?",
                "retrieval_coverage": {
                    "expected_source_ids": ["evidence", "citations"],
                    "retrieved_source_ids": ["evidence"],
                    "found_source_ids": ["evidence"],
                    "missing_source_ids": ["citations"],
                    "coverage_ratio": 0.5,
                },
            },
        ]
    }

    markdown = build_retrieval_gap_report_markdown(quality_report)

    assert markdown == "\n".join(
        [
            "## Retrieval Gap Report",
            "",
            "| Question | Coverage | Missing sources | Retrieved sources |",
            "| --- | ---: | --- | --- |",
            "| How does Agentic RAG check evidence \\| citations? | 50% | citations | evidence |",
        ]
    )


def test_summarize_failure_diagnoses_counts_modes_for_dashboard() -> None:
    diagnoses = [
        {
            "question": "Question A",
            "failure_mode": "pass",
            "reason": "Retrieval covered expected sources and all citations are supported.",
        },
        {
            "question": "Question B",
            "failure_mode": "retrieval_gap",
            "reason": "Missing expected sources: source-b.",
        },
        {
            "question": "Question C",
            "failure_mode": "retrieval_gap",
            "reason": "Missing expected sources: source-c.",
        },
        {
            "question": "Question D",
            "failure_mode": "unsupported_citation",
            "reason": "Answer cited sources that were not retrieved: missing-source.",
        },
    ]

    summary = summarize_failure_diagnoses(diagnoses)

    assert summary == {
        "total_questions": 4,
        "passed_questions": 1,
        "failed_questions": 3,
        "failure_mode_counts": {
            "pass": 1,
            "retrieval_gap": 2,
            "unsupported_citation": 1,
        },
        "top_failure_mode": "retrieval_gap",
    }


def test_recommend_next_evaluation_action_turns_failure_summary_into_actionable_guidance() -> None:
    summary = {
        "total_questions": 4,
        "passed_questions": 1,
        "failed_questions": 3,
        "failure_mode_counts": {
            "pass": 1,
            "retrieval_gap": 2,
            "unsupported_citation": 1,
        },
        "top_failure_mode": "retrieval_gap",
    }

    action = recommend_next_evaluation_action(summary)

    assert action == {
        "priority": "fix_retrieval",
        "message": "Top failure mode is retrieval_gap across 2 of 4 questions; improve query rewriting, retrieval recall, or reranking before changing answer synthesis.",
    }


def test_build_evaluation_metric_cards_summarizes_trace_quality_for_dashboard() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )

    cards = build_evaluation_metric_cards(trace, expected_source_ids=["decomposition", "sufficiency"])

    assert cards == [
        {
            "label": "Retrieval coverage",
            "value": "100%",
            "status": "pass",
            "detail": "Found 2 of 2 expected sources.",
        },
        {
            "label": "Citation support",
            "value": "100%",
            "status": "pass",
            "detail": "Supported 2 of 2 cited sources.",
        },
        {
            "label": "Unsupported claims",
            "value": "0",
            "status": "pass",
            "detail": "0 unsupported claims out of 2 checked claims.",
        },
    ]



def test_build_latency_metric_cards_summarizes_retrieval_latency_for_dashboard() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )
    trace.retrieval_steps[0].latency_ms = 12.345
    trace.retrieval_steps[1].latency_ms = 7.655

    cards = build_latency_metric_cards(trace)

    assert cards == [
        {
            "label": "Total retrieval latency",
            "value": "20.00 ms",
            "status": "pass",
            "detail": "2 retrieval steps completed without recorded errors.",
        },
        {
            "label": "Average retrieval latency",
            "value": "10.00 ms",
            "status": "pass",
            "detail": "Average latency across 2 retrieval steps.",
        },
    ]


def test_build_latency_metric_cards_flags_slow_retrieval_threshold() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )
    trace.retrieval_steps[0].latency_ms = 80.0
    trace.retrieval_steps[1].latency_ms = 90.0

    cards = build_latency_metric_cards(trace, slow_threshold_ms=100.0)

    assert cards[0] == {
        "label": "Total retrieval latency",
        "value": "170.00 ms",
        "status": "warn",
        "detail": "2 retrieval steps exceeded the 100.00 ms slow threshold.",
    }
    assert cards[1]["status"] == "warn"


def test_build_retrieval_error_metric_cards_summarizes_failed_retrieval_steps() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )
    trace.retrieval_steps[1].error = "timeout while retrieving evidence"

    cards = build_retrieval_error_metric_cards(trace)

    assert cards == [
        {
            "label": "Retrieval errors",
            "value": "1",
            "status": "fail",
            "detail": "1 of 2 retrieval steps recorded errors.",
        },
        {
            "label": "First retrieval error",
            "value": "query 2",
            "status": "fail",
            "detail": "timeout while retrieving evidence",
        },
    ]


def test_build_trace_health_metric_cards_summarizes_run_readiness_for_dashboard() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )

    cards = build_trace_health_metric_cards(trace)

    assert cards == [
        {
            "label": "Evidence sufficiency",
            "value": "sufficient",
            "status": "pass",
            "detail": "All answer claims are supported by retrieved evidence.",
        },
        {
            "label": "Trace completeness",
            "value": "2/2 steps",
            "status": "pass",
            "detail": "2 planned queries produced 2 retrieval steps and 2 evidence items.",
        },
        {
            "label": "Claim support",
            "value": "2 supported / 0 unsupported",
            "status": "pass",
            "detail": "Checked 2 answer claims against retrieved evidence.",
        },
    ]


def test_build_observability_dashboard_sections_groups_health_evaluation_latency_and_errors() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )
    trace.retrieval_steps[0].latency_ms = 5.0
    trace.retrieval_steps[1].latency_ms = 15.0

    sections = build_observability_dashboard_sections(
        trace,
        expected_source_ids=["decomposition", "sufficiency"],
        slow_threshold_ms=100.0,
    )

    assert [section["title"] for section in sections] == [
        "Trace health",
        "Evaluation quality",
        "Retrieval latency",
        "Retrieval errors",
    ]
    assert sections[0]["cards"][0]["label"] == "Evidence sufficiency"
    assert sections[1]["cards"][0]["value"] == "100%"
    assert sections[2]["cards"][0] == {
        "label": "Total retrieval latency",
        "value": "20.00 ms",
        "status": "pass",
        "detail": "2 retrieval steps completed without recorded errors.",
    }
    assert sections[3]["cards"][0] == {
        "label": "Retrieval errors",
        "value": "0",
        "status": "pass",
        "detail": "0 of 2 retrieval steps recorded errors.",
    }


def test_build_observability_dashboard_markdown_formats_sections_for_demo_page() -> None:
    documents = [
        Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
        Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
    ]
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        documents,
        top_k=1,
    )
    trace.retrieval_steps[0].latency_ms = 5.0
    trace.retrieval_steps[1].latency_ms = 15.0
    sections = build_observability_dashboard_sections(
        trace,
        expected_source_ids=["decomposition", "sufficiency"],
        slow_threshold_ms=100.0,
    )

    markdown = build_observability_dashboard_markdown(sections)

    assert markdown == "\n".join(
        [
            "## Observability Dashboard",
            "",
            "### Trace health",
            "| Metric | Value | Status | Detail |",
            "| --- | --- | --- | --- |",
            "| Evidence sufficiency | sufficient | pass | All answer claims are supported by retrieved evidence. |",
            "| Trace completeness | 2/2 steps | pass | 2 planned queries produced 2 retrieval steps and 2 evidence items. |",
            "| Claim support | 2 supported / 0 unsupported | pass | Checked 2 answer claims against retrieved evidence. |",
            "",
            "### Evaluation quality",
            "| Metric | Value | Status | Detail |",
            "| --- | --- | --- | --- |",
            "| Retrieval coverage | 100% | pass | Found 2 of 2 expected sources. |",
            "| Citation support | 100% | pass | Supported 2 of 2 cited sources. |",
            "| Unsupported claims | 0 | pass | 0 unsupported claims out of 2 checked claims. |",
            "",
            "### Retrieval latency",
            "| Metric | Value | Status | Detail |",
            "| --- | --- | --- | --- |",
            "| Total retrieval latency | 20.00 ms | pass | 2 retrieval steps completed without recorded errors. |",
            "| Average retrieval latency | 10.00 ms | pass | Average latency across 2 retrieval steps. |",
            "",
            "### Retrieval errors",
            "| Metric | Value | Status | Detail |",
            "| --- | --- | --- | --- |",
            "| Retrieval errors | 0 | pass | 0 of 2 retrieval steps recorded errors. |",
        ]
    )


def test_build_observability_dashboard_markdown_escapes_table_pipes_in_card_text() -> None:
    sections = [
        {
            "title": "Demo notes",
            "cards": [
                {
                    "label": "Prompt | retrieval",
                    "value": "pass | warn",
                    "status": "warn",
                    "detail": "Check query | evidence before demo export.",
                }
            ],
        }
    ]

    markdown = build_observability_dashboard_markdown(sections)

    assert "| Prompt \\| retrieval | pass \\| warn | warn | Check query \\| evidence before demo export. |" in markdown


def test_build_unsupported_claim_report_markdown_lists_failed_claims_for_review() -> None:
    from traceable_rag_agent import pipeline

    evidence = [
        Evidence(
            source_id="retrieval",
            text="Agentic RAG checks retrieved evidence before answering.",
            score=1.0,
        )
    ]
    claim_checks = check_answer_claims(
        "Agentic RAG checks retrieved evidence before answering. It guarantees perfect answers.",
        evidence,
    )

    markdown = pipeline.build_unsupported_claim_report_markdown(claim_checks)

    assert markdown == "\n".join(
        [
            "## Unsupported Claim Review",
            "",
            "| Claim | Status | Supporting sources |",
            "| --- | --- | --- |",
            "| It guarantees perfect answers. | unsupported | none |",
        ]
    )


def test_build_retrieval_retry_report_markdown_shows_recovery_from_weak_evidence() -> None:
    from traceable_rag_agent import pipeline

    trace = answer_question(
        "How are hallucinations prevented?",
        [
            Document(
                source_id="faithfulness",
                text="Unsupported claim checks flag claims without evidence before final answers.",
            )
        ],
        top_k=1,
    )

    markdown = pipeline.build_retrieval_retry_report_markdown(trace)

    assert markdown == "\n".join(
        [
            "## Retrieval Retry Report",
            "",
            "| Retry | Original query | Retry query | Before sources | After sources | Status |",
            "| ---: | --- | --- | --- | --- | --- |",
            "| 1 | How are hallucinations prevented? | unsupported claims evidence | none | faithfulness | recovered |",
        ]
    )



def test_build_retrieval_plan_markdown_shows_query_reasons_for_interview_demo() -> None:
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        [
            Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
            Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
        ],
        top_k=1,
    )

    markdown = build_retrieval_plan_markdown(trace)

    assert markdown == "\n".join(
        [
            "## Retrieval Plan",
            "",
            "| Step | Query | Reason | Retrieved sources |",
            "| ---: | --- | --- | --- |",
            "| 1 | How does Agentic RAG decompose questions? | Retrieve evidence for one focused part of the complex question. | decomposition |",
            "| 2 | How does Agentic RAG check evidence sufficiency? | Retrieve evidence for one focused part of the complex question. | sufficiency |",
        ]
    )


def test_build_source_attribution_markdown_shows_retrieved_cited_and_supporting_sources() -> None:
    trace = RagTrace(
        question="How does Agentic RAG attribute answer sources?",
        planned_queries=[
            RetrievalQuery(
                query="How does Agentic RAG attribute answer sources?",
                reason="Use the original user question for first-pass retrieval.",
            )
        ],
        retrieval_steps=[
            RetrievalStep(
                query="How does Agentic RAG attribute answer sources?",
                retrieved_source_ids=["retrieval", "faithfulness", "notes|draft"],
                retrieved_chunk_ids=["retrieval#0", "faithfulness#0", "notes|draft#0"],
            )
        ],
        evidence=[
            Evidence(source_id="retrieval", text="Agentic RAG retrieves evidence before synthesis.", score=0.91),
            Evidence(source_id="faithfulness", text="Faithfulness checks compare answer claims to evidence.", score=0.82),
            Evidence(source_id="notes|draft", text="Draft notes are retrieved but not cited in final answers.", score=0.7),
        ],
        answer=(
            "Agentic RAG retrieves evidence before synthesis. [retrieval] "
            "Faithfulness checks compare answer claims to evidence. [faithfulness]"
        ),
        claim_checks=[
            ClaimCheck(
                claim="Agentic RAG retrieves evidence before synthesis.",
                status="supported",
                supporting_source_ids=["retrieval"],
            ),
            ClaimCheck(
                claim="Faithfulness checks compare answer claims to evidence.",
                status="supported",
                supporting_source_ids=["faithfulness"],
            ),
        ],
    )

    markdown = build_source_attribution_markdown(trace)

    assert markdown == "\n".join(
        [
            "## Source Attribution",
            "",
            "| Source | Evidence ranks | Cited in answer | Supported claims | Top score |",
            "| --- | --- | --- | ---: | ---: |",
            "| retrieval | 1 | yes | 1 | 0.91 |",
            "| faithfulness | 2 | yes | 1 | 0.82 |",
            "| notes\\|draft | 3 | no | 0 | 0.70 |",
        ]
    )


def test_build_quality_report_markdown_formats_dashboard_ready_summary() -> None:
    summary = {
        "total_questions": 4,
        "passed_questions": 1,
        "failed_questions": 3,
        "failure_mode_counts": {
            "pass": 1,
            "retrieval_gap": 2,
            "unsupported_citation": 1,
        },
        "top_failure_mode": "retrieval_gap",
    }
    action = {
        "priority": "fix_retrieval",
        "message": "Top failure mode is retrieval_gap across 2 of 4 questions; improve query rewriting, retrieval recall, or reranking before changing answer synthesis.",
    }

    markdown = build_quality_report_markdown(summary, action)

    assert markdown == "\n".join(
        [
            "## RAG Quality Report",
            "",
            "- Total questions: 4",
            "- Passed questions: 1",
            "- Failed questions: 3",
            "- Top failure mode: retrieval_gap",
            "- Next priority: fix_retrieval",
            "- Recommendation: Top failure mode is retrieval_gap across 2 of 4 questions; improve query rewriting, retrieval recall, or reranking before changing answer synthesis.",
            "",
            "| Failure mode | Count |",
            "| --- | ---: |",
            "| pass | 1 |",
            "| retrieval_gap | 2 |",
            "| unsupported_citation | 1 |",
        ]
    )


def test_build_unsupported_citation_report_markdown_lists_invalid_citations_for_debugging() -> None:
    from traceable_rag_agent import pipeline

    quality_report = {
        "results": [
            {
                "question": "How does Agentic RAG cite evidence?",
                "citation_support": {
                    "cited_source_ids": ["decomposition"],
                    "evidence_source_ids": ["decomposition"],
                    "supported_cited_source_ids": ["decomposition"],
                    "unsupported_cited_source_ids": [],
                    "citation_support_ratio": 1.0,
                },
            },
            {
                "question": "How does Agentic RAG check citations | evidence?",
                "citation_support": {
                    "cited_source_ids": ["evidence", "missing|source"],
                    "evidence_source_ids": ["evidence"],
                    "supported_cited_source_ids": ["evidence"],
                    "unsupported_cited_source_ids": ["missing|source"],
                    "citation_support_ratio": 0.5,
                },
            },
        ]
    }

    markdown = pipeline.build_unsupported_citation_report_markdown(quality_report)

    assert markdown == "\n".join(
        [
            "## Unsupported Citation Report",
            "",
            "| Question | Citation support | Unsupported citations | Retrieved evidence sources |",
            "| --- | ---: | --- | --- |",
            "| How does Agentic RAG check citations \\| evidence? | 50% | missing\\|source | evidence |",
        ]
    )


def test_build_evidence_sufficiency_gap_report_markdown_lists_insufficient_traces() -> None:
    from traceable_rag_agent import pipeline

    supported_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    unsupported_trace = answer_question(
        "How does Agentic RAG check unsupported | missing evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )

    markdown = pipeline.build_evidence_sufficiency_gap_report_markdown(
        [
            {"label": "supported demo", "trace": supported_trace},
            {"label": "unsupported | demo", "trace": unsupported_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Evidence Sufficiency Gap Report",
            "",
            "| Trace | Status | Evidence items | Unsupported claims | Reason |",
            "| --- | --- | ---: | ---: | --- |",
            "| unsupported \\| demo | insufficient | 0 | 1 | No evidence was retrieved for this question. |",
        ]
    )


def test_build_interview_risk_register_markdown_summarizes_demo_risks_and_mitigations() -> None:
    from traceable_rag_agent import pipeline

    safe_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    no_evidence_trace = answer_question(
        "How does Agentic RAG handle missing retrieval?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )
    unsupported_trace = RagTrace(
        question="How does Agentic RAG prevent unsupported claims?",
        planned_queries=[RetrievalQuery(query="unsupported claims", reason="Use the original user question for first-pass retrieval.")],
        retrieval_steps=[RetrievalStep(query="unsupported claims", retrieved_source_ids=["faithfulness"], retrieved_chunk_ids=["faithfulness#0"])],
        evidence=[Evidence(source_id="faithfulness", text="Faithfulness checks compare answer claims to retrieved evidence.", score=0.9)],
        answer="Faithfulness checks compare answer claims to retrieved evidence. [faithfulness] The agent always has perfect recall.",
        claim_checks=[
            ClaimCheck(
                claim="Faithfulness checks compare answer claims to retrieved evidence.",
                status="supported",
                supporting_source_ids=["faithfulness"],
            ),
            ClaimCheck(
                claim="The agent always has perfect recall.",
                status="unsupported",
                supporting_source_ids=[],
            ),
        ],
        evidence_sufficiency=check_evidence_sufficiency(
            [Evidence(source_id="faithfulness", text="Faithfulness checks compare answer claims to retrieved evidence.", score=0.9)],
            [
                ClaimCheck(
                    claim="Faithfulness checks compare answer claims to retrieved evidence.",
                    status="supported",
                    supporting_source_ids=["faithfulness"],
                ),
                ClaimCheck(
                    claim="The agent always has perfect recall.",
                    status="unsupported",
                    supporting_source_ids=[],
                ),
            ],
        ),
    )

    markdown = pipeline.build_interview_risk_register_markdown(
        [
            {"label": "safe demo", "trace": safe_trace},
            {"label": "no evidence | demo", "trace": no_evidence_trace},
            {"label": "unsupported claim demo", "trace": unsupported_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Interview Risk Register",
            "",
            "| Trace | Risk level | Risk signal | Mitigation | Interview framing |",
            "| --- | --- | --- | --- | --- |",
            "| safe demo | low | evidence_sufficient | Deliver with citations. | Use as the happy-path grounded-answer demo. |",
            "| no evidence \\| demo | medium | no_retrieved_evidence | Retry or broaden retrieval before answering. | Shows the agent refuses weak evidence instead of hallucinating. |",
            "| unsupported claim demo | high | unsupported_claims | Revise answer or escalate to human review. | Shows faithfulness checks catch claims that retrieval did not support. |",
        ]
    )



def test_build_trace_replay_plan_markdown_lists_reproducible_debug_steps() -> None:
    trace = answer_question(
        "How does Agentic RAG decompose questions and check evidence sufficiency?",
        [
            Document(source_id="decomposition", text="Agentic RAG decomposes questions into focused retrieval queries."),
            Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis."),
        ],
        top_k=1,
    )

    markdown = build_trace_replay_plan_markdown(trace)

    assert markdown == "\n".join(
        [
            "## Trace Replay Plan",
            "",
            "- Original question: How does Agentic RAG decompose questions and check evidence sufficiency?",
            "- Replay goal: reproduce the retrieval-to-answer path for debugging or interview review.",
            "",
            "| Step | Stage | What to replay | Expected trace signal |",
            "| ---: | --- | --- | --- |",
            "| 1 | query_planning | How does Agentic RAG decompose questions? | reason: Retrieve evidence for one focused part of the complex question. |",
            "| 2 | retrieval | How does Agentic RAG decompose questions? | sources: decomposition; chunks: decomposition#0 |",
            "| 3 | query_planning | How does Agentic RAG check evidence sufficiency? | reason: Retrieve evidence for one focused part of the complex question. |",
            "| 4 | retrieval | How does Agentic RAG check evidence sufficiency? | sources: sufficiency; chunks: sufficiency#0 |",
            "| 5 | synthesis | citation-grounded answer | evidence items: 2; checked claims: 2 |",
            "| 6 | evaluation | evidence sufficiency gate | sufficient: All answer claims are supported by retrieved evidence. |",
        ]
    )


def test_build_answer_approval_gate_markdown_marks_safe_and_blocked_answers() -> None:
    from traceable_rag_agent import pipeline

    supported_trace = answer_question(
        "How does Agentic RAG check evidence sufficiency?",
        [Document(source_id="sufficiency", text="Agentic RAG checks evidence sufficiency before final synthesis.")],
        top_k=1,
    )
    unsupported_trace = answer_question(
        "How does Agentic RAG handle missing | evidence?",
        [Document(source_id="unrelated", text="Workflow agents inspect tool results before continuing.")],
        top_k=1,
    )

    markdown = pipeline.build_answer_approval_gate_markdown(
        [
            {"label": "supported answer", "trace": supported_trace},
            {"label": "missing | evidence", "trace": unsupported_trace},
        ]
    )

    assert markdown == "\n".join(
        [
            "## Answer Approval Gate",
            "",
            "| Trace | Gate decision | Evidence status | Reviewer note |",
            "| --- | --- | --- | --- |",
            "| supported answer | approve_for_delivery | sufficient | Evidence is sufficient; answer can be delivered with citations. |",
            "| missing \\| evidence | block_for_revision | insufficient | No evidence was retrieved for this question. |",
        ]
    )
