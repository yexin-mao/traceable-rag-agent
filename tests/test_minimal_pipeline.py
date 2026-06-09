import json

from traceable_rag_agent.models import Evidence

from traceable_rag_agent.pipeline import (
    BenchmarkQuestion,
    Document,
    answer_question,
    build_evidence_table,
    build_quality_report_markdown,
    build_trace_report_markdown,
    build_trace_status_summary,
    build_trace_timeline_events,
    build_citation_evidence_map,
    build_evaluation_metric_cards,
    build_latency_metric_cards,
    check_answer_claims,
    check_evidence_sufficiency,
    chunk_documents,
    diagnose_rag_quality_failures,
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
            "event_type": "evaluate_sufficiency",
            "title": "Evaluated evidence sufficiency",
            "detail": "sufficient: All answer claims are supported by retrieved evidence.",
            "source_ids": ["decomposition", "sufficiency"],
            "chunk_ids": ["decomposition#0", "sufficiency#0"],
        },
    ]


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
