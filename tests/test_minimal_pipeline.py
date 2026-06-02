import json

from traceable_rag_agent.models import Evidence

from traceable_rag_agent.pipeline import (
    Document,
    answer_question,
    build_evidence_table,
    check_answer_claims,
    check_evidence_sufficiency,
    chunk_documents,
    measure_retrieval_coverage,
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
