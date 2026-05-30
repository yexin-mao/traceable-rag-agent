from traceable_rag_agent.models import Evidence
from traceable_rag_agent.pipeline import Document, answer_question, check_answer_claims, chunk_documents, retrieve


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
