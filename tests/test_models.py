from traceable_rag_agent.models import Evidence, RagTrace, RetrievalQuery


def test_minimal_trace_model() -> None:
    trace = RagTrace(
        question="What makes Agentic RAG different from simple RAG?",
        planned_queries=[
            RetrievalQuery(
                query="agentic rag multi step retrieval",
                reason="Find evidence about iterative retrieval workflows.",
            )
        ],
        evidence=[
            Evidence(
                source_id="doc-1",
                text="Agentic RAG can plan multiple retrieval actions before answering.",
                score=0.9,
            )
        ],
        answer="Agentic RAG plans and checks retrieval before answering. [doc-1]",
    )

    assert trace.planned_queries[0].query == "agentic rag multi step retrieval"
    assert trace.evidence[0].source_id == "doc-1"
