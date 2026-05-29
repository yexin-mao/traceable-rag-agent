# System Design

## Project thesis

RAG becomes much more valuable when it is **traceable** and **measurable**.

A production RAG system should not only return an answer. It should also show:

- what it searched for;
- what it retrieved;
- which evidence was used;
- which claims are supported;
- where the system failed;
- how the answer quality changed over time.

## Core data model

### Run

One user question and its full execution.

Fields:

- `run_id`
- `question`
- `status`
- `created_at`
- `latency_ms`
- `final_answer`
- `evaluation_summary`

### Step

One action inside a run.

Examples:

- `plan_queries`
- `retrieve_documents`
- `rerank_evidence`
- `synthesize_answer`
- `check_citations`
- `evaluate_faithfulness`

Fields:

- `step_id`
- `run_id`
- `name`
- `type`
- `input_json`
- `output_json`
- `latency_ms`
- `error`

### Evidence

A retrieved source snippet.

Fields:

- `evidence_id`
- `run_id`
- `source_id`
- `text`
- `score`
- `metadata`

### Claim

A sentence or atomic statement in the final answer.

Fields:

- `claim_id`
- `run_id`
- `claim_text`
- `supporting_evidence_ids`
- `support_status`: `supported | weak | unsupported`

## MVP workflow

```text
1. User asks a question
2. Planner creates 2-4 search queries
3. Retriever gets candidate chunks
4. Reranker selects the best evidence
5. Synthesizer writes answer with citations
6. Evaluator checks whether each claim is supported
7. Recorder stores the trace
8. Dashboard shows the run
```

## Evaluation philosophy

The goal is not only to get a fluent answer. The goal is to know whether the answer is **grounded**.

Key metrics:

- Retrieval coverage: did we retrieve enough relevant evidence?
- Citation accuracy: do citations point to the right evidence?
- Faithfulness: are answer claims supported by evidence?
- Latency: how long did each stage take?
- Cost: how many model calls were used?

## Demo scenario ideas

Use public technical documents first, not private data.

Good initial datasets:

- LangGraph documentation pages
- OpenAI / Anthropic tool calling docs
- selected AI Agent blog posts
- small benchmark set written manually

Example demo question:

> How does LangGraph's state model help build more reliable AI agents compared with a simple chain?

The project should show:

- the decomposed search queries;
- retrieved evidence;
- answer citations;
- unsupported claims, if any;
- the trace timeline.
