# Traceable RAG Agent

A competitive **Agentic RAG** project for AI Agent engineer interviews.

This is not a simple "upload PDF and chat" demo. The goal is to build a RAG agent that can:

- decompose complex questions into retrieval sub-queries;
- retrieve and rerank evidence;
- generate citation-grounded answers;
- detect unsupported claims;
- record every retrieval, tool call, latency, and evaluation result;
- expose the whole run through an observability dashboard.

## Why this project exists

Many RAG demos only show the happy path:

```text
question -> retrieve chunks -> answer
```

Real AI Agent systems need more than that:

```text
question
  -> plan retrieval strategy
  -> run multiple searches
  -> inspect evidence quality
  -> synthesize answer with citations
  -> evaluate faithfulness
  -> debug failures from traces
```

This repository is designed to prove practical ability in **RAG chain optimization, Agent workflow design, evaluation, and observability**.

## Interview positioning

> I built a traceable RAG agent that decomposes complex queries, retrieves and ranks evidence, generates citation-grounded answers, evaluates unsupported claims, and records the full execution trace for debugging and improvement.

## Target capabilities

### 1. Agentic RAG workflow

- Query decomposition
- Multi-query retrieval
- Reranking
- Evidence selection
- Citation-grounded synthesis
- Follow-up retrieval when evidence is insufficient

### 2. Evaluation

- Retrieval coverage
- Citation accuracy
- Faithfulness / unsupported-claim detection
- Latency and cost tracking
- Regression test set for RAG quality

### 3. Observability

- Trace every step of the RAG workflow
- Store query, retrieved chunks, selected evidence, model outputs, metrics, and errors
- Inspect why an answer failed: bad retrieval, weak evidence, hallucination, or synthesis issue

### 4. Product demo

- FastAPI backend
- Simple web dashboard
- Example datasets
- Demo video script
- Public online demo target

## Planned architecture

```text
User Question
   |
   v
Query Planner
   |
   v
Retriever -> Reranker -> Evidence Store
   |
   v
Answer Synthesizer
   |
   v
Citation Checker / Faithfulness Evaluator
   |
   v
Trace Dashboard
```

## Portfolio context

This project is the portfolio's **knowledge/evidence agent**. It pairs with **Deep Agent Workbench**, the separate action/workflow agent project:

```text
Traceable RAG Agent -> proves reliable knowledge grounding, citations, and RAG evaluation
Deep Agent Workbench -> proves planning, tool use, state, approval, recovery, and task evaluation
```

Together they show two sides of AI Agent engineering: trustworthy answers from evidence, and safe multi-step action workflows.

## Roadmap

### Milestone 1 — Minimal RAG pipeline

- [x] Load in-memory documents
- [x] Chunk documents with source-preserving chunk IDs
- [ ] Embed chunks
- [x] Retrieve top-k evidence with a deterministic lexical baseline
- [x] Generate citation-grounded answer skeleton

### Milestone 2 — Citation-grounded answers

- [x] Return answer with source citations
- [ ] Show retrieved evidence snippets
- [x] Add structured output format

### Milestone 3 — Agentic retrieval

- [ ] Decompose complex question into sub-queries
- [ ] Run multi-query retrieval
- [x] Add evidence sufficiency check
- [ ] Retry retrieval with rewritten query when evidence is weak

### Milestone 4 — Evaluation

- [ ] Create benchmark questions
- [ ] Measure retrieval coverage
- [ ] Check citation support
- [x] Flag unsupported claims with deterministic claim checks

### Milestone 5 — Observability dashboard

- [ ] Store every run as a trace
- [ ] Visualize retrieval steps
- [ ] Visualize evidence and citations
- [ ] Show latency, errors, and evaluation metrics

## Development log

### 2026-05-30

- Added deterministic evidence sufficiency summaries to each `RagTrace`, including retrieved evidence count, supported claim count, unsupported claim count, and a plain-English reason.
- Added regression tests for sufficient traces and no-evidence insufficient traces.
- Added deterministic claim-level support checks that classify answer sentences as `supported` or `unsupported` against retrieved evidence.
- The pipeline now stores `claim_checks` directly in `RagTrace`, making every answer easier to inspect and evaluate.
- Verified with pytest regression tests for citation-grounded traces and unsupported-claim detection.

## Tech stack target

- Python
- FastAPI
- SQLite / Postgres
- Vector database: Chroma, Qdrant, or pgvector
- LangChain or LlamaIndex for selected components
- LangGraph for agent workflow control
- React or lightweight HTML dashboard
- pytest for regression tests

## What makes this competitive

A normal RAG demo says:

> It can answer questions from documents.

This project says:

> It can explain, evaluate, and debug how a RAG answer was produced.

That is closer to what real AI Agent engineering roles need.
