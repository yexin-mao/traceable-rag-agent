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

## Relationship to Agent Flight Recorder

This project will focus on **RAG capability**. Agent Flight Recorder focuses on **agent observability**.

The long-term plan is to connect them:

```text
Traceable RAG Agent produces traces
Agent Flight Recorder visualizes and evaluates those traces
```

That gives the portfolio two connected but distinct projects:

1. **Traceable RAG Agent** — proves RAG and agent workflow ability.
2. **Agent Flight Recorder** — proves observability and debugging infrastructure ability.

## Roadmap

### Milestone 1 — Minimal RAG pipeline

- [x] Load in-memory documents
- [x] Chunk documents with source-preserving chunk IDs
- [ ] Embed chunks
- [x] Retrieve top-k evidence with a deterministic lexical baseline
- [x] Generate citation-grounded answer skeleton

### Milestone 2 — Citation-grounded answers

- [ ] Return answer with source citations
- [ ] Show retrieved evidence snippets
- [ ] Add structured output format

### Milestone 3 — Agentic retrieval

- [ ] Decompose complex question into sub-queries
- [ ] Run multi-query retrieval
- [ ] Add evidence sufficiency check
- [ ] Retry retrieval with rewritten query when evidence is weak

### Milestone 4 — Evaluation

- [ ] Create benchmark questions
- [ ] Measure retrieval coverage
- [ ] Check citation support
- [ ] Flag unsupported claims

### Milestone 5 — Observability dashboard

- [ ] Store every run as a trace
- [ ] Visualize retrieval steps
- [ ] Visualize evidence and citations
- [ ] Show latency, errors, and evaluation metrics

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
