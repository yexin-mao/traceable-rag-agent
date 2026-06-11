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
- [x] Show retrieved evidence snippets
- [x] Add structured output format

### Milestone 3 — Agentic retrieval

- [x] Decompose complex question into sub-queries
- [x] Run multi-query retrieval
- [x] Add evidence sufficiency check
- [ ] Retry retrieval with rewritten query when evidence is weak

### Milestone 4 — Evaluation

- [x] Create benchmark questions
- [x] Measure retrieval coverage
- [x] Check citation support
- [x] Flag unsupported claims with deterministic claim checks
- [x] Summarize retrieval and citation quality in one benchmark report
- [x] Diagnose question-level failure modes from benchmark results
- [x] Summarize failure modes for dashboard/report status cards
- [x] Recommend the next evaluation/debugging priority from failure summaries
- [x] Format benchmark quality summaries as Markdown reports for dashboard/demo use

### Milestone 5 — Observability dashboard

- [x] Store per-query retrieval steps in each trace
- [x] Store every run as a persisted trace
- [x] Format one-run trace reports as Markdown for dashboard/demo rendering
- [x] Summarize one-run trace status metrics for dashboard cards
- [x] Build ordered timeline events for visualizing retrieval and evaluation steps
- [x] Visualize evidence and citations
- [x] Show evaluation metric cards for retrieval coverage, citation support, and unsupported claims
- [x] Record retrieval latency per step and summarize total retrieval latency for status cards
- [x] Show latency metric cards for total and average retrieval time
- [x] Flag slow retrieval runs against a configurable latency threshold
- [x] Show retrieval error metric cards
- [x] Show trace health metric cards for sufficiency, completeness, and claim support
- [x] Group health, evaluation, latency, and retrieval-error cards into dashboard-ready sections

## Development log

### 2026-06-11

- Added `build_observability_dashboard_sections(...)` to group trace health, evaluation quality, retrieval latency, and retrieval error cards into a single render-ready structure for a future dashboard.
- Added a strict TDD regression test proving a multi-query trace produces deterministic section titles and card contents.
- Verified RED/GREEN with the focused dashboard-section test, then ran the full test suite (`30 passed`) and `ruff check .`.

### 2026-06-10

- Added `build_trace_health_metric_cards(...)` to summarize one RAG run as dashboard-ready health cards: evidence sufficiency, trace completeness, and claim support.
- Added a strict TDD regression test proving a multi-query trace renders deterministic pass/fail health-card details.
- Verified RED/GREEN with the focused trace-health-card test, then ran the full test suite (`29 passed`) and `ruff check .`.

### 2026-06-10

- Added an optional `error` field to each `RetrievalStep` so traces can preserve retrieval failures alongside latency and retrieved-source metadata.
- Added `build_retrieval_error_metric_cards(...)` to turn failed retrieval steps into dashboard-ready error cards, including total error count and first-error detail.
- Verified strict RED/GREEN with the focused retrieval-error-card test, then ran the full test suite (`28 passed`) and `ruff check .`.

### 2026-06-09

- Added configurable `slow_threshold_ms` support to `build_latency_metric_cards(...)`, so dashboard latency cards now switch to `warn` when total retrieval time exceeds a chosen threshold.
- Added a strict TDD regression test proving slow retrieval is flagged with an interview/demo-ready warning detail.
- Verified RED/GREEN with the focused threshold test, then ran the full test suite (`27 passed`) and `ruff check .`.

### 2026-06-09

- Added `build_latency_metric_cards(...)` to turn per-step retrieval timings into dashboard-ready total and average latency cards.
- Added a regression test proving deterministic latency card formatting for future observability UI work.
- Verified strict RED/GREEN with the focused latency-card test, then ran the full test suite (`26 passed`) and `ruff check .`.

### 2026-06-08

- Added per-retrieval `latency_ms` tracking to each `RetrievalStep` and exposed `total_retrieval_latency_ms` in `build_trace_status_summary(...)`.
- This makes the trace dashboard path more production-oriented: interviewers can see not only whether retrieval worked, but how expensive each retrieval step was.
- Verified strict RED/GREEN on the focused retrieval-step and status-summary tests, then ran the full test suite (`25 passed`) and `ruff check .`.

### 2026-06-08

- Added `build_evaluation_metric_cards(...)` to turn one RAG trace into dashboard-ready cards for retrieval coverage, citation support, and unsupported-claim counts.
- Added a regression test proving the metric cards are deterministic and interview/demo ready.
- Verified RED/GREEN with the focused test, then ran the full test suite (`25 passed`) and `ruff check .`.

### 2026-06-07

- Added `build_citation_evidence_map(...)` to connect answer citations back to retrieved evidence for future dashboard rendering.
- The map reports whether each cited source was retrieved, its evidence rank, chunk ID, snippet, and supporting-claim count.
- Added a regression test proving multi-source answers produce deterministic citation-to-evidence links.
- Verified the focused test, full test suite (`24 passed`), and `ruff check .`.

### 2026-06-07

- Added `build_trace_timeline_events(...)` to convert one RAG trace into ordered observability events for a future dashboard timeline.
- The timeline now includes query-planning events, retrieval events with source/chunk IDs, and final evidence-sufficiency evaluation.
- Added a regression test proving multi-query traces produce deterministic timeline events.
- Verified the focused test, full test suite (`23 passed`), and `ruff check .`.

### 2026-06-06

- Added `build_trace_status_summary(...)` to turn one RAG run into compact dashboard-card metrics: sufficiency status, planned-query count, retrieval-step count, evidence count, supported/unsupported claim counts, and cited-source count.
- Added a regression test proving a multi-query trace produces deterministic status metrics for a future observability UI.
- Verified the focused test, full test suite, and `ruff check .`.

### 2026-06-06

- Added `build_trace_report_markdown(...)` to render one full RAG trace as Markdown for a future observability dashboard or demo page.
- The report shows the original question, evidence sufficiency status, planned-query count, evidence count, retrieval-step table, and ranked evidence table.
- Tightened deterministic claim checks so citation-only fragments such as `[source-id]` do not become false unsupported claims.
- Added a regression test proving a multi-query run formats into a deterministic trace report.
- Verified the focused test, full test suite (`21 passed`), and `ruff check .`.

### 2026-06-05

- Added `build_quality_report_markdown(...)` to turn evaluation summaries and recommended next actions into a compact Markdown report for dashboard/demo use.
- The report includes total/pass/fail counts, top failure mode, next priority, recommendation text, and a failure-mode table.
- Added a regression test proving the Markdown output is deterministic and ready to render in a project page, README excerpt, or future observability dashboard.
- Verified the focused test, full test suite, and `ruff check .`.

### 2026-06-05

- Added `recommend_next_evaluation_action(...)` to convert benchmark failure summaries into one concrete next debugging priority.
- The helper distinguishes retrieval gaps from unsupported citations and returns an interview-friendly action message, making evaluation results more operational instead of just descriptive.
- Added a regression test proving a retrieval-gap-heavy benchmark recommends improving query rewriting, retrieval recall, or reranking before changing answer synthesis.
- Verified the focused test, full test suite, and `ruff check .`.

### 2026-06-04

- Added `summarize_failure_diagnoses(...)` to turn question-level failure labels into dashboard-ready counts: total, passed, failed, per-mode counts, and top failure mode.
- This makes benchmark failure analysis easier to show in a report or UI status card before building the full dashboard.
- Added a regression test covering one passing item, two retrieval gaps, and one unsupported-citation failure.
- Verified the focused test, full test suite, and `ruff check .`.

### 2026-06-04

- Added `diagnose_rag_quality_failures(...)` to convert benchmark metrics into question-level failure-mode labels: `pass`, `retrieval_gap`, or `unsupported_citation`.
- The diagnosis report explains whether a failed RAG answer is caused by missing expected evidence or by citing a source that was not retrieved.
- Added a regression test covering one passing question, one retrieval gap, and one unsupported-citation failure.
- Verified the focused test, full test suite, and `ruff check .`.

### 2026-06-03

- Added `run_rag_quality_benchmark(...)` so one benchmark run now reports both retrieval coverage and citation-support quality.
- The report keeps per-question retrieval coverage and citation support side by side, plus average retrieval and citation ratios across the benchmark set.
- Added a regression test proving a two-question benchmark reports average retrieval coverage of `0.75`, average citation support of `1.0`, and question-level missing-source details.
- Verified the full test suite: `16 passed`; `ruff check .` passed.

### 2026-06-03

- Added `BenchmarkQuestion` and `run_retrieval_coverage_benchmark(...)` so multiple benchmark questions can be run through the current RAG pipeline and summarized in one evaluation report.
- The report includes per-question expected sources, retrieved sources, missing sources, coverage ratios, and an average retrieval coverage score across the benchmark set.
- Added a regression test proving a two-question benchmark reports one fully covered question, one partially covered question, and an average coverage ratio of `0.75`.
- Verified the full test suite: `15 passed`; `ruff check .` passed.

### 2026-06-02

- Added `measure_citation_support(trace)` to verify that answer citations point back to retrieved evidence sources.
- The helper reports cited sources, evidence sources, supported/unsupported citation IDs, and a citation-support ratio, making citation accuracy easier to evaluate before a dashboard exists.
- Added a regression test proving an answer with one valid citation and one missing citation reports a `0.5` support ratio and identifies the unsupported source.
- Verified the full test suite: `14 passed`; `ruff check .` passed.

### 2026-06-02

- Added `measure_retrieval_coverage(trace, expected_source_ids)` to compare retrieved evidence sources against an expected source set.
- The helper reports expected sources, retrieved sources, found sources, missing sources, and a coverage ratio, creating a small deterministic evaluation signal for RAG quality checks.
- Added a regression test proving partial retrieval coverage is reported as `0.5` with the missing source identified.
- Verified the full test suite: `13 passed`; `ruff check .` passed.

### 2026-06-01

- Added `save_trace_json(trace, output_path)` to persist one full RAG run as readable JSON for later trace inspection and dashboard work.
- Added a regression test proving saved traces preserve the question, retrieval-step chunk IDs, and evidence sufficiency status.
- Verified the full test suite: `12 passed`; `ruff check .` passed.

### 2026-06-01

- Added `build_evidence_table(trace)`, a dashboard-ready view of ranked evidence snippets with rank, source ID, chunk ID, retrieval score, and snippet text.
- Added a regression test proving multi-query traces can be rendered into a simple evidence table without losing citation/source metadata.
- Verified the full test suite: `11 passed`.

### 2026-05-31

- Added per-query retrieval step records to `RagTrace`, including the planned query, retrieved source IDs, and retrieved chunk IDs for each retrieval action.
- Added a regression test proving multi-query runs keep a readable step-by-step retrieval trail for future observability/debugging UI work.
- Added deterministic retrieval planning for complex `and` questions, splitting one broad question into focused sub-queries before retrieval.
- Updated `answer_question` to run retrieval for every planned sub-query and deduplicate repeated chunks in the trace evidence list.
- Added regression tests for focused sub-query planning and multi-query evidence collection.

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
