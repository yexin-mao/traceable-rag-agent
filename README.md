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
- [x] Format retrieval plans with query reasons for interview/demo pages
- [x] Add evidence sufficiency check
- [x] Retry retrieval with rewritten query when evidence is weak

### Milestone 4 — Evaluation

- [x] Create benchmark questions
- [x] Measure retrieval coverage
- [x] Check citation support
- [x] Flag unsupported claims with deterministic claim checks
- [x] Summarize retrieval and citation quality in one benchmark report
- [x] Diagnose question-level failure modes from benchmark results
- [x] Format question-level failure diagnoses as Markdown debug tables
- [x] Format retrieval-gap debug reports that list missing expected sources per failed benchmark question
- [x] Format unsupported-citation debug reports that list invalid citations and retrieved evidence sources
- [x] Format evidence-decision logs that mark traces as ready for answer delivery or needing retry/escalation
- [x] Format answer-approval gates that block delivery when evidence is insufficient
- [x] Format human-review queues that list only blocked answer traces and reviewer actions
- [x] Format human-review action summaries that group blocked traces by reviewer next action
- [x] Format human-review checklists that turn blocked traces into concrete operator tasks
- [x] Format human-review priority boards that order blocked traces by review risk
- [x] Format human-review decision logs that record reviewer approval/retry outcomes for auditability
- [x] Summarize human-review decision outcomes for approval vs revision tracking
- [x] Format human-review decision summaries as Markdown for dashboard/report pages
- [x] Format human-review revision queues for non-approved reviewer decisions and follow-up actions
- [x] Summarize human-review workload for approved, blocked, insufficient, and not-yet-evaluated traces
- [x] Format human-review workload summaries as Markdown for dashboard/report pages
- [x] Format human-review escalation briefs that summarize blocked traces for reviewer handoff
- [x] Format recovery action plans that map insufficient traces to concrete next steps
- [x] Format evidence-sufficiency gap reports that list insufficient traces and reasons
- [x] Format source-attribution reports that connect retrieved sources to citations and supported claims
- [x] Summarize failure modes for dashboard/report status cards
- [x] Recommend the next evaluation/debugging priority from failure summaries
- [x] Format benchmark quality summaries as Markdown reports for dashboard/demo use
- [x] Format unsupported-claim review tables for faithfulness debugging

### Milestone 5 — Observability dashboard

- [x] Store per-query retrieval steps in each trace
- [x] Store every run as a persisted trace
- [x] Format one-run trace reports as Markdown for dashboard/demo rendering
- [x] Summarize one-run trace status metrics for dashboard cards
- [x] Build ordered timeline events for visualizing retrieval and evaluation steps
- [x] Include answer-synthesis events in the trace timeline before evaluation
- [x] Format timeline events as Markdown tables for demo/report pages
- [x] Format interviewer-friendly trace summaries that connect evidence metrics to Agentic RAG concepts
- [x] Visualize evidence and citations
- [x] Format citation-to-evidence maps as Markdown tables for demo/review pages
- [x] Show evaluation metric cards for retrieval coverage, citation support, and unsupported claims
- [x] Record retrieval latency per step and summarize total retrieval latency for status cards
- [x] Show latency metric cards for total and average retrieval time
- [x] Flag slow retrieval runs against a configurable latency threshold
- [x] Show retrieval error metric cards
- [x] Show trace health metric cards for sufficiency, completeness, and claim support
- [x] Group health, evaluation, latency, and retrieval-error cards into dashboard-ready sections
- [x] Format observability dashboard sections as Markdown tables for demo/report pages
- [x] Escape Markdown table separators in dashboard card text so demo exports stay render-safe
- [x] Escape Markdown table separators in trace-report fields so retrieval-step and evidence exports stay render-safe
- [x] Format interview evidence packets that summarize one trace in recruiter/interviewer-friendly language
- [x] Format interview walkthroughs that turn a trace into a concise Agentic RAG story
- [x] Format interview readiness scorecards that summarize demo batches for recruiter/interviewer review
- [x] Format trace-backed interview follow-up questions for common Agentic RAG reviewer prompts
- [x] Format interview risk registers that map demo-trace risks to mitigations and talking points
- [x] Format interview concept maps that connect trace metrics to Agentic RAG interview concepts
- [x] Format interview demo scripts that sequence ready and blocked traces into a recruiter/interviewer walkthrough
- [x] Format interview objection-handling tables that answer common reviewer concerns with trace-backed metrics
- [x] Format trace replay plans that reproduce query planning, retrieval, synthesis, and evaluation steps for debugging/interviews
- [x] Format honest interview system-limit notes with current constraints and next engineering steps
- [x] Format architecture tradeoff tables that explain design choices, limits, and next engineering steps for interviews
- [x] Format honest portfolio demo status notes that separate current tested assets from planned hosted demo work
- [x] Format recruiter-facing portfolio progress snapshots from tested trace batches
- [x] Format portfolio capability matrices that pair each claimed capability with evidence, status, and next steps
- [x] Format portfolio evidence indexes that link recruiter-facing claims to concrete artifacts
- [x] Format portfolio next-step plans that prioritize planned public-demo and retrieval improvements without overclaiming live features
- [x] Format portfolio release notes that summarize completed daily increments, validation, and next steps for GitHub Pages updates
- [x] Format trace-backed portfolio interview highlights that explain Agentic RAG value in recruiter-friendly bullets
- [x] Format portfolio demo-readiness checklists that separate tested Agentic RAG capabilities from planned hosted-demo work
- [x] Format portfolio interview talking points that answer common reviewer prompts with trace-backed proof
- [x] Format honest portfolio demo FAQs that answer recruiter questions without claiming unverified live features
- [x] Format portfolio project-status tables that separate current proof from planned demo work for GitHub Pages updates
- [x] Format portfolio review links that separate current reviewable artifacts from planned demo links
- [x] Format portfolio evaluation snapshots that summarize benchmark coverage and citation-support metrics for recruiters
- [x] Format compact portfolio metric badges that summarize tests, benchmark quality, retrieval gaps, citation support, and hosted-demo status
- [x] Format a recruiter-facing Agentic RAG vs simple ChatPDF comparison with concrete proof points

## Development log

### 2026-07-09

- Added `build_portfolio_agentic_rag_comparison_markdown(...)` to explain why the project is an evidence-debuggable Agentic RAG system rather than a generic ChatPDF clone.
- The comparison renderer contrasts single-search PDF chat with query decomposition, traceable retrieval steps, evidence gates, and honest hosted-demo status.
- Verified strict RED/GREEN with the focused comparison test before running the full suite.
- Added `build_portfolio_metric_badges_markdown(...)` to render compact recruiter-facing proof badges for GitHub Pages portfolio updates.
- The badge renderer summarizes passing tests, benchmark count, average retrieval coverage, citation support, retrieval gaps, unsupported citations, and hosted-demo status without implying a live demo exists.
- Verified strict RED/GREEN with the focused metric-badges test before running the full suite.

### 2026-07-08

- Added `build_portfolio_evaluation_snapshot_markdown(...)` to turn RAG quality benchmark output into a recruiter-facing evaluation snapshot.
- The snapshot reports benchmark count, average retrieval coverage, citation support, retrieval gaps, and per-question status while explicitly avoiding hosted-demo overclaims.
- Verified strict RED/GREEN with the focused evaluation-snapshot test before running the full suite.

### 2026-07-08

- Added `build_portfolio_project_status_markdown(...)` to render current-vs-planned project status tables for GitHub Pages portfolio updates.
- The status renderer keeps recruiter-facing copy honest by pairing each project area with current proof and a separate planned next step, including Markdown table escaping for labels such as hosted dashboard/demo.
- Verified strict RED/GREEN with the focused project-status test before running the full suite.

### 2026-07-07

- Added `build_portfolio_review_links_markdown(...)` to render recruiter/interviewer-facing links for current reviewable artifacts and planned demo placeholders.
- The review-links renderer keeps the portfolio honest by labeling the GitHub repo as current while marking hosted dashboard/demo links as planned until deployment is verified.
- Verified strict RED/GREEN with the focused review-links test before running broader validation.

### 2026-07-07

- Added `build_portfolio_demo_faq_markdown(...)` to render concise recruiter-facing FAQ rows about demo availability and current reviewable proof.
- The FAQ renderer keeps portfolio language honest by marking hosted demo work as planned while pointing reviewers to tested local traces and report helpers.
- Verified strict RED/GREEN with the focused FAQ test before running the full suite and linter.

### 2026-07-06

- Added `build_portfolio_interview_talking_points_markdown(...)` to turn common reviewer prompts into concise answer/proof rows for portfolio and interview prep.
- The talking-points renderer keeps claims anchored to deterministic local traces and explicitly avoids claiming hosted-demo behavior before deployment.
- Verified strict RED/GREEN with the focused talking-points test before running the full suite and linter.

### 2026-07-06

- Added `build_portfolio_recruiter_summary_markdown(...)` to turn tested trace batches into a concise recruiter-facing status block.
- The summary reports trace/query/evidence/claim counts, links the GitHub repo, and explicitly marks the hosted demo as planned unless a verified URL is provided.
- Verified strict RED/GREEN with the focused recruiter-summary test before running the full suite and linter.

### 2026-07-05

- Added `build_portfolio_demo_readiness_checklist_markdown(...)` to summarize demo readiness across query planning, evidence grounding, answer safety gates, and hosted-demo status.
- The checklist keeps portfolio claims honest by marking local/reporting capabilities as current while hosted public demo deployment remains planned unless explicitly available.
- Verified strict RED/GREEN with the focused demo-readiness checklist test before running the full suite and linter.

### 2026-07-05

- Added `build_portfolio_interview_highlights_markdown(...)` to turn tested trace batches into concise recruiter/interviewer bullets.
- The highlight renderer explains why the project is beyond ChatPDF, counts evidence/claim checks, and keeps hosted-demo status honest as planned rather than live.
- Verified strict RED/GREEN with the focused interview-highlights test before running broader validation.

### 2026-07-04

- Added `build_portfolio_release_notes_markdown(...)` to render completed daily increments as honest portfolio release notes for GitHub Pages/project-status updates.
- The release-note renderer separates shipped work, validation evidence, and next steps so portfolio updates can stay concrete without implying an unverified hosted demo.
- Verified strict RED/GREEN with the focused release-notes test before running the full suite and linter.

### 2026-07-04

- Added `build_portfolio_next_step_plan_markdown(...)` to render recruiter/interviewer-facing roadmap tables with priority, rationale, and honest planned status.
- The roadmap connects tested local Agentic RAG capabilities to the next public-demo steps without claiming that a hosted dashboard is already live.
- Verified strict RED/GREEN with the focused next-step-plan test before running the full suite and linter.

### 2026-07-03

- Added `build_portfolio_evidence_index_markdown(...)` to render recruiter/interviewer-facing artifact indexes that connect portfolio claims to concrete proof and value.
- The evidence index explicitly labels planned assets, so the portfolio can mention a hosted dashboard as planned without pretending it is live.
- Verified strict RED/GREEN with the focused evidence-index test before running the full suite and linter.

### 2026-07-03

- Added `build_portfolio_capability_matrix_markdown(...)` to render recruiter/interviewer-facing capability tables with honest `current` vs `planned` status.
- The matrix keeps portfolio claims evidence-based by pairing each capability with proof and a concrete next engineering step, without implying a hosted demo is live.
- Verified strict RED/GREEN with the focused capability-matrix test before running broader validation.

### 2026-07-02

- Added `build_portfolio_progress_snapshot_markdown(...)` to summarize tested demo-trace counts, implemented capabilities, and planned next steps for recruiter/interviewer portfolio updates.
- The snapshot keeps the project honest: current local Agentic RAG backend/reporting is implemented and tested, while hosted UI work remains planned.

### 2026-07-02

- Added `build_portfolio_demo_status_markdown(...)` to summarize current tested demo assets and planned public demo work without overclaiming.
- This gives the portfolio site and interview prep material an honest "current vs planned" status block for recruiter/interviewer review.

### 2026-07-01

- Added `build_interview_architecture_tradeoffs_markdown(...)` to turn project design choices into concise interview-ready tradeoff explanations.
- The report maps each architecture choice to its benefit, current limitation, and next engineering step, so the project can be presented honestly without overclaiming hosted/semantic-search capabilities.
- Verified strict RED/GREEN with the focused architecture-tradeoffs test before running the full suite and linter.

### 2026-07-01

- Added `build_interview_system_limits_markdown(...)` to state current system limits and planned engineering steps in a recruiter/interviewer-friendly format.
- The report keeps the portfolio honest by distinguishing implemented local trace/report helpers from still-planned embedding retrieval, reranking, and hosted dashboard work.
- Verified strict RED/GREEN with the focused system-limits test before running the full suite and linter.

### 2026-06-30

- Added `build_trace_replay_plan_markdown(...)` to render a deterministic replay plan for one RAG trace.
- The replay plan lists query planning, retrieval, citation-grounded synthesis, and evidence-sufficiency evaluation steps so failures can be reproduced during debugging or explained in interviews.
- Verified strict RED/GREEN with the focused trace-replay-plan test before running the full suite and linter.

### 2026-06-30

- Added `build_interview_objection_handling_markdown(...)` to turn demo batch metrics into concise answers for common interviewer objections.
- The table explains why this is more than ChatPDF, how faithfulness is checked, what happens when evidence is weak, and how the trace can be audited.
- Verified strict RED/GREEN with the focused objection-handling test before running the full suite and linter.

### 2026-06-29

- Added `build_interview_concept_map_markdown(...)` to turn demo trace metrics into an interviewer-facing Agentic RAG concept map.
- The concept map ties query decomposition, evidence grounding, faithfulness evaluation, and the answer safety gate to concrete trace counts so the project is easier to explain beyond a generic ChatPDF demo.
- Verified strict RED/GREEN with the focused concept-map test before running the full suite and linter.

### 2026-06-29

- Added `build_interview_demo_script_markdown(...)` to turn demo traces into a concise interview walkthrough script.
- The script sequences a ready grounded-answer trace before a blocked risky trace, so interviewers can see both citation-grounded delivery and evidence-gate failure handling.
- Verified strict RED/GREEN with the focused demo-script test before running the full suite and linter.

### 2026-06-28

- Added `build_interview_risk_register_markdown(...)` to summarize demo traces as a risk register with low/medium/high risk, concrete mitigations, and interview framing.
- The report distinguishes sufficient evidence, no retrieved evidence, and unsupported-claim risks so interviewers can see how the agent handles weak evidence instead of hiding it.
- Verified strict RED/GREEN with the focused risk-register test before running the full suite and linter.

### 2026-06-28

- Added `build_interview_followup_questions_markdown(...)` to turn a trace into common interviewer follow-up questions with evidence-backed answer angles.
- The report explains why the system is more than ChatPDF, how grounding is verified, and how weak evidence triggers retry or human review.
- Verified strict RED/GREEN with the focused follow-up-question test before running the full suite and linter.

### 2026-06-27

- Added `build_interview_readiness_scorecard_markdown(...)` to summarize multiple demo traces as an interview readiness scorecard.
- The scorecard counts ready-vs-needs-review traces, evidence items, unsupported claims, and gives a plain-English interview angle for each trace.
- Verified strict RED/GREEN with the focused scorecard test before running the full suite.

### 2026-06-27

- Added `build_interview_walkthrough_markdown(...)` to turn one traceable RAG run into a concise interview walkthrough.
- The walkthrough explains planned retrieval, evidence selection, citation-grounded answer checking, delivery readiness, and the Agentic RAG framing in plain language.
- Verified strict RED/GREEN with the focused interview-walkthrough test before running the full suite and linter.

### 2026-06-26

- Added `build_interview_evidence_packet_markdown(...)` to turn one traceable RAG run into a concise interview/demo packet.
- The packet highlights question, evidence status, cited sources, unsupported-claim count, and plain-English Agentic RAG talking points for portfolio pages or interview notes.
- Verified strict RED/GREEN with the focused interview-evidence-packet test before running the full suite and linter.

### 2026-06-26

- Added `build_human_review_revision_queue_markdown(...)` to list only non-approved reviewer decisions after the answer approval gate.
- The queue maps `retry_retrieval` to rerun retrieval/evaluation and `escalated` to human escalation review, while keeping labels and reasons Markdown-safe for dashboard/report exports.
- Verified strict RED/GREEN with the focused revision-queue test before running the full suite and linter.

### 2026-06-25

- Added `build_human_review_decision_summary_markdown(...)` to render reviewer outcome summaries as dashboard-ready Markdown.
- The report shows total/approved/revision decisions, escaped blocked trace labels, and per-outcome counts so audit metrics can be copied directly into demo pages.
- Verified strict RED/GREEN with the focused decision-summary Markdown test before running the full suite and linter.

### 2026-06-25

- Added `build_human_review_decision_summary(...)` to count final reviewer decisions after the answer approval gate.
- The summary reports total decisions, approved vs revision decisions, per-outcome counts, and blocked trace labels so the dashboard can show human-review audit outcomes at a glance.
- Verified strict RED/GREEN with the focused decision-summary test before running the full suite and linter.

### 2026-06-24

- Added `build_human_review_decision_log_markdown(...)` to record final human-review decisions after the evidence gate recommends approve/block.
- The decision log shows evidence status, gate recommendation, reviewer outcome, and an audit note, while keeping labels/notes Markdown-safe for portfolio/demo exports.
- Verified strict RED/GREEN with the focused decision-log test before running the full suite and linter.

### 2026-06-24

- Added `build_human_review_priority_board_markdown(...)` to order blocked/not-yet-evaluated traces by review risk for human-in-the-loop answer safety.
- The priority board filters out already-approved traces, promotes unsupported-claim cases above no-evidence and not-yet-evaluated cases, and keeps trace labels/reasons Markdown-safe for dashboard/report exports.
- Verified strict RED/GREEN with the focused priority-board test before running the full suite and linter.

### 2026-06-23

- Added `build_human_review_checklist_markdown(...)` to turn blocked/not-yet-evaluated traces into concrete reviewer checklist tasks.
- The checklist filters out already-approved traces, separates missing evaluation from insufficient evidence, and keeps trace labels Markdown-safe for dashboard/report exports.
- Verified strict RED/GREEN with the focused checklist test before running the full suite and linter.

### 2026-06-23

- Added `build_human_review_action_summary_markdown(...)` to group blocked/not-yet-evaluated traces by the exact reviewer next action.
- The report counts actions such as `inspect_retrieval_and_revise_answer` and `run_evidence_evaluation`, with Markdown-safe trace labels for dashboard/report exports.
- Verified strict RED/GREEN with the focused action-summary test before running the full suite and linter.

### 2026-06-22

- Added `build_human_review_workload_markdown(...)` to format human-review workload counts as a concise Markdown section for dashboard/report pages.
- The report shows approved vs blocked traces, not-evaluated and insufficient-evidence breakdowns, evidence items/unsupported claims needing review, and Markdown-safe blocked trace labels.
- Verified strict RED/GREEN with the focused workload-Markdown test before running the full suite and linter.

### 2026-06-22

- Added `build_human_review_workload_summary(...)` to count approved, blocked, insufficient-evidence, and not-yet-evaluated traces for reviewer workload planning.
- The summary also totals evidence items needing review, unsupported claims, and blocked trace labels so a dashboard can show human-in-the-loop review volume at a glance.
- Verified strict RED/GREEN with the focused workload-summary test before running the full suite and linter.

### 2026-06-21

- Added `build_human_review_escalation_brief_markdown(...)` to summarize blocked/not-evaluated traces for reviewer handoff.
- The brief filters out approved traces, counts blocked items, evidence items needing review, and unsupported claims, then renders a Markdown-safe reviewer action table.
- Verified strict RED/GREEN with the focused escalation-brief test before running the full suite and linter.

### 2026-06-21

- Added `build_human_review_queue_markdown(...)` to turn blocked answer traces into a reviewer-facing queue.
- The report filters out already-approved traces and lists only insufficient/not-evaluated cases with a review trigger, recommended reviewer action, and Markdown-safe reason.
- Verified strict RED/GREEN with the focused human-review queue test before running the full suite and linter.

### 2026-06-20

- Added `build_answer_approval_gate_markdown(...)` to turn evidence sufficiency into an explicit delivery gate for answer review.
- The gate approves citation-grounded answers only when evidence is sufficient, and blocks insufficient/not-evaluated traces for revision or evaluation before delivery.
- Verified strict RED/GREEN with the focused answer-approval gate test before running the full suite and linter.

### 2026-06-20

- Added `build_recovery_action_plan_markdown(...)` to turn insufficient or not-yet-evaluated traces into concrete recovery actions.
- The report maps no-evidence failures to retrieval retry, unsupported claims to answer revision/escalation, and unevaluated traces to faithfulness evaluation before delivery.
- Verified strict RED/GREEN with the focused recovery-action plan test before running the full suite and linter.

### 2026-06-19

- Added `build_evidence_decision_markdown(...)` to render an evidence decision log for multiple traces.
- The report marks sufficient traces as `ready_for_answer` and insufficient/not-evaluated traces as `retry_or_escalate`, preserving the plain-English sufficiency reason with Markdown-safe escaping.
- Verified strict RED/GREEN with the focused evidence-decision Markdown test before running the full suite and linter.

### 2026-06-19

- Added `build_source_attribution_markdown(...)` to render retrieved sources as an attribution table with evidence ranks, answer-citation status, supported-claim counts, and top retrieval score.
- This makes source-level grounding easier to inspect during interviews: reviewers can see which retrieved sources actually influenced the final citation-grounded answer versus unused context.
- Verified strict RED/GREEN with the focused source-attribution Markdown test before running the full suite and linter.

### 2026-06-18

- Added `build_evidence_sufficiency_gap_report_markdown(...)` to render insufficient traces as a focused Markdown debug table.
- The report filters out sufficient traces and shows trace label, sufficiency status, evidence count, unsupported-claim count, and the plain-English insufficiency reason with Markdown-safe escaping.
- Verified strict RED/GREEN with the focused evidence-sufficiency gap report test before running the full suite and linter.

### 2026-06-18

- Added `build_unsupported_citation_report_markdown(...)` to render unsupported citation failures as a focused Markdown debug table.
- The report filters to questions with invalid citations and shows citation-support ratio, unsupported cited source IDs, and retrieved evidence sources with Markdown-safe escaping.
- Verified strict RED/GREEN with the focused unsupported-citation report test before running the full suite and linter.

### 2026-06-17

- Added `build_retrieval_gap_report_markdown(...)` to render retrieval-coverage failures as a focused Markdown debug table.
- The report filters to benchmark questions with missing expected sources and shows coverage, missing sources, and retrieved sources with Markdown-safe escaping.
- Verified strict RED/GREEN with the focused retrieval-gap report test before running the full suite and linter.

### 2026-06-17

- Added `build_failure_diagnosis_markdown(...)` to render question-level RAG failure diagnoses as a Markdown table.
- The report shows each benchmark question, its failure mode, and the plain-English reason, with Markdown table escaping for interview/demo exports.
- Verified strict RED/GREEN with the focused failure-diagnosis Markdown test before running the full suite and linter.

### 2026-06-16

- Added `build_retrieval_plan_markdown(...)` to render planned retrieval queries, why each query exists, and which sources each step retrieved.
- This makes query decomposition explainable for interview/demo pages instead of hiding it inside the trace object.
- Verified strict RED/GREEN with the focused retrieval-plan Markdown test before running the full suite and linter.

### 2026-06-16

- Added `build_retrieval_retry_report_markdown(...)` to turn weak-evidence recovery attempts into a demo-ready Markdown table.
- The report shows the original failed query, rewritten retry query, before/after retrieved sources, and whether the retry recovered evidence.
- Verified strict RED/GREEN with the focused retry-report test before running the full suite and linter.

### 2026-06-15

- Added a weak-evidence retry path in `answer_question(...)`: when an initial retrieval returns no evidence, the trace can append a rewritten retrieval query and run a second retrieval step.
- Added deterministic hallucination-control rewriting from weak queries to `unsupported claims evidence`, so the pipeline can recover evidence about faithfulness checks instead of stopping after the first miss.
- Verified strict RED/GREEN with the focused retry test before running the full suite (`39 passed`) and `ruff check .`.

### 2026-06-15

- Added `build_unsupported_claim_report_markdown(...)` to render unsupported answer claims as a focused Markdown review table.
- The report filters to failed faithfulness checks, lists the exact unsupported claim, and keeps supporting-source display explicit as `none` when no evidence supports it.
- Verified RED/GREEN with the focused unsupported-claim review test before running the full suite and linter.

### 2026-06-14

- Added `build_citation_evidence_map_markdown(...)` to render citation-to-evidence links as a Markdown table for demo/review pages.
- The export shows each answer citation, whether the source was retrieved, the evidence rank, chunk ID, supporting-claim count, and evidence snippet.
- Verified RED/GREEN with the focused citation-map Markdown test before running the full suite and linter.

### 2026-06-14

- Added `build_trace_interview_summary_markdown(...)` to turn one RAG trace into a concise interviewer-facing summary.
- The summary connects Agentic RAG concepts to concrete run metrics: planned queries, retrieved evidence, retrieval coverage, citation support, unsupported claims, and evidence sufficiency.
- Verified RED/GREEN with the focused interview-summary test before running the full suite and linter.

### 2026-06-13

- Added a `synthesize_answer` event to `build_trace_timeline_events(...)`, so trace timelines now show when retrieved evidence becomes a citation-grounded answer before sufficiency evaluation.
- Added a strict TDD regression test proving answer synthesis appears between retrieval and evaluation with evidence source/chunk IDs preserved.
- Verified RED/GREEN with the focused synthesis-timeline test, then reran the related timeline tests.

### 2026-06-13

- Added `build_trace_timeline_markdown(...)` to render ordered RAG timeline events as a Markdown table for demo/report pages.
- Added a strict TDD regression test proving query planning, retrieval, and evidence-sufficiency events render in deterministic order with sources and chunk IDs.
- Verified RED/GREEN with the focused timeline-Markdown test, then ran the full test suite (`34 passed`) and `ruff check .`.

### 2026-06-12

- Hardened `build_trace_report_markdown(...)` so queries, source IDs, chunk IDs, and snippets containing `|` are escaped before entering Markdown tables.
- Added a strict TDD regression test proving trace-report exports remain valid when retrieved evidence contains table separator characters.
- Verified RED/GREEN with the focused trace-report escaping test, then ran the full test suite (`33 passed`) and `ruff check .`.

### 2026-06-12

- Hardened `build_observability_dashboard_markdown(...)` so labels, values, and details containing `|` are escaped before being inserted into Markdown tables.
- Added a strict TDD regression test proving dashboard/demo exports remain valid when card text contains table separator characters.
- Verified RED/GREEN with the focused Markdown escaping test, then ran the full test suite (`32 passed`) and `ruff check .`.

### 2026-06-11

- Added `build_observability_dashboard_markdown(...)` to render grouped observability sections as Markdown tables for a future demo page or report export.
- Added a strict TDD regression test proving health, evaluation, latency, and retrieval-error sections produce deterministic Markdown output.
- Verified RED/GREEN with the focused dashboard-Markdown test, then ran the full test suite (`31 passed`) and `ruff check .`.

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
