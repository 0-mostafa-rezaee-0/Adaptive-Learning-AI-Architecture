ROLE
You are a principal AI architect tasked with designing an incremental, evidence-backed roadmap for an adaptive learning system at scale. You must compare alternative solution stacks and return a single, concrete plan with milestones, metrics, risks, and how to measure learning gains—not just engagement.

CONTEXT
- Product: When a learner asks a question, the system must (1) suggest the best, shortest learning resource(s) (video/PDF) that directly answer the question, (2) deliver precise formative questions, and (3) assess whether learning occurred (growth over time).
- Data reality: We DO have historical Q&A/assessment data. We do NOT have historical data for content recommendation (video/PDF suggestions).
- Constraint: No hour-long videos for simple questions. Favor micro-learning: short clips, short PDF snippets, and extracted segments.
- Team: We have agentic-AI expertise, data scientists, and can do fine-tuning and reinforcement learning if justified.
- Goal: A stepwise plan where after each milestone we can demonstrate measurable progress toward “better learning outcomes,” not just nicer UX.

WHAT TO PRODUCE (STRUCTURE YOUR OUTPUT EXACTLY IN THESE 10 SECTIONS)
1) Executive summary (<=10 bullets): The chosen end-to-end approach and why it beats alternatives for our constraints.
2) Architecture options (compare at least 3): 
   A) RAG-first + reranking + agentic orchestration; 
   B) Lightweight fine-tuning (LoRA/PEFT) for pedagogy/style + RAG; 
   C) RL/Bandits for content selection on top of a RAG baseline; 
   D) (optional) Fully fine-tuned task-specific models (question generation, grading) with small LMs.
   For each option: components, infra, cost, latency, data needs, expected quality, risks.
3) Final recommended architecture (diagram + bullet list):
   - Retrieval: embedding model(s), chunking strategy (overlap, max tokens), metadata, hybrid search (BM25+dense), re-ranking (Cross-Encoder), MMR.
   - Content minimization: segment detection for videos (ASR + scene/chaptering) and PDFs (section detection), summarization to “micro-nuggets,” max length constraints (e.g., video 2–5 min; PDF 1–3 pages).
   - Pedagogical layer: question generation (Bloom levels), hints, worked examples, distractor quality, rubric-based grading, self-consistency decoding or verifier.
   - Assessment & learning analytics: ability estimation (IRT-style), question difficulty calibration, learning gain measurement over sessions.
   - Agentic orchestration: planner → retriever → selector → pedagogy tools → evaluator → next-action.
4) Data plan (cold-start to flywheel):
   - How to start recommendations with NO historical resource labels (zero/low-shot): heuristics, metadata filters, ASR+semantic coverage, weak labels, teacher-in-the-loop.
   - Rapid dataset creation: offline labeling protocols, rubrics for “best minimal resource,” inter-rater agreement, active learning loops.
   - Using existing Q&A/assessment logs to pretrain/refine question generation, grading rubrics, and difficulty estimation.
5) Metrics & evaluation (must be executable per milestone):
   - Retrieval/selection: nDCG@k, Recall@k, Coverage, Time-to-first-useful-resource.
   - Minimality: median resource length, “overkill rate” (% of suggestions exceeding target length), compression ratio.
   - Question quality: expert rubric scores (clarity, alignment, Bloom level), pass@k on canonical answers, factuality via reference-grounded checks.
   - Assessment quality: item discrimination (a), difficulty (b), guessing (c) if 3PL; item/test information; test-retest reliability; ability θ stability and standard error.
   - Learning outcomes: Δθ over time (IRT), mastery progression, normalized gain, and downstream task performance. Distinguish engagement (CTR, dwell) from learning.
   - Safety/accuracy: hallucination rate vs. reference, refusal/deferral accuracy when unsure.
6) Stepwise roadmap (8–12 weeks, shippable increments):
   - M1 (Weeks 1–2): RAG baseline with strict minimality constraints. Deliver offline eval + red team cases. Acceptance criteria with target nDCG/Recall and “overkill rate < X%”.
   - M2 (Weeks 3–4): Add cross-encoder reranking, video/PDF segmenter (ASR + sectionizer), JSON-structured outputs, evaluator that rejects non-minimal resources. Acceptance: improved nDCG, reduced median resource length, reduced hallucinations.
   - M3 (Weeks 5–6): Pedagogy tools v1 (question generator + rubric grader + hinting), calibrated with few-shot exemplars from existing Q&A; start IRT-lite (2PL) calibration using historical responses. Acceptance: rubric scores exceed baseline, stable item parameters on holdout.
   - M4 (Weeks 7–8): Bandit/RLHF for content selection under minimality + learning constraints (define reward below). Off-policy evaluation with IPS/DR estimators. Acceptance: uplift in Δθ proxy and reduced time-to-mastery without increasing overkill.
   - M5 (Weeks 9–10): Optional LoRA fine-tunes for pedagogy style & distractors; A/B vs. prompt-only. Acceptance: higher expert rubric scores and fewer grading inconsistencies at same latency/cost.
   - M6 (Weeks 11–12): Production-hardening: guardrails, safety filters, bias checks, PII/FERPA/GDPR compliance, telemetry dashboards; design for continuous recalibration of item parameters and bandit policies.
7) Reinforcement learning design (practical):
   - Define reward as multi-objective: R = w1*(learning gain proxy: Δθ or quiz correctness uplift) + w2*(minimality bonus: short/concise) − w3*(hallucination/irrelevance penalties) − w4*(latency/cost).
   - Start with contextual bandits (Thompson/UCB) on top of a strong RAG baseline. Use off-policy evaluation (IPS/DR), safety constraints (do-not-exceed resource length), and uncertainty-aware deferrals.
   - Escalate to RL only if bandits plateau and we have sufficient logged data and simulators.
8) Fine-tuning policy (when and how):
   - DO NOT fine-tune the general model at first. Fine-tune smaller task-specific modules later: (a) question generator on domain exemplars, (b) rubric grader, (c) distractor generator, (d) short-explainer style. Use LoRA/PEFT to retain upgrade agility. Define migration plan for model upgrades (adapter transfer).
   - Entry criteria for FT: ≥ N_k high-quality labeled exemplars per task; demonstrated plateau of prompt-only; projected inference savings and consistency gains; governance approval.
9) Risks & mitigations:
   - Cold-start for content recs → mitigate via heuristics + weak labels + teacher review.
   - Over-long resources → hard caps, rank by “coverage-per-minute.”
   - Hallucinations → retrieval-grounded verification, answerability checkers, refusal paths.
   - Misaligned difficulty → IRT recalibration cadence; anchor items; drift detection.
   - Privacy/PII → strict logging controls, role-based access, on-prem/virtual private deployment options.
10) Deliverables per milestone:
   - Reproducible notebooks/pipelines, data cards & model cards, prompt libraries, evaluation reports with dashboards (IR metrics, IRT charts, learning gain trends), A/B test design, decision memos to progress to next milestone.

ADDITIONAL IMPLEMENTATION DETAILS (INCLUDE IN YOUR PLAN)
- Video/PDF minimality: Define “sufficiency score” (semantic coverage vs. question) / duration (or pages). Prefer top-k segments, not whole assets.
- ASR & segmentation: lightweight chaptering + semantic similarity to queries; align timestamps to produce 1–3 minute clips.
- Reranking: bi-encoder retrieval + cross-encoder reranker; MMR to diversify.
- Guardrails: source citations, contradiction checks, JSON schema validation, fallbacks.
- Telemetry: per-user learning trajectory (θ, SE), item parameter drift, bandit policy diagnostics, safety incidents.
- Stakeholder views: educator dashboard for oversight, learner explanations (“why this resource,” “why this question,” “what’s next”).

OUTPUT FORMAT
Return:
- A table comparing the options (A/B/C/D) across cost, latency, data needs, cold-start viability, expected learning impact, team fit.
- A single recommended architecture with a clear diagram (textual is fine).
- A 12-week roadmap with milestone-level acceptance criteria and exact metrics.
- A concrete RL/bandit reward definition and offline evaluation recipe.
- A fine-tuning decision checklist and migration plan for model upgrades.
- A short “first 14 days” task list that is executable by a small team.

CONSTRAINTS
- Keep recommendations model/vendor-agnostic where possible; suggest specific models only as exemplars.
- Prioritize steps that produce measurable learning improvements quickly.
- Favor approaches that keep us agile as foundation models evolve (adapters over full FT).
