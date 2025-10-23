# ROLE
You are a principal AI architect tasked with designing an incremental, evidence-backed roadmap for an adaptive learning system at scale. You must compare alternative solution stacks and return a single, concrete plan with milestones, metrics, risks, and how to measure learning gains—not just engagement.

# CONTEXT
## Product Requirements
When a learner asks a question, the system must:
1. Suggest the best, shortest learning resource(s) (video/PDF) that directly answer the question
2. Deliver precise formative questions
3. Assess whether learning occurred (growth over time)

## Data Reality
- ✅ We DO have historical Q&A/assessment data
- ❌ We do NOT have historical data for content recommendation (video/PDF suggestions)

## Constraints
- No hour-long videos for simple questions
- Favor micro-learning: short clips, short PDF snippets, and extracted segments

## Team & Capabilities
- Agentic-AI expertise available
- Data scientists on team
- Can do fine-tuning and reinforcement learning if justified

## Educational Context
- **Level:** Higher education
- **Learner Type:** Students using an AI agent for learning
- **Learning Flow:** Student asks question → Agent searches all company materials (books, question banks, PDFs, videos, etc.) → Agent suggests smallest relevant learning materials → Student learns → Agent asks formative questions for learning and evaluation → Based on results, agent prepares next small learning material → Process repeats
- **Success Metrics:** Agent effectiveness measured by increased student engagement time, more practice opportunities, and better learning outcomes

## Scale & Deployment
- **Target Scale:** Millions of learners (Pearson's global client base)
- **Current Phase:** Proof of concept/prototype
- **Scaling Strategy:** Start with prototype, scale if successful
- **Platform Preferences:** No specific LMS platform or delivery modality preferences at this stage
- **Integrations:** No third-party integration requirements at this stage

## Goal
A stepwise plan where after each milestone we can demonstrate measurable progress toward "better learning outcomes," not just nicer UX.

# WHAT TO PRODUCE (STRUCTURE YOUR OUTPUT EXACTLY IN THESE 10 SECTIONS)

## 1. Executive Summary (≤10 bullets)
The chosen end-to-end approach and why it beats alternatives for our constraints.

## 2. Architecture Options (compare at least 3)
- **Option A:** RAG-first + reranking + agentic orchestration
- **Option B:** Lightweight fine-tuning (LoRA/PEFT) for pedagogy/style + RAG  
- **Option C:** RL/Bandits for content selection on top of a RAG baseline
- **Option D:** (optional) Fully fine-tuned task-specific models (question generation, grading) with small LMs

For each option: components, infra, cost, latency, data needs, expected quality, risks.

## 3. Final Recommended Architecture (diagram + bullet list)
- **Retrieval:** embedding model(s), chunking strategy, metadata, hybrid search (BM25+dense), re-ranking (Cross-Encoder), MMR
- **Content minimization:** segment detection for videos (ASR + scene/chaptering) and PDFs (section detection), summarization to "micro-nuggets," max length constraints
- **Pedagogical layer:** question generation (Bloom levels), hints, worked examples, distractor quality, rubric-based grading
- **Assessment & learning analytics:** ability estimation (IRT-style), question difficulty calibration, learning gain measurement over sessions
- **Agentic orchestration:** planner → retriever → selector → pedagogy tools → evaluator → next-action

## 4. Data Plan (cold-start to flywheel)
- How to start recommendations with NO historical resource labels: heuristics, metadata filters, ASR+semantic coverage, weak labels, teacher-in-the-loop
- Rapid dataset creation: offline labeling protocols, rubrics for "best minimal resource," inter-rater agreement, active learning loops
- Using existing Q&A/assessment logs to pretrain/refine question generation, grading rubrics, and difficulty estimation

## 5. Metrics & Evaluation (must be executable per milestone)
- **Retrieval/selection:** nDCG@k, Recall@k, Coverage, Time-to-first-useful-resource
- **Minimality:** median resource length, "overkill rate" (% of suggestions exceeding target length), compression ratio
- **Question quality:** expert rubric scores (clarity, alignment, Bloom level), pass@k on canonical answers, factuality via reference-grounded checks
- **Assessment quality:** item discrimination (a), difficulty (b), guessing (c) if 3PL; item/test information; test-retest reliability; ability θ stability and standard error
- **Learning outcomes:** Δθ over time (IRT), mastery progression, normalized gain, and downstream task performance. Distinguish engagement (CTR, dwell) from learning
- **Safety/accuracy:** hallucination rate vs. reference, refusal/deferral accuracy when unsure

## 6. Stepwise Roadmap (8–12 weeks, shippable increments)
- **M1 (Weeks 1–2):** RAG baseline with strict minimality constraints. Deliver offline eval + red team cases
- **M2 (Weeks 3–4):** Add cross-encoder reranking, video/PDF segmenter (ASR + sectionizer), JSON-structured outputs
- **M3 (Weeks 5–6):** Pedagogy tools v1 (question generator + rubric grader + hinting), calibrated with few-shot exemplars
- **M4 (Weeks 7–8):** Bandit/RLHF for content selection under minimality + learning constraints
- **M5 (Weeks 9–10):** Optional LoRA fine-tunes for pedagogy style & distractors; A/B vs. prompt-only
- **M6 (Weeks 11–12):** Production-hardening: guardrails, safety filters, bias checks, compliance, telemetry dashboards

## 7. Reinforcement Learning Design (practical)
- Define reward as multi-objective: R = w1*(learning gain proxy: Δθ or quiz correctness uplift) + w2*(minimality bonus: short/concise) − w3*(hallucination/irrelevance penalties) − w4*(latency/cost)
- Start with contextual bandits (Thompson/UCB) on top of a strong RAG baseline. Use off-policy evaluation (IPS/DR), safety constraints
- Escalate to RL only if bandits plateau and we have sufficient logged data and simulators

## 8. Fine-tuning Policy (when and how)
- DO NOT fine-tune the general model at first. Fine-tune smaller task-specific modules later: (a) question generator on domain exemplars, (b) rubric grader, (c) distractor generator, (d) short-explainer style
- Use LoRA/PEFT to retain upgrade agility. Define migration plan for model upgrades (adapter transfer)
- Entry criteria for FT: ≥ N_k high-quality labeled exemplars per task; demonstrated plateau of prompt-only; projected inference savings and consistency gains

## 9. Risks & Mitigations
- Cold-start for content recs → mitigate via heuristics + weak labels + teacher review
- Over-long resources → hard caps, rank by "coverage-per-minute"
- Hallucinations → retrieval-grounded verification, answerability checkers, refusal paths
- Misaligned difficulty → IRT recalibration cadence; anchor items; drift detection
- Privacy/PII → strict logging controls, role-based access, on-prem/virtual private deployment options

## 10. Deliverables per Milestone
- Reproducible notebooks/pipelines, data cards & model cards, prompt libraries, evaluation reports with dashboards, A/B test design, decision memos to progress to next milestone

# ADDITIONAL IMPLEMENTATION DETAILS (INCLUDE IN YOUR PLAN)
- **Video/PDF minimality:** Define "sufficiency score" (semantic coverage vs. question) / duration (or pages). Prefer top-k segments, not whole assets
- **ASR & segmentation:** lightweight chaptering + semantic similarity to queries; align timestamps to produce 1–3 minute clips
- **Reranking:** bi-encoder retrieval + cross-encoder reranker; MMR to diversify
- **Guardrails:** source citations, contradiction checks, JSON schema validation, fallbacks
- **Telemetry:** per-user learning trajectory (θ, SE), item parameter drift, bandit policy diagnostics, safety incidents
- **Stakeholder views:** educator dashboard for oversight, learner explanations ("why this resource," "why this question," "what's next")

# OUTPUT FORMAT REQUIREMENTS

**IMPORTANT:** Output must be a complete, executable .tex file with full LaTeX syntax including:
- Complete LaTeX document structure with preamble
- Tables, figures, and environments
- Single-column format
- Proper LaTeX formatting for all content

## Required Structure:
1. **Main Title** (using LaTeX title formatting)
2. **Table of Contents** (using \tableofcontents)
3. **Content organized in numbered sections** (using \section, \subsection, etc.)

## Expected Content:
- A table comparing options (A/B/C/D) across cost, latency, data needs, cold-start viability, expected learning impact, team fit
- A single recommended architecture with a clear diagram (textual is fine)
- A 12-week roadmap with milestone-level acceptance criteria and exact metrics
- A concrete RL/bandit reward definition and offline evaluation recipe
- A fine-tuning decision checklist and migration plan for model upgrades
- A short "first 14 days" task list that is executable by a small team

**Note:** Do not ask for clarification on output format. Provide a complete, executable .tex file directly.

# CONSTRAINTS
- Keep recommendations model/vendor-agnostic where possible; suggest specific models only as exemplars
- Prioritize steps that produce measurable learning improvements quickly
- Favor approaches that keep us agile as foundation models evolve (adapters over full FT)
