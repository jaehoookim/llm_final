# AI Newsroom — A Multi-Agent SLM Pipeline for Automated Newsletters

**Course:** CAS4133 (Large Language Models) — Final Project
**Deadline:** 2026-06-21 23:59 (submit *summary slides* + *implementation code*)
**Compute:** Vessl GPU (single RTX 3090, 24 GB VRAM) for all local models; external API for the large-LLM upper-bound baseline only.

---

## 1. One-Line Thesis (the research framing)

> **Research question:** *How far does role decomposition (a multi-agent pipeline of small open-source models) compensate for the quality limits of a single small language model (SLM) on a real generation task — and at what efficiency cost?*

We use **automated newsletter generation** as the testbed. The deliverable is **not** "a system that makes nice newsletters"; it is a **measured study** of *how far* decomposition and self-correction close the gap — and where they fail. The measured answer is **differentiated**: role decomposition improves faithfulness (above even the large-LLM ceiling) at parity on judged quality, while a naive Editor→Writer self-correction loop **degrades** a 3B model's output. The headline insight: **self-correction's payoff is gated by base-model capability** — a weak model is a weak self-critic.

This framing is deliberate: it converts an engineering demo into a research project, which is what unlocks the quantitative-evaluation and research-problem points below.

---

## 2. Scoring Map (target: 22 / 20)

| Rubric item | Pts | How this project satisfies it |
|---|---|---|
| Slides + code submitted on time | 10 | Schedule in §9; hard buffer on Day 6. |
| New task/method not covered in lecture | +3 | Multi-agent "newsroom" **with an Editor→Writer self-correction feedback loop** (verification + regeneration), studied as a controlled comparison. Not the lecture examples (OPRO / jailbreak). The method is implemented and *measured*; we report where it helps (decomposition) and where it hurts (self-correction with a weak reviser). |
| Quantitative evaluation results | +3 | Three measured axes: (a) LLM-as-judge pointwise rubric scores, (b) LLM pairwise win-rate vs baselines, (c) faithfulness/hallucination rate + efficiency. Reported as tables + plots. |
| Demo (mp4) | +2 | Screen recording of one end-to-end run (collect → summarize → write → edit → newsletter). |
| Multiple models / datasets | +2 | **Models:** agent SLM(s) + single-SLM baseline + large-LLM API baseline + judge model (3–4 distinct models). **Datasets:** HackerNews (tech) + arXiv (research) — two domains. Either alone satisfies the item; we have both. |
| New research problem (extra credit) | +2 | The thesis in §1 is an explicitly stated, self-proposed research question with a measured, non-obvious answer (decomposition helps faithfulness; self-correction is gated by base-model capability). |

**Key point:** every bonus item is a *byproduct of doing the baseline comparison properly*, not an artificial add-on. We do not contort the architecture to tick boxes.

---

## 3. System Architecture

Four specialized agents in a pipeline, plus one feedback loop.

```
                         ┌─────────────────────────────────────────────┐
                         │                                             │
  [Sources]              ▼                                             │
  HackerNews API   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  arXiv API   ──▶  │  SCOUT   │──▶│  READER  │──▶│  WRITER  │──▶│  EDITOR  │──▶ Newsletter
                   │ collect  │   │ summarize│   │  draft   │   │ verify+  │
                   │ top-N    │   │ (parallel│   │  unify   │   │ title    │
                   │ items+URL│   │  3-line) │   │          │   │          │
                   └──────────┘   └──────────┘   └──────────┘   └────┬─────┘
                                                       ▲              │
                                                       │   reject &   │ quality
                                                       └──────────────┘ < threshold
                                                         feedback loop
```

- **Scout (collector):** pulls the day's top-N items (title, URL, body/comments) from HackerNews (and arXiv for the second domain). Pure API calls — no scraping.
- **Reader (summarizer):** produces a faithful **3-line summary** per item. Run **in parallel** (async / batched) — this is where the latency win comes from and it is *measured*.
- **Writer (lead author):** fuses the N summaries into one coherent article.
- **Editor (chief editor):** checks grammar/coherence, generates a title, and **scores the draft against a rubric**. If the score is below threshold (or it detects a factual inconsistency vs the source summaries), it returns structured feedback and the Writer regenerates — up to `K` iterations. **This loop is the "new method" contribution.**

### 3.1 Self-correction loop (the +3 "new method")
The Editor emits a JSON verdict: `{coherence, factuality, readability, pass: bool, feedback: str}`. On `pass=false`, the Writer is re-prompted with the feedback and the previous draft. Capped at `K=2` retries to bound latency. We **log every iteration** and report how many drafts converge and how quality changes across iterations. **Finding (n=30):** with a 3B reviser, the loop does *not* improve quality — judged quality and faithfulness both drop vs the no-feedback pipeline while latency ~2.4×. This is the project's key insight, not a bug: self-correction is gated by base-model capability.

---

## 4. Models (sized for a single 24 GB 3090)

| Role | Model (proposed) | Footprint | Notes |
|---|---|---|---|
| Agent SLMs (Scout logic / Reader / Writer / Editor) | `Llama-3.2-3B-Instruct` | ~6–7 GB (bf16) | Default: one shared 3B model with role-specific prompts. |
| (Optional) per-agent models | e.g. Reader=`Llama-3.2-3B`, Writer=`Qwen2.5-7B` | sequential load | Strengthens "multiple models" but adds memory juggling — **optional, not required**. |
| **Baseline A — lower bound** | single SLM, monolithic prompt (`Llama-3.2-3B`) | shared | "Do everything in one call." Isolates the value of decomposition. |
| **Baseline B — upper bound** | large LLM via **external API** (default `gemini-3.5-flash`; Claude also supported) | API | Reference ceiling. Small eval set ⇒ negligible cost. |
| **Judge** | `Qwen2.5-14B-Instruct` (4-bit) | ~9–10 GB on 3090 | Different family + larger than the 3B generators ⇒ mitigates self-preference bias. |

**Bias control:** the judge (Qwen-14B) is a *different family and larger* than the generators (Llama-3B). For pairwise judging we also randomize A/B order and average both directions to cancel position bias. (Optional cross-check: also judge with the API model and report agreement.)

---

## 5. Datasets

| Dataset | Source | Domain | Why |
|---|---|---|---|
| Tech news | HackerNews **official Firebase API** (free, no auth) | Technology | Primary; stable, no scraping risk. |
| Research feed | arXiv API (cs.CL / cs.AI recent) | Research papers | Second domain ⇒ tests whether decomposition helps consistently; satisfies "multiple datasets". |

**Evaluation set size:** 15–20 newsletter inputs per domain (each input = a fixed set of 5 source items). This is enough for clear trends; no large benchmark or significance testing required. Inputs are **collected once and frozen** to a local JSON so all systems run on identical data (reproducibility).

---

## 6. Evaluation Protocol

Three axes. All produce numbers for the report.

### 6.1 LLM-as-Judge — pointwise rubric (1–5)
Judge scores each newsletter on `factuality`, `coherence`, `readability`, `conciseness`. Output constrained to JSON. Report per-system mean per axis.

### 6.2 LLM-as-Judge — pairwise win-rate
For each input, judge compares **[multi-agent SLM]** vs **[single-SLM baseline]** and vs **[large-LLM baseline]**: Win / Tie / Lose. Average both A/B orderings. Report win-rate %. (Most human-aligned signal.)

### 6.3 Faithfulness / hallucination (reference-based, objective)
Because we *hold the source articles*, we can measure factuality without a judge:
- **NLI entailment:** source = premise, each generated sentence = hypothesis, scored by an off-the-shelf NLI model (`cross-encoder/nli-deberta-v3-base`). The source is split into **overlapping windows** so a sentence about *any* article (not just the first ~500 tokens) can be matched against its evidence. Report % entailed (faithful) sentences. *(This is the `faithful_pct` column in `scores.csv`.)*
- *Hypothesis:* the single SLM, fusing 5 articles in one shot, hallucinates more; per-article decomposition (Reader) plus the Editor→Writer feedback loop reduces it. Proven by the number.
- *(Optional extension, not in the final table: count named entities / numbers in the output but absent from the source.)*

### 6.4 Efficiency (free objective numbers)
| Metric | How | In `scores.csv` |
|---|---|---|
| End-to-end latency (s) | `time.perf_counter()` per system; shows Reader parallelism benefit | `latency_s` |
| Revision iterations | Editor→Writer loops actually taken | `avg_iters` |
| Total tokens / peak GPU (GB) | tokenizer count / `torch.cuda.max_memory_allocated()` | *optional, not reported* |

---

## 7. Experiment Design & Results Template

Systems compared: **(1)** large-LLM baseline (ceiling), **(2)** single-SLM baseline (floor), **(3)** multi-agent SLM — *no* feedback loop, **(4)** multi-agent SLM — *with* feedback loop (full proposal).

(Columns below mirror `scores.csv`: 4 rubric axes → `judge_avg`, `winrate_vs_floor_pct`, `faithful_pct`, `latency_s`, `avg_iters`.)

| System | Judge avg (1–5) | Win-rate vs floor % | Faithful % | Latency (s) | Avg iters |
|---|---|---|---|---|---|
| Large LLM (API, ceiling) | — | — | — | — | — |
| Single SLM (floor) | — | — | — | — | — |
| Multi-agent SLM | — | — | — | — | — |
| **+ feedback loop (ours)** | — | — | — | — | — |

Plots: (a) bar chart of judge scores per system per domain; (b) quality-vs-cost scatter (judge score vs latency); (c) quality across feedback-loop iterations.

**Headline story to land:** *"Role decomposition lifts faithfulness above even the large-LLM ceiling (single-SLM 25.6% → multi-agent 32.7%, API 31.2%) at parity on judged quality — but naive Editor→Writer self-correction with a 3B reviser hurts (judged quality and faithfulness drop, latency 2.4×). Self-correction's payoff is gated by base-model capability."* (Lead with faithfulness + pairwise win-rate; judge_avg differences are small and reported as secondary.)

---

## 8. Repository Structure

```
finalproject/
├── PROJECT_PLAN.md            # this file
├── README.md
├── requirements.txt
├── config.yaml                # model ids, N, K, thresholds, paths
├── data/
│   ├── collect.py             # Scout: HackerNews + arXiv API → frozen JSON
│   └── eval_set/              # frozen inputs (tech.json, research.json)
├── src/
│   ├── models.py              # model loading (vLLM/transformers), unload helpers
│   ├── agents/
│   │   ├── scout.py
│   │   ├── reader.py          # parallel summarization
│   │   ├── writer.py
│   │   └── editor.py          # scoring + feedback verdict
│   ├── pipeline.py            # orchestration + self-correction loop
│   └── baselines.py           # single-SLM + large-LLM-API runners
├── eval/
│   ├── judge.py               # pointwise + pairwise (bias controls)
│   ├── faithfulness.py        # NLI + hallucination count
│   ├── efficiency.py          # latency / tokens / GPU mem
│   └── run_all.py             # produces results tables + plots
├── results/                   # tables (csv), figures (png), logs
├── slides/                    # summary slides (export to pdf)
└── demo/                      # demo.mp4
```

---

## 9. Timeline (6 days — today is 2026-06-15)

| Day | Date | Goal |
|---|---|---|
| 1 | Jun 15–16 | Scaffold repo; `collect.py` (HN + arXiv), freeze eval set; load 3B model on Vessl; smoke test. |
| 2 | Jun 17 | Implement 4 agents + base pipeline (no loop). First end-to-end newsletter. |
| 3 | Jun 18 | Add feedback loop; implement both baselines (single-SLM, API large-LLM). |
| 4 | Jun 19 | Eval harness: judge (pointwise+pairwise), faithfulness, efficiency. |
| 5 | Jun 20 | Run all experiments on both domains; collect tables + plots. |
| 6 | Jun 21 | Slides + record demo.mp4; cleanup README; **submit with buffer** (no late penalty). |

---

## 10. Deliverables Checklist

- [ ] Summary slides (PDF) — problem, method, architecture, results tables/plots, conclusion.
- [ ] Implementation code (this repo), reproducible via `eval/run_all.py`.
- [ ] `demo.mp4` of one end-to-end run.
- [ ] `results/` with all tables and figures.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| External crawling eats time, scores 0 | **Use only official APIs** (HackerNews, arXiv); freeze data once. Source count does not affect grade. |
| 24 GB VRAM overflow with multiple models | Default to one shared 3B generator; load judge (14B-4bit) separately / sequentially; explicit `unload()` between stages. |
| Judge self-preference bias | Judge is a different, larger family (Qwen-14B vs Llama-3B); randomize pairwise order; optional API cross-check. |
| Text quality feels unmeasurable | Solved by the 3-axis protocol (§6): relative win-rate + decomposed rubric + reference-based faithfulness. |
| Time overrun near deadline | Feedback loop, per-agent models, and second domain are all *optional enhancements*; core (single domain, shared SLM, baselines, eval) is the must-ship MVP. |

---

## 12. Scope Discipline (must-ship vs nice-to-have)

**MVP (guarantees ~18–20 pts):** 4-agent pipeline (shared 3B) + single-SLM baseline + API baseline + judge + faithfulness + efficiency on **one** domain, with slides + demo.

**Enhancements (push to 22):** feedback loop (+3 method strength), second domain (datasets), per-agent distinct models, feedback-iteration plot.

Build the MVP first; add enhancements only after it runs end-to-end.
