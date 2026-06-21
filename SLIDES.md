# Slide skeleton — AI Newsroom (multi-agent SLM study)

> Skeleton only: each slide's title + the points it must make + where assets go.
> A **document the professor reads** (no live talk) → each slide stands on its own.
> Numbers are the final n=30 run (`results/scores.csv`). **Lead with faithfulness +
> win-rate**; judge_avg spread is small, so keep it secondary.
>
> **Placeholders:** `<pic N: file>` = drop the figure here · `<table: file>` = drop the table here.
> The author inserts the assets. **The demo video is a separate deliverable — NOT in this deck.**
>
> Assets used: `<pic 1>` architecture (author-drawn) · `<pic 2>` `results/fig4_faithfulness.png`
> · `<pic 3>` `results/fig5_feedback.png` · `<table>` `results/scores_table.tex`.
> Optional extras (only if space): `fig1_rubric.png`, `fig2_quality_cost.png`, `fig3_winrate.png`.

---

## 1. Title
- **AI Newsroom: How far does multi-agent decomposition + self-correction take a small LM?**
- A measured study (not a product). Course / name / date.
- Hook: *a small open-source LM, split into role agents — can it close the gap to a large LLM?*

## 2. Research question & motivation
- SLMs are cheap/local but weaker than large LLMs.
- **RQ:** does **role decomposition** + **self-correction** recover large-LLM quality — and at what cost?
- Testbed: automated newsletter generation (fuse 5 source articles → 1 article).

## 3. System — the newsroom
- 4 agents: **Scout** (collect) → **Reader** (parallel 3-line summaries) → **Writer** (fuse) → **Editor** (score + feedback).
- **Editor→Writer feedback loop** (K≤2): re-write if below threshold ← the studied "new method".
- `<pic 1: architecture diagram>`  (author-drawn; layout in PROJECT_PLAN §3)

## 4. Setup & metrics
- **Systems (identical frozen inputs):** `api` Gemini-3.5-flash (ceiling) · `single_slm` Llama-3.2-3B (floor) · `multi_agent` · `multi_agent_fb` (+loop).
- **Data:** HackerNews (tech) + arXiv (research), 30 newsletters.
- **Metrics:** LLM-judge rubric (4 axes) · pairwise win-rate · NLI faithfulness (judge-free) · latency/iters.
- **Bias control:** judge = independent **Qwen2.5-14B** (≠ generator/ceiling family → less self-preference); pairwise order swapped (position bias).

## 5. Result 1 — decomposition helps faithfulness ✅
- single_slm **25.6%** → multi_agent **32.7%** faithful — **above the API ceiling (31.2%)**.
- Per-article summarization (Reader) reduces hallucination vs one-shot fusion.
- `<pic 2: fig4_faithfulness.png>`

## 6. Result 2 — self-correction hurts ❌
- Adding the loop: faithful 32.7% → **25.9%** (back to floor), judge 3.21 → **3.14**, latency **2.4×** (14s→34s).
- The 3B reviser can't act on its own critique → revisions drift / degrade.
- Tellingly, the **Editor's own score stays flat** (3.53→3.52) though it fired on 27/30 → it can't see its own regression.
- `<pic 3: fig5_feedback.png>`  (faithfulness ↓ and latency ↑ after the loop)

## 7. Results table
- All systems × metrics; api is a clear ceiling (judge 3.34, win-rate 66.7%).
- Reminder: judge_avg spread is small → read faithfulness + win-rate first.
- `<table: results/scores_table.tex>`  (best per column bolded)

## 8. Discussion — insight & limitations
- **Key insight:** self-correction's payoff is **gated by base-model capability** — a weak model is a weak self-critic. Decomposition (structural) helps; iterative self-revision (capability-dependent) does not, at 3B.
- **Limitations:** n=30, single judge, small judge_avg spread (no significance test); faithfulness is an NLI proxy; conciseness is everyone's weakest axis (task-inherent).
- **Future:** stronger/separate reviser model; per-agent specialized models; human eval.

## 9. Reproducibility & takeaway
- Free official APIs (HackerNews, arXiv); eval set frozen → all systems on identical inputs; one `config.yaml` swaps any model; `python main.py all` reproduces end-to-end.
- **Takeaway:** decomposition lifts small-LM faithfulness above a large-LLM ceiling; naive self-correction needs a capable base model.
