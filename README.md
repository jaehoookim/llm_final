# AI Newsroom — Multi-Agent SLM Pipeline for Automated Newsletters

A small study: **does role decomposition across lightweight open-source models
(SLMs) recover most of a large LLM's newsletter quality — and at what cost?**
Newsletter generation is the testbed.

## What it does

Four agents in a pipeline, plus a self-correction loop:

```
Scout (collect.py) → Reader (parallel 3-line summaries)
                   → Writer (fuse into one article)
                   → Editor (score + title; if weak, send feedback back to Writer)
```

It then compares four systems with quantitative metrics:

| System | What |
|---|---|
| `api` | single large LLM (Gemini/Claude), one shot — **upper bound** |
| `single_slm` | single SLM, one monolithic prompt — **lower bound** |
| `multi_agent` | multi-agent SLM, no feedback loop |
| `multi_agent_fb` | multi-agent SLM + Editor→Writer loop — **our method** |

Metrics:
- **LLM-as-judge rubric** (1–5) on 4 axes — factuality, coherence, readability,
  conciseness. The Editor scores the *same* 4 axes so the feedback loop optimizes
  what the judge measures.
- **Pairwise win-rate** vs the SLM floor, order-swapped to cancel position bias.
- **Faithfulness** — fraction of generated sentences entailed by the source (NLI).
  The source is scored in overlapping windows so sentences about *any* article
  count, not just the first ~500 tokens.
- **Efficiency** — latency and revision iterations.

## Files

| File | Role |
|---|---|
| `config.yaml` | all model IDs, sizes, thresholds, paths |
| `collect.py` | Scout — builds the frozen eval set from HackerNews + arXiv APIs |
| `newsroom.py` | model wrappers + the 4 agents + feedback loop + baselines |
| `evaluate.py` | the metrics (pointwise judge, pairwise judge, NLI faithfulness) |
| `main.py` | CLI: `collect` / `generate` / `evaluate` / `all` |
| `utils.py` | config + `.env` loading + robust JSON extraction |
| `plot.py` | slide figures from `results/scores.csv` → `results/fig1..5.png` |
| `demo/showcase.py` | demo helper: live agent walkthrough / results replay / export |

## Setup (Vessl GPU, single 24GB 3090)

```bash
pip install -r requirements.txt
cp .env.example .env                  # then fill in your keys (gitignored)
```

Put all secrets in `.env` (auto-loaded by `utils.py`):
`HF_TOKEN` (Llama-3.2 / Qwen2.5 downloads) and `GEMINI_API_KEY` /
`ANTHROPIC_API_KEY` for the `api` upper-bound baseline. Plain `export`s or
`huggingface-cli login` also work — anything that lands in the environment.

The external baseline provider is chosen in `config.yaml` → `api_baseline`
(`provider: gemini | anthropic`, plus `model` and an optional inline `api_key`).
Leave `api_key: ""` to read it from the environment (`GEMINI_API_KEY` or
`ANTHROPIC_API_KEY`). Set `enabled: false` to skip the ceiling (no key needed).

## Run

```bash
python main.py collect     # → data/tech.json, data/research.json   (~1 min, no GPU)
python main.py generate    # → results/generations.json            (SLM on GPU)
python main.py evaluate    # → results/scores.csv + printed table   (judge on GPU)
python plot.py             # → results/fig1..5.png (slide figures)   (no GPU)
# or end-to-end (collect → generate → evaluate):
python main.py all
```

`generate` and `evaluate` are deliberately separate so the 3B generators and the
14B judge are never in VRAM at once. The API baseline is skipped automatically if
its key is unset (or `api_baseline.enabled: false` in `config.yaml`).

## Results & demo

After a run, `results/` holds:
- `scores.csv` — the comparison table (per-axis judge scores, judge_avg,
  faithful_pct, win-rate, latency, avg_iters). Higher is better except latency.
- `fig1..5.png` — slide figures (`plot.py`).

Inspect the generated newsletters with the demo helper:

```bash
python demo/showcase.py            # LIVE: run the agents on one newsletter,
                                   #   step by step (Reader→Writer→Editor→feedback)
python demo/showcase.py --results  # replay the scores table + sample outputs (no GPU)
python demo/showcase.py --export   # write results/comparison.md — every newsletter,
                                   #   4 systems side by side (overview / appendix)
python demo/showcase.py --slow     # reveal text slowly (nicer for screen recording)
```

## Notes

- Data comes only from official, free APIs (no scraping). The eval set is frozen
  once so every system runs on identical inputs (reproducibility).
- To run smaller/faster first, lower `num_newsletters` in `config.yaml`.
- Swap any model in `config.yaml` — nothing else needs to change. The default
  ceiling is `gemini-3.5-flash` (thinking disabled so the full token budget goes
  to the newsletter, not hidden reasoning). Free-tier quotas are small
  (3.5-flash is ~20 requests/day *per project*), so the Gemini client throttles,
  **rotates across keys** (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`, … from separate
  projects), and backs off on 429.
- `transformers` is pinned `<5` because 5.x requires torch ≥ 2.4 (the 3090 box
  ships torch 2.3.1).
