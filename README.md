# AI Newsroom — Multi-Agent SLM Pipeline for Automated Newsletters

A small study: **does role decomposition across lightweight open-source models
(SLMs) recover most of a large LLM's newsletter quality — and at what cost?**
Newsletter generation is the testbed. See `PROJECT_PLAN.md` for the full design.

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

Metrics: LLM-as-judge rubric scores (1–5), pairwise win-rate vs the SLM floor
(order-swapped to cancel position bias), reference-based faithfulness (NLI
entailment), and efficiency (latency, revision iterations).

## Files

| File | Role |
|---|---|
| `config.yaml` | all model IDs, sizes, thresholds, paths |
| `collect.py` | Scout — builds the frozen eval set from HackerNews + arXiv APIs |
| `newsroom.py` | model wrappers + the 4 agents + feedback loop + baselines |
| `evaluate.py` | the 3 metric axes (pointwise judge, pairwise judge, faithfulness) |
| `main.py` | CLI: `collect` / `generate` / `evaluate` / `all` |
| `utils.py` | config loading + robust JSON extraction |

## Setup (Vessl GPU, single 24GB 3090)

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...            # for the `api` upper-bound baseline (default provider)
huggingface-cli login                # Llama-3.2 / Qwen2.5 are gated on some accounts
```

The external baseline provider is chosen in `config.yaml` → `api_baseline`
(`provider: gemini | anthropic`, plus `model` and an optional inline `api_key`).
Leave `api_key: ""` to read it from the environment (`GEMINI_API_KEY` or
`ANTHROPIC_API_KEY`). Set `enabled: false` to skip the ceiling (no key needed).

## Run

```bash
python main.py collect     # → data/tech.json, data/research.json   (~1 min, no GPU)
python main.py generate    # → results/generations.json            (SLM on GPU)
python main.py evaluate    # → results/scores.csv + printed table   (judge on GPU)
# or end-to-end:
python main.py all
```

`generate` and `evaluate` are deliberately separate so the 3B generators and the
14B judge are never in VRAM at once. The API baseline is skipped automatically if
`ANTHROPIC_API_KEY` is unset (or `api_baseline.enabled: false` in `config.yaml`).

## Notes

- Data comes only from official, free APIs (no scraping). The eval set is frozen
  once so every system runs on identical inputs (reproducibility).
- To run smaller/faster first, lower `num_newsletters` in `config.yaml`.
- Swap any model in `config.yaml` — nothing else needs to change.
