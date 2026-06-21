# NEXT STEPS — running on the Vessl server

Self-note / runbook. Everything in this repo is implemented and pushed; what
remains is running it on a GPU, then making slides + a demo video.
**Deadline: 2026-06-21 23:59.**

---

## 0. Before anything — security

- [ ] **Revoke the old HF token** that was pasted in chat, then make a new one:
      https://huggingface.co/settings/tokens (read scope is enough).
- Never put tokens/keys in committed files. Use env vars only (below).

## 1. One-time setup on the server

```bash
git clone https://github.com/jaehoookim/llm_final.git
cd llm_final
pip install -r requirements.txt

# Auth — put keys in .env (gitignored, auto-loaded by utils.py):
cp .env.example .env
#   HF_TOKEN=...         Llama-3.2 / Qwen2.5 download
#   GEMINI_API_KEY=...   API baseline (get one at aistudio.google.com/apikey)
# (plain `export`s still work too)

df -h .        # check there's ~35GB free for the model caches (3B + 14B weights)
nvidia-smi     # confirm the 3090 is visible
```

Model access is already sorted: Llama-3.2-3B granted, Qwen2.5-14B is ungated.

## 2. Smoke test first (fast, catches setup bugs)

Edit `config.yaml`: set `num_newsletters: 2`, then:

```bash
python main.py collect      # ~1 min, no GPU
python main.py generate     # loads Llama-3B, then Gemini baseline
python main.py evaluate     # loads Qwen-14B judge (4-bit)
python plot.py              # writes results/fig1..5.png
```

Watch for: HF auth errors, bitsandbytes/CUDA issues, OOM, messy JSON from the
small models (Editor/judge parsing). If JSON parsing looks bad, tweak the
prompts/threshold in `newsroom.py` / `config.yaml`.

## 3. Full run

Set `num_newsletters: 15` back in `config.yaml`, then:

```bash
python main.py all          # collect -> generate -> evaluate  (~half a day, mostly unattended)
python plot.py
```

Outputs:
- `results/scores.csv`  — the results table (judge scores, win-rate, faithfulness, latency)
- `results/fig1..5.png` — slide figures

## 4. Deliverables

- [ ] Summary slides (see `PROJECT_PLAN.md` §1, §2, §7 for the story + figures to use)
- [ ] `demo/demo.mp4` — screen-record one end-to-end run
- [ ] Submit code + slides before the deadline

---

## Notes

- `generate` (3B) and `evaluate` (14B judge) are phase-separated → the two models
  never sit in VRAM at the same time, so a single 24GB 3090 is enough.
- External baseline provider is set in `config.yaml` → `api_baseline`
  (`gemini` by default; switch to `anthropic`/`claude-opus-4-8` if you ever want).
- To skip the API ceiling entirely: `api_baseline.enabled: false` (no key needed).
- Grading target is 22/22 — see `PROJECT_PLAN.md` §2 for how each rubric item is met.
