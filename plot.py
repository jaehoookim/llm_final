"""
Make slide-ready figures from the evaluation outputs. Run AFTER `main.py evaluate`.

  python plot.py        # reads results/scores.csv (+ results/generations.json)
                        # writes PNGs into results/

Figures:
  fig1_rubric.png        grouped bars: judge rubric axes per system
  fig2_quality_cost.png  scatter: judged quality vs latency (the headline tradeoff)
  fig3_winrate.png       bars: pairwise win-rate vs the single-SLM floor
  fig4_faithfulness.png  bars: faithfulness % per system
  fig5_feedback.png      feedback loop effect: Editor score before vs after + #iters

Only matplotlib is required; missing inputs are skipped with a warning.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless (Vessl has no display)
import matplotlib.pyplot as plt

from utils import load_config

# Stable display order + friendly labels for whatever systems are present.
SYSTEM_LABELS = {
    "api": "Large LLM\n(ceiling)",
    "single_slm": "Single SLM\n(floor)",
    "multi_agent": "Multi-agent\nSLM",
    "multi_agent_fb": "Multi-agent\n+ feedback (ours)",
}
ORDER = ["api", "single_slm", "multi_agent", "multi_agent_fb"]
AXES = ["factuality", "coherence", "readability", "conciseness"]


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def load_scores(results_dir: str) -> list[dict]:
    path = os.path.join(results_dir, "scores.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Keep configured order, only systems actually present.
    present = {r["system"]: r for r in rows}
    return [present[s] for s in ORDER if s in present]


def _labels(rows):
    return [SYSTEM_LABELS.get(r["system"], r["system"]) for r in rows]


def _save(fig, results_dir, name):
    out = os.path.join(results_dir, name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out}")


# --------------------------------------------------------------------------- #
def fig_rubric(rows, results_dir):
    """Grouped bars: one cluster per system, one bar per rubric axis."""
    import numpy as np
    x = np.arange(len(rows))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, axis in enumerate(AXES):
        vals = [_num(r[axis]) for r in rows]
        ax.bar(x + (i - 1.5) * width, vals, width, label=axis.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(rows))
    ax.set_ylabel("Judge score (1–5)")
    ax.set_ylim(0, 5)
    ax.set_title("LLM-as-judge rubric scores by system")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, results_dir, "fig1_rubric.png")


def fig_quality_cost(rows, results_dir):
    """Scatter: judged quality vs latency — the 'most quality, less cost' story."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for r in rows:
        lat, q = _num(r["latency_s"]), _num(r["judge_avg"])
        if lat is None or q is None:
            continue
        ax.scatter(lat, q, s=90)
        ax.annotate(SYSTEM_LABELS.get(r["system"], r["system"]).replace("\n", " "),
                    (lat, q), textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xlabel("End-to-end latency (s)  — lower is better")
    ax.set_ylabel("Judge avg (1–5)  — higher is better")
    ax.set_title("Quality vs. cost")
    ax.grid(True, alpha=0.3)
    _save(fig, results_dir, "fig2_quality_cost.png")


def fig_winrate(rows, results_dir):
    """Bars: pairwise win-rate vs the single-SLM floor (floor itself excluded)."""
    data = [(r, _num(r["winrate_vs_floor_pct"])) for r in rows]
    data = [(r, v) for r, v in data if v is not None]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(_labels([r for r, _ in data]), [v for _, v in data])
    ax.axhline(50, ls="--", c="gray", lw=1, label="50% (tie with floor)")
    ax.set_ylabel("Win-rate vs single-SLM floor (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Pairwise win-rate (order-swapped judge)")
    ax.legend(fontsize=8)
    _save(fig, results_dir, "fig3_winrate.png")


def fig_faithfulness(rows, results_dir):
    """Bars: faithfulness % (fraction of sentences entailed by the source)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(_labels(rows), [_num(r["faithful_pct"]) for r in rows])
    ax.set_ylabel("Faithful sentences (%)  — higher is better")
    ax.set_ylim(0, 100)
    ax.set_title("Faithfulness (NLI entailment vs. source)")
    _save(fig, results_dir, "fig4_faithfulness.png")


def fig_feedback(results_dir: str):
    """Feedback-loop effect: Editor score before (multi_agent) vs after
    (multi_agent_fb) the loop, plus the distribution of revision counts."""
    path = os.path.join(results_dir, "generations.json")
    if not os.path.exists(path):
        print("[plot] generations.json missing — skipping fig5")
        return
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)
    before, after, iters = [], [], []
    for r in recs:
        if "multi_agent" in r and "editor_avg" in r["multi_agent"]:
            before.append(r["multi_agent"]["editor_avg"])
        if "multi_agent_fb" in r:
            after.append(r["multi_agent_fb"].get("editor_avg"))
            iters.append(r["multi_agent_fb"].get("iterations", 0))
    if not before or not after:
        print("[plot] no feedback data — skipping fig5")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    mb = sum(before) / len(before)
    ma = sum(v for v in after if v is not None) / max(len([v for v in after if v is not None]), 1)
    ax1.bar(["Before loop\n(no feedback)", "After loop\n(feedback)"], [mb, ma],
            color=["#bbb", "#4c72b0"])
    ax1.set_ylim(0, 5)
    ax1.set_ylabel("Mean Editor score (1–5)")
    ax1.set_title("Self-correction loop raises Editor score")
    for i, v in enumerate([mb, ma]):
        ax1.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=9)

    maxk = max(iters) if iters else 0
    bins = range(0, maxk + 2)
    ax2.hist(iters, bins=bins, align="left", rwidth=0.8, color="#4c72b0")
    ax2.set_xlabel("Revision iterations per newsletter")
    ax2.set_ylabel("Count")
    ax2.set_xticks(range(0, maxk + 1))
    ax2.set_title("How often the loop fired")
    _save(fig, results_dir, "fig5_feedback.png")


# --------------------------------------------------------------------------- #
def main():
    cfg = load_config()
    rd = cfg["results_dir"]
    rows = load_scores(rd)
    if rows:
        fig_rubric(rows, rd)
        fig_quality_cost(rows, rd)
        fig_winrate(rows, rd)
        fig_faithfulness(rows, rd)
    else:
        print("[plot] results/scores.csv missing — run `python main.py evaluate` first")
    fig_feedback(rd)


if __name__ == "__main__":
    main()
