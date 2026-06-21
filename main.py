"""
Entry point. Phases are separated so the 3B generators and the 14B judge are
never resident in VRAM at the same time (fits a single 24GB 3090).

  python main.py collect     # Scout: build the frozen eval set (data/*.json)
  python main.py generate    # run all 4 systems -> results/generations.json
  python main.py evaluate    # judge + faithfulness -> results/scores.csv
  python main.py all         # collect -> generate -> evaluate

Systems compared:
  api          large LLM (Claude), one shot ............ upper bound
  single_slm   single SLM, one shot .................... lower bound
  multi_agent  multi-agent SLM, no feedback loop
  multi_agent_fb  multi-agent SLM + feedback loop ...... our method
"""
import csv
import json
import os
import statistics
import sys

from utils import load_config


# --------------------------------------------------------------------------- #
def _load_newsletters(cfg) -> list[dict]:
    nls = []
    for domain in cfg["domains"]:
        path = os.path.join(cfg["data_dir"], f"{domain}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                nls.extend(json.load(f))
    if not nls:
        sys.exit("No data found. Run: python main.py collect")
    return nls


def _source_text(items: list[dict]) -> str:
    return "\n".join(f"{it['title']}. {it['text']}" for it in items)


# --------------------------------------------------------------------------- #
def cmd_generate(cfg):
    """Generate one newsletter per system for every input. SLM systems first
    (GPU), then the API baseline (no GPU)."""
    import newsroom

    newsletters = _load_newsletters(cfg)
    records = []

    print(f"[generate] loading SLM: {cfg['slm_model']}")
    slm = newsroom.LocalLM(cfg["slm_model"], temperature=cfg["gen_temperature"])
    for i, nl in enumerate(newsletters):
        print(f"[generate] {i+1}/{len(newsletters)} ({nl['domain']})")
        items = nl["items"]
        rec = {"domain": nl["domain"], "id": nl["id"], "source": _source_text(items)}
        rec["single_slm"] = newsroom.baseline_single_slm(slm, items, cfg)
        rec["multi_agent"] = newsroom.run_pipeline(slm, items, cfg, use_feedback=False)
        rec["multi_agent_fb"] = newsroom.run_pipeline(slm, items, cfg, use_feedback=True)
        records.append(rec)
    slm.unload()

    if cfg["api_baseline"]["enabled"]:
        ab = cfg["api_baseline"]
        print(f"[generate] API baseline: {ab['provider']}/{ab['model']}")
        try:
            api = newsroom.make_api_lm(ab)
            for rec, nl in zip(records, newsletters):
                rec["api"] = newsroom.baseline_api(api, nl["items"], cfg)
        except Exception as e:  # noqa: BLE001 — keep SLM results if API/key fails
            print(f"[generate] API baseline skipped ({e})")

    path = os.path.join(cfg["results_dir"], "generations.json")
    os.makedirs(cfg["results_dir"], exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[generate] wrote {len(records)} records -> {path}")


# --------------------------------------------------------------------------- #
def cmd_evaluate(cfg):
    """Judge + faithfulness over the generated newsletters; write scores.csv."""
    import evaluate as ev
    import newsroom

    path = os.path.join(cfg["results_dir"], "generations.json")
    if not os.path.exists(path):
        sys.exit("No generations found. Run: python main.py generate")
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    systems = [s for s in ("api", "single_slm", "multi_agent", "multi_agent_fb")
               if any(s in r for r in records)]

    print(f"[evaluate] loading judge: {cfg['judge_model']}")
    judge = newsroom.LocalLM(cfg["judge_model"],
                             load_in_4bit=cfg["judge_load_in_4bit"], temperature=0.0)
    faith = ev.Faithfulness(cfg["nli_model"])

    # Per-system accumulators.
    agg = {s: {ax: [] for ax in ev.AXES} for s in systems}
    for s in systems:
        agg[s].update(faithful=[], latency=[], iterations=[],
                      win=0, lose=0, tie=0)

    for r in records:
        src = r["source"]
        for s in systems:
            if s not in r:  # a system may be missing on some records (e.g. API
                continue    # baseline cut short by a rate limit) — skip, don't crash
            text = r[s]["newsletter"]
            pw = ev.judge_pointwise(judge, src, text)
            for ax in ev.AXES:
                agg[s][ax].append(pw[ax])
            agg[s]["faithful"].append(faith.score(src, text))
            agg[s]["latency"].append(r[s]["latency_s"])
            agg[s]["iterations"].append(r[s].get("iterations", 0))
            # Pairwise vs the single-SLM floor (skip self-comparison).
            if s != "single_slm":
                outcome = ev.judge_pairwise(judge, src, text, r["single_slm"]["newsletter"])
                agg[s][{"win": "win", "lose": "lose", "tie": "tie"}[outcome]] += 1

    # Build + write the results table.
    rows = []
    for s in systems:
        a = agg[s]
        decided = a["win"] + a["lose"] + a["tie"]
        rows.append({
            "system": s,
            **{ax: round(statistics.mean(a[ax]), 2) for ax in ev.AXES},
            "judge_avg": round(statistics.mean(
                [statistics.mean(a[ax]) for ax in ev.AXES]), 2),
            "faithful_pct": round(100 * statistics.mean(a["faithful"]), 1),
            "winrate_vs_floor_pct": "-" if s == "single_slm"
                else round(100 * a["win"] / max(decided, 1), 1),
            "latency_s": round(statistics.mean(a["latency"]), 2),
            "avg_iters": round(statistics.mean(a["iterations"]), 2),
        })

    out = os.path.join(cfg["results_dir"], "scores.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[evaluate] results ({len(records)} inputs) -> {out}\n")
    _print_table(rows)


def _print_table(rows: list[dict]):
    headers = list(rows[0].keys())
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(widths[h]) for h in headers))


# --------------------------------------------------------------------------- #
def main():
    cfg = load_config()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("collect", "all"):
        import collect
        collect.main()
    if cmd in ("generate", "all"):
        cmd_generate(cfg)
    if cmd in ("evaluate", "all"):
        cmd_evaluate(cfg)
    if cmd not in ("collect", "generate", "evaluate", "all"):
        sys.exit(f"Unknown command: {cmd}\nUse: collect | generate | evaluate | all")


if __name__ == "__main__":
    main()
