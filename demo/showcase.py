"""
demo/showcase.py — a fast, screen-friendly walkthrough of the AI Newsroom.
Built for the demo video: it conveys *how the system feels* without sitting
through the multi-hour full run.

Two modes (neither needs the full run):

  python demo/showcase.py            # LIVE  : run the agents on ONE newsletter,
                                     #         printing each step Reader -> Writer
                                     #         -> Editor -> feedback loop, then the
                                     #         single-SLM baseline for contrast.
                                     #         ~30-60s on the 3B (model cached).
  python demo/showcase.py --results  # REPLAY: no GPU. Print the final scores table
                                     #         + a sample newsletter per system.

Add --slow to reveal text line-by-line (reads better on camera).
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config  # noqa: E402

# --- tiny ANSI helpers ------------------------------------------------------ #
B, D, CY, GR, YE, RE, X = (
    "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")
SLOW = False


def rule(title: str, color: str = CY):
    bar = "─" * max(4, 70 - len(title))
    print(f"\n{color}{B}┌── {title} {bar}{X}")


def reveal(text: str, indent: str = "  ", color: str = ""):
    for line in str(text).splitlines() or [""]:
        print(f"{color}{indent}{line}{X}")
        if SLOW:
            time.sleep(0.04)


def pause(s: float = 0.8):
    if SLOW:
        time.sleep(s)


# --------------------------------------------------------------------------- #
def live(cfg):
    import newsroom

    # Pick one collected newsletter as the running example.
    nls = []
    for d in cfg["domains"]:
        p = os.path.join(cfg["data_dir"], f"{d}.json")
        if os.path.exists(p):
            nls.extend(json.load(open(p, encoding="utf-8")))
    if not nls:
        sys.exit("No data. Run `python main.py collect` first.")
    nl = nls[0]
    items = nl["items"]

    print(f"\n{B}AI NEWSROOM — live walkthrough{X}  "
          f"{D}(domain: {nl['domain']}, {len(items)} source articles){X}")
    print(f"{D}Pipeline: Scout → Reader → Writer → Editor ⮌ feedback{X}")

    rule("INPUT  ·  Scout collected these source articles")
    for i, it in enumerate(items):
        reveal(f"[{i+1}] {it['title']}", color=D)
    pause()

    print(f"\n{D}loading SLM: {cfg['slm_model']} …{X}")
    slm = newsroom.LocalLM(cfg["slm_model"], temperature=cfg["gen_temperature"])

    # 1) READER — one faithful 3-line summary per article (batched = parallel).
    rule("READER  ·  3-line faithful summary per article (batched)")
    summaries = newsroom.reader_summarize(slm, items, cfg["max_new_tokens_summary"])
    for i, s in enumerate(summaries):
        reveal(f"• item {i+1}:", color=YE)
        reveal(s, indent="    ", color=D)
    pause()

    # 2) WRITER — fuse the summaries into one article.
    rule("WRITER  ·  fuse summaries into one newsletter draft")
    draft = newsroom.writer_draft(slm, summaries, cfg["max_new_tokens_article"])
    reveal(draft[:700] + ("…" if len(draft) > 700 else ""))
    pause()

    # 3) EDITOR — score the draft + emit a title and one feedback line.
    rule("EDITOR  ·  score the draft (1-5) + title + feedback")
    verdict = newsroom.editor_review(slm, summaries, draft)
    thr = cfg["editor_pass_threshold"]
    for k in ("factuality", "coherence", "readability"):
        print(f"  {k:12s}: {B}{verdict.get(k, '?')}{X}/5")
    print(f"  {'avg':12s}: {B}{verdict['avg']:.1f}{X}/5   "
          f"(pass threshold {thr})")
    reveal(f'title   : "{verdict.get("title")}"', color=GR)
    reveal(f'feedback: {verdict.get("feedback")}', color=YE)
    pause()

    # 4) FEEDBACK LOOP — Editor → Writer until the draft passes (or retries run out).
    rule("FEEDBACK LOOP  ·  Editor → Writer self-correction")
    iterations = 0
    if verdict["avg"] >= thr:
        print(f"  {GR}Draft already ≥ {thr}. No revision needed.{X}")
    for _ in range(cfg["feedback_max_retries"]):
        if verdict["avg"] >= thr:
            break
        iterations += 1
        print(f"  {YE}↻ revision {iterations}: Writer applies the feedback…{X}")
        draft = newsroom.writer_revise(slm, summaries, draft,
                                       verdict["feedback"], cfg["max_new_tokens_article"])
        verdict = newsroom.editor_review(slm, summaries, draft)
        print(f"    new avg: {B}{verdict['avg']:.1f}{X}/5")
        pause()

    rule("OUR METHOD  ·  final newsletter (multi-agent + feedback)", GR)
    print(f"  {B}{verdict.get('title')}{X}\n")
    reveal(draft[:900] + ("…" if len(draft) > 900 else ""))

    # Contrast: the single-SLM lower bound on the SAME input, one monolithic prompt.
    rule("CONTRAST  ·  single-SLM baseline, one prompt (lower bound)", RE)
    base = newsroom.baseline_single_slm(slm, items, cfg)
    reveal(base["newsletter"][:500] + "…", color=D)
    print(f"\n  {D}→ Same inputs. The multi-agent version decomposes the job and "
          f"self-corrects;\n    the single-SLM version does it all in one shot. "
          f"evaluate.py quantifies the gap.{X}")
    slm.unload()


# --------------------------------------------------------------------------- #
def results(cfg):
    rdir = cfg["results_dir"]
    scores = os.path.join(rdir, "scores.csv")
    gens = os.path.join(rdir, "generations.json")
    if not os.path.exists(scores):
        sys.exit("No results yet. Run `python main.py all` (or evaluate) first.")

    rows = list(csv.DictReader(open(scores, encoding="utf-8")))
    rule("RESULTS  ·  4 systems, identical frozen inputs", GR)
    headers = list(rows[0].keys())
    w = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    print("  " + " | ".join(f"{B}{h.ljust(w[h])}{X}" for h in headers))
    for r in rows:
        print("  " + " | ".join(str(r[h]).ljust(w[h]) for h in headers))
    print(f"\n  {D}api = large-LLM ceiling · single_slm = floor · "
          f"multi_agent_fb = our method{X}")

    if os.path.exists(gens):
        recs = json.load(open(gens, encoding="utf-8"))
        r = recs[0]
        rule(f"SAMPLE OUTPUTS  ·  same input ({r['domain']}), 4 systems")
        for s in ("single_slm", "multi_agent", "multi_agent_fb", "api"):
            if s in r:
                print(f"\n  {B}{s}{X} {D}(latency {r[s].get('latency_s')}s, "
                      f"iters {r[s].get('iterations', '-')}):{X}")
                reveal(r[s]["newsletter"][:300] + "…", indent="    ", color=D)


SYSTEMS = ("single_slm", "multi_agent", "multi_agent_fb", "api")
LABELS = {"single_slm": "single SLM (floor)", "multi_agent": "multi-agent",
          "multi_agent_fb": "multi-agent + feedback (ours)", "api": "large LLM (ceiling)"}


def export(cfg, limit: int = 0, chars: int = 600):
    """Write a model-by-model comparison of the generated newsletters to a single
    markdown file — handy as a slide appendix or a quick scroll-through overview."""
    gens = os.path.join(cfg["results_dir"], "generations.json")
    if not os.path.exists(gens):
        sys.exit("No generations yet. Run `python main.py generate` (or all) first.")
    recs = json.load(open(gens, encoding="utf-8"))
    if limit:
        recs = recs[:limit]
    out = os.path.join(cfg["results_dir"], "comparison.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Newsletter comparison — 4 systems on identical inputs\n\n")
        f.write(f"_{len(recs)} newsletters · excerpts truncated to ~{chars} chars "
                "(full text in `generations.json`)._\n")
        for i, r in enumerate(recs):
            f.write(f"\n---\n\n## {i+1}. {r['domain']}\n\n")
            for s in SYSTEMS:
                if s not in r:
                    continue
                d = r[s]
                meta = f"latency {d.get('latency_s')}s"
                if d.get("iterations") is not None:
                    meta += f" · {d['iterations']} revision(s)"
                body = d["newsletter"].strip().replace("\n", " ")
                f.write(f"### {LABELS[s]}  \n_{meta}_\n\n")
                f.write(f"> {body[:chars]}{'…' if len(body) > chars else ''}\n\n")
    print(f"wrote {out}  ({len(recs)} newsletters, 4 systems each)")


# --------------------------------------------------------------------------- #
def main():
    global SLOW
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="store_true", help="replay scores (no GPU)")
    ap.add_argument("--export", action="store_true",
                    help="write results/comparison.md (model-by-model overview)")
    ap.add_argument("--slow", action="store_true", help="reveal text slowly (for video)")
    args = ap.parse_args()
    SLOW = args.slow

    cfg = load_config()
    if args.export:
        export(cfg)
    elif args.results:
        results(cfg)
    else:
        live(cfg)
    print(f"\n{GR}{B}done.{X}\n")


if __name__ == "__main__":
    main()
