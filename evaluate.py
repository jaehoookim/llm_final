"""
Evaluation: the three quantitative axes used to compare the systems.

  1. judge_pointwise  — local LLM scores each newsletter 1-5 on 4 rubric axes
  2. judge_pairwise   — local LLM picks A vs B; A/B order swapped to cancel bias
  3. faithfulness     — reference-based hallucination rate via an NLI model
  (efficiency: latency is recorded by newsroom.py during generation)

The judge is a separate, larger model than the generators (see config.yaml) to
limit self-preference bias.
"""
import re

from utils import extract_json

# --------------------------------------------------------------------------- #
# 1. Pointwise rubric scoring                                                  #
# --------------------------------------------------------------------------- #

JUDGE_SYS = "You are a strict evaluator of news writing. Return ONLY JSON."
POINTWISE_PROMPT = (
    "Score the NEWSLETTER from 1 (poor) to 5 (excellent) on each axis, judging "
    "against the SOURCE. Return ONLY JSON with integer keys: "
    "factuality, coherence, readability, conciseness.\n\n"
    "SOURCE:\n{source}\n\nNEWSLETTER:\n{newsletter}\n\nJSON:"
)
AXES = ["factuality", "coherence", "readability", "conciseness"]


def judge_pointwise(judge, source: str, newsletter: str) -> dict:
    prompt = POINTWISE_PROMPT.format(source=source[:3000], newsletter=newsletter)
    raw = judge.generate([prompt], max_new_tokens=120, system=JUDGE_SYS)[0]
    v = extract_json(raw)
    out = {}
    for ax in AXES:
        try:
            out[ax] = max(1, min(5, int(v.get(ax, 3))))
        except (ValueError, TypeError):
            out[ax] = 3
    return out


# --------------------------------------------------------------------------- #
# 2. Pairwise win-rate (order-swapped to cancel position bias)                 #
# --------------------------------------------------------------------------- #

PAIRWISE_PROMPT = (
    "Two newsletters (A and B) were written from the same SOURCE. Which is "
    'better overall (accuracy, coherence, readability)? Answer ONLY JSON: '
    '{{"winner": "A" or "B" or "tie"}}.\n\n'
    "SOURCE:\n{source}\n\nNEWSLETTER A:\n{a}\n\nNEWSLETTER B:\n{b}\n\nJSON:"
)


def _judge_once(judge, source: str, a: str, b: str) -> str:
    raw = judge.generate(
        [PAIRWISE_PROMPT.format(source=source[:3000], a=a, b=b)],
        max_new_tokens=30, system=JUDGE_SYS)[0]
    w = str(extract_json(raw).get("winner", "tie")).strip().upper()
    return w if w in ("A", "B") else "TIE"


def judge_pairwise(judge, source: str, cand: str, ref: str) -> str:
    """Compare `cand` vs `ref`; return 'win' / 'lose' / 'tie' for `cand`.

    Runs both orderings and only counts a decisive result when the two agree —
    otherwise it is a tie. This cancels the judge's position bias."""
    r1 = _judge_once(judge, source, cand, ref)   # cand=A
    r2 = _judge_once(judge, source, ref, cand)   # cand=B
    cand_wins = (r1 == "A") + (r2 == "B")
    cand_loses = (r1 == "B") + (r2 == "A")
    if cand_wins == 2:
        return "win"
    if cand_loses == 2:
        return "lose"
    return "tie"


# --------------------------------------------------------------------------- #
# 3. Faithfulness / hallucination rate (reference-based, NLI)                  #
# --------------------------------------------------------------------------- #


class Faithfulness:
    """Fraction of generated sentences entailed by the source text (higher=better).

    A generated sentence is 'faithful' if the NLI model labels (source -> sentence)
    as entailment. The complementary rate approximates hallucination."""

    def __init__(self, model_id: str):
        from transformers import pipeline
        self.nli = pipeline("text-classification", model=model_id, top_k=None)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]

    def score(self, source: str, newsletter: str) -> float:
        sents = self._sentences(newsletter)
        if not sents:
            return 0.0
        entailed = 0
        for s in sents:
            preds = self.nli({"text": source[:2000], "text_pair": s})
            label = max(preds, key=lambda p: p["score"])["label"].lower()
            if "entail" in label:
                entailed += 1
        return round(entailed / len(sents), 3)
