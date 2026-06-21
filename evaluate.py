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
    "Rate the single NEWSLETTER below as ONE whole piece (do NOT score the source "
    "articles individually). Give one integer from 1 (poor) to 5 (excellent) per "
    "axis, judged against the SOURCE.\n"
    "Return ONLY this flat JSON object, nothing else:\n"
    '{{"factuality": <1-5>, "coherence": <1-5>, "readability": <1-5>, "conciseness": <1-5>}}\n\n'
    "SOURCE:\n{source}\n\nNEWSLETTER:\n{newsletter}\n\nJSON:"
)
AXES = ["factuality", "coherence", "readability", "conciseness"]


def _flatten_scores(v: dict) -> dict:
    """Pull axis scores out of the judge's JSON, tolerating the model nesting them
    under per-article keys (e.g. {"1": {...}, "2": {...}}) instead of returning a
    flat object. Falls back to averaging the nested values for each axis."""
    if any(ax in v for ax in AXES):
        return v
    nested = [d for d in v.values() if isinstance(d, dict) and any(ax in d for ax in AXES)]
    if not nested:
        return v
    flat = {}
    for ax in AXES:
        vals = [d[ax] for d in nested if ax in d]
        if vals:
            try:
                flat[ax] = sum(int(x) for x in vals) / len(vals)
            except (ValueError, TypeError):
                pass
    return flat


def judge_pointwise(judge, source: str, newsletter: str) -> dict:
    prompt = POINTWISE_PROMPT.format(source=source[:3000], newsletter=newsletter)
    raw = judge.generate([prompt], max_new_tokens=120, system=JUDGE_SYS)[0]
    v = _flatten_scores(extract_json(raw))
    out = {}
    for ax in AXES:
        try:
            out[ax] = max(1, min(5, round(float(v[ax]))))
        except (ValueError, TypeError, KeyError):
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
        import torch
        from transformers import pipeline
        # Run on GPU if present, and cap inputs at the model's 512-token limit so
        # long sources don't overflow (the "513 > 512" indexing error) or fall
        # back to slow CPU inference.
        device = 0 if torch.cuda.is_available() else -1
        self.nli = pipeline("text-classification", model=model_id, top_k=None,
                            device=device, truncation=True, max_length=512)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]

    @staticmethod
    def _chunks(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
        """Split the source into overlapping windows that fit the NLI 512-token
        limit. Sources here run 3.5k-7k chars, so checking only the first window
        would falsely flag any sentence about a later article as unsupported."""
        if len(text) <= size:
            return [text]
        out, i = [], 0
        while i < len(text):
            out.append(text[i:i + size])
            i += size - overlap
        return out

    def score(self, source: str, newsletter: str) -> float:
        sents = self._sentences(newsletter)
        if not sents:
            return 0.0
        chunks = self._chunks(source)
        entailed = 0
        for s in sents:
            # A sentence is faithful if ANY source window entails it.
            preds = self.nli([{"text": c, "text_pair": s} for c in chunks])
            for p in preds:
                if "entail" in max(p, key=lambda d: d["score"])["label"].lower():
                    entailed += 1
                    break
        return round(entailed / len(sents), 3)
