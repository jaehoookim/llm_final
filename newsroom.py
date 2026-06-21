"""
The newsroom: model wrappers + the 4 agents + self-correction loop + baselines.

Systems produced here (compared in evaluate.py):
  1. baseline_api         — single large LLM (Claude) ............ upper bound
  2. baseline_single_slm  — single SLM, one monolithic prompt .... lower bound
  3. run_pipeline(...,K=0) — multi-agent SLM, no feedback loop
  4. run_pipeline(...,K>0) — multi-agent SLM + Editor->Writer loop  (our method)

Scout (collection) lives in collect.py; here a "newsletter" already has its
frozen `items`, so the pipeline starts at Reader.
"""
import time

from utils import extract_json

# --------------------------------------------------------------------------- #
# Model wrappers                                                               #
# --------------------------------------------------------------------------- #


class LocalLM:
    """HuggingFace chat model wrapper with batched generation (for the SLM agents
    and the local judge). `generate` takes a list of prompts and returns a list
    of completions — batching is how Reader summarizes 5 articles in parallel."""

    def __init__(self, model_id: str, load_in_4bit: bool = False, temperature: float = 0.7):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def generate(self, prompts: list[str], max_new_tokens: int = 256, system: str = "") -> list[str]:
        import torch

        texts = []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": p}]
            texts.append(self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=4096).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5), top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]

    def unload(self):
        """Free VRAM so a different model (e.g. the judge) can be loaded after."""
        import gc
        import torch
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GeminiLM:
    """External large-LLM baseline via the Google Gemini API (default ceiling)."""

    _MIN_INTERVAL = 13.0  # free tier allows ~5 req/min -> space calls ~13s apart

    def __init__(self, model_id: str, api_key: str = ""):
        import os
        from google import genai
        self.model = model_id
        # The free tier caps requests *per project*, so collect every key the user
        # provided (GEMINI_API_KEY, GEMINI_API_KEY_2, _3, ... from separate
        # projects) and rotate across them when one hits its quota.
        keys = [api_key] if api_key else []
        for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
                     "GEMINI_API_KEY_4", "GEMINI_API_KEY_5"):
            v = os.environ.get(name)
            if v and v not in keys:
                keys.append(v)
        if not keys:
            raise RuntimeError("No Gemini API key (set GEMINI_API_KEY in .env)")
        self._clients = [genai.Client(api_key=k) for k in keys]
        self._idx = 0
        self._dead: set[int] = set()  # keys whose daily quota is used up
        self._last = 0.0

    def generate(self, prompt: str, max_tokens: int = 1024, system: str = "") -> str:
        import time

        from google.genai import types
        # Gemini 2.5/3.x models "think" by default, and thinking tokens are billed
        # against max_output_tokens -> the visible newsletter gets truncated to a
        # few sentences. We want one-shot generation, not reasoning, so disable
        # thinking and give the whole budget to the answer (keeps the ceiling fair).
        cfg = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system or None,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        n = len(self._clients)
        for _ in range(n * 3 + 2):
            if len(self._dead) >= n:
                break
            while self._idx in self._dead:           # skip exhausted keys
                self._idx = (self._idx + 1) % n
            wait = self._MIN_INTERVAL - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self._clients[self._idx].models.generate_content(
                    model=self.model, contents=prompt, config=cfg)
                self._last = time.time()
                return (resp.text or "").strip()
            except Exception as e:  # noqa: BLE001
                self._last = time.time()
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    if "PerDay" in msg or "per day" in msg.lower():
                        self._dead.add(self._idx)    # daily quota gone for this key
                    self._idx = (self._idx + 1) % n  # rotate to the next key
                    continue
                raise
        raise RuntimeError("Gemini API: all keys hit their quota")


class AnthropicLM:
    """External large-LLM baseline via the Anthropic Claude API."""

    def __init__(self, model_id: str, api_key: str = ""):
        import anthropic
        self.model = model_id
        # Empty key -> SDK falls back to ANTHROPIC_API_KEY in the environment.
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def generate(self, prompt: str, max_tokens: int = 1024, system: str = "") -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()


def make_api_lm(api_cfg: dict):
    """Build the external-API baseline client from config (provider/model/key)."""
    provider = api_cfg.get("provider", "gemini").lower()
    key = api_cfg.get("api_key") or ""
    if provider == "gemini":
        return GeminiLM(api_cfg["model"], key)
    if provider == "anthropic":
        return AnthropicLM(api_cfg["model"], key)
    raise ValueError(f"Unknown api_baseline.provider: {provider!r} (use 'gemini' or 'anthropic')")


# --------------------------------------------------------------------------- #
# Prompts                                                                      #
# --------------------------------------------------------------------------- #

READER_SYS = "You are a concise news summarizer. Summarize faithfully; invent nothing."
READER_PROMPT = (
    "Summarize the article below in exactly 3 short sentences. "
    "Use only facts stated in the text.\n\nTITLE: {title}\n\nTEXT: {text}\n\nSUMMARY:"
)

WRITER_SYS = "You are a newsletter writer. Weave summaries into one coherent article."
WRITER_PROMPT = (
    "Write a single cohesive newsletter article covering all of the items below.\n"
    "Requirements: exactly 4 short paragraphs, ~250 words total, tight and "
    "concise — no filler, no repetition, no preamble. Stay faithful to the "
    "summaries; add no facts that are not present.\n\n{summaries}\n\nARTICLE:"
)
WRITER_REVISE_PROMPT = (
    "Improve the draft below using the editor's feedback. Make ONLY the changes "
    "needed to address the feedback — keep everything that already works. Do not "
    "pad, do not repeat, and add no facts beyond the summaries. Keep it to exactly "
    "4 short paragraphs (~250 words), concise.\n\n"
    "SUMMARIES:\n{summaries}\n\nDRAFT:\n{draft}\n\n"
    "EDITOR FEEDBACK:\n{feedback}\n\nIMPROVED ARTICLE:"
)

EDITOR_SYS = "You are a chief editor. Return ONLY JSON."
EDITOR_PROMPT = (
    "Evaluate the newsletter draft against the source summaries on a 1-5 scale. "
    "Score the SAME axes a downstream judge uses. Return ONLY a JSON object with "
    "keys: factuality, coherence, readability, conciseness (ints 1-5), "
    "title (string), feedback (string: one specific, actionable sentence "
    "targeting the lowest-scoring axis).\n\n"
    "SUMMARIES:\n{summaries}\n\nDRAFT:\n{draft}\n\nJSON:"
)

SINGLE_SLM_PROMPT = (
    "You are a newsletter writer. Read the {n} articles below and write ONE "
    "coherent newsletter article covering all of them.\n"
    "Requirements: exactly 4 short paragraphs, ~250 words total, tight and "
    "concise — no filler, no repetition, no preamble. Use only facts present in "
    "the articles.\n\n{articles}\n\nARTICLE:"
)


def _format_summaries(summaries: list[str]) -> str:
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(summaries))


def _format_articles(items: list[dict]) -> str:
    return "\n\n".join(f"[{i+1}] {it['title']}\n{it['text']}" for i, it in enumerate(items))


# --------------------------------------------------------------------------- #
# Agents                                                                       #
# --------------------------------------------------------------------------- #


def reader_summarize(llm: LocalLM, items: list[dict], max_new_tokens: int) -> list[str]:
    """Reader: one faithful 3-line summary per item, generated as a single batch
    (this batched call is the measured 'parallel processing' speedup)."""
    prompts = [READER_PROMPT.format(title=it["title"], text=it["text"]) for it in items]
    return llm.generate(prompts, max_new_tokens=max_new_tokens, system=READER_SYS)


def writer_draft(llm: LocalLM, summaries: list[str], max_new_tokens: int) -> str:
    prompt = WRITER_PROMPT.format(summaries=_format_summaries(summaries))
    return llm.generate([prompt], max_new_tokens=max_new_tokens, system=WRITER_SYS)[0]


def writer_revise(llm: LocalLM, summaries: list[str], draft: str, feedback: str,
                  max_new_tokens: int) -> str:
    prompt = WRITER_REVISE_PROMPT.format(
        summaries=_format_summaries(summaries), draft=draft, feedback=feedback)
    return llm.generate([prompt], max_new_tokens=max_new_tokens, system=WRITER_SYS)[0]


def editor_review(llm: LocalLM, summaries: list[str], draft: str) -> dict:
    """Editor: score the draft + emit a title and one feedback sentence (JSON)."""
    prompt = EDITOR_PROMPT.format(summaries=_format_summaries(summaries), draft=draft)
    raw = llm.generate([prompt], max_new_tokens=256, system=EDITOR_SYS)[0]
    verdict = extract_json(raw)
    axes = ("factuality", "coherence", "readability", "conciseness")
    scores = [verdict.get(k, 3) for k in axes]
    try:
        verdict["avg"] = sum(int(s) for s in scores) / len(axes)
    except (ValueError, TypeError):
        verdict["avg"] = 3.0
    verdict.setdefault("title", "Newsletter")
    verdict.setdefault("feedback", "Improve coherence and remove unsupported claims.")
    return verdict


# --------------------------------------------------------------------------- #
# Systems                                                                      #
# --------------------------------------------------------------------------- #


def run_pipeline(llm: LocalLM, items: list[dict], cfg: dict, use_feedback: bool) -> dict:
    """Multi-agent SLM newsroom. `use_feedback` toggles the Editor->Writer loop
    (the +3 'new method'). Returns the newsletter, the Editor verdict, the number
    of revision iterations, and the wall-clock latency."""
    t0 = time.perf_counter()
    summaries = reader_summarize(llm, items, cfg["max_new_tokens_summary"])
    draft = writer_draft(llm, summaries, cfg["max_new_tokens_article"])
    verdict = editor_review(llm, summaries, draft)

    iterations = 0
    if use_feedback:
        for _ in range(cfg["feedback_max_retries"]):
            if verdict["avg"] >= cfg["editor_pass_threshold"]:
                break
            draft = writer_revise(llm, summaries, draft, verdict["feedback"],
                                  cfg["max_new_tokens_article"])
            verdict = editor_review(llm, summaries, draft)
            iterations += 1

    return {
        "title": verdict.get("title", "Newsletter"),
        "newsletter": draft,
        "summaries": summaries,
        "editor_avg": verdict["avg"],
        "iterations": iterations,
        "latency_s": round(time.perf_counter() - t0, 2),
    }


def baseline_single_slm(llm: LocalLM, items: list[dict], cfg: dict) -> dict:
    """Lower bound: one SLM call fuses all articles at once (no decomposition)."""
    t0 = time.perf_counter()
    prompt = SINGLE_SLM_PROMPT.format(n=len(items), articles=_format_articles(items))
    text = llm.generate([prompt], max_new_tokens=cfg["max_new_tokens_article"])[0]
    return {"title": "Newsletter", "newsletter": text,
            "latency_s": round(time.perf_counter() - t0, 2)}


def baseline_api(api_llm: AnthropicLM, items: list[dict], cfg: dict) -> dict:
    """Upper bound: one large-LLM (Claude) call fuses all articles at once."""
    t0 = time.perf_counter()
    prompt = SINGLE_SLM_PROMPT.format(n=len(items), articles=_format_articles(items))
    text = api_llm.generate(prompt, max_tokens=cfg["max_new_tokens_article"])
    return {"title": "Newsletter", "newsletter": text,
            "latency_s": round(time.perf_counter() - t0, 2)}
