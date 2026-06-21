"""Tiny shared helpers: config loading + robust JSON extraction from LLM text."""
import json
import os
import re

import yaml

# Load secrets from a local .env (gitignored) into the environment, so HF_TOKEN /
# GEMINI_API_KEY / ANTHROPIC_API_KEY are available before any model is loaded.
# Every entry point imports utils, so this runs first. No-op if .env is absent.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_json(text: str) -> dict:
    """Pull the first {...} block out of a model response and parse it.

    Small models often wrap JSON in prose or code fences; this is forgiving.
    Returns {} if nothing parseable is found.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
