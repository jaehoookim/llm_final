"""
Scout (collector) — build a FROZEN evaluation set from official, free APIs.

Sources (no scraping, no auth, stable):
  - tech     -> HackerNews Firebase API (top stories + their text/comments)
  - research -> arXiv Atom API (recent cs.AI / cs.CL submissions)

Output: data/<domain>.json — a list of "newsletters", each a dict:
  { "id": int, "domain": str, "items": [ {title, url, text}, ... ] }

The set is collected once and reused so every system runs on identical inputs.
"""
import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import requests

from utils import load_config

HN_BASE = "https://hacker-news.firebaseio.com/v0"
ARXIV_API = "http://export.arxiv.org/api/query"


def _clean(text: str, limit: int = 1500) -> str:
    """Strip HTML tags/entities and collapse whitespace; cap length."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def collect_hackernews(n_items: int) -> list[dict]:
    """Return the top `n_items` HN stories as {title, url, text}."""
    ids = requests.get(f"{HN_BASE}/topstories.json", timeout=30).json()
    items = []
    for sid in ids:
        if len(items) >= n_items:
            break
        story = requests.get(f"{HN_BASE}/item/{sid}.json", timeout=30).json()
        if not story or story.get("type") != "story":
            continue
        # Body = story text if present, else the title + a couple top comments.
        body = _clean(story.get("text", ""))
        if not body and story.get("kids"):
            comments = []
            for kid in story["kids"][:3]:
                c = requests.get(f"{HN_BASE}/item/{kid}.json", timeout=30).json()
                if c and c.get("text"):
                    comments.append(_clean(c["text"], 400))
            body = " ".join(comments)
        items.append({
            "title": story.get("title", "(untitled)"),
            "url": story.get("url", f"https://news.ycombinator.com/item?id={sid}"),
            "text": body or story.get("title", ""),
        })
        time.sleep(0.05)  # be polite to the API
    return items


def collect_arxiv(n_items: int) -> list[dict]:
    """Return the `n_items` most recent cs.AI/cs.CL papers as {title, url, text}."""
    params = {
        "search_query": "cat:cs.AI OR cat:cs.CL",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": n_items,
    }
    resp = requests.get(ARXIV_API, params=params, timeout=30)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    items = []
    for entry in root.findall("a:entry", ns):
        title = _clean(entry.findtext("a:title", default="", namespaces=ns))
        summary = _clean(entry.findtext("a:summary", default="", namespaces=ns))
        url = entry.findtext("a:id", default="", namespaces=ns)
        items.append({"title": title, "url": url, "text": summary})
    return items


def build_domain(domain: str, cfg: dict) -> list[dict]:
    """Group collected items into `num_newsletters` buckets of `items_per_newsletter`."""
    k = cfg["items_per_newsletter"]
    total = cfg["num_newsletters"] * k
    print(f"[collect] {domain}: fetching {total} items...")
    raw = collect_hackernews(total) if domain == "tech" else collect_arxiv(total)
    newsletters = []
    for i in range(0, len(raw) - k + 1, k):
        newsletters.append({
            "id": len(newsletters),
            "domain": domain,
            "items": raw[i:i + k],
        })
        if len(newsletters) >= cfg["num_newsletters"]:
            break
    return newsletters


def main():
    cfg = load_config()
    os.makedirs(cfg["data_dir"], exist_ok=True)
    for domain in cfg["domains"]:
        newsletters = build_domain(domain, cfg)
        path = os.path.join(cfg["data_dir"], f"{domain}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(newsletters, f, ensure_ascii=False, indent=2)
        print(f"[collect] wrote {len(newsletters)} newsletters -> {path}")


if __name__ == "__main__":
    main()
