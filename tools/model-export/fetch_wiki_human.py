"""Fetch plain, factual HUMAN text for the hard eval set: lead paragraphs of
English Wikipedia articles as they stood in mid-2019 (before LLM-written text),
on topics near Tracer's real use (gaming platforms, products, how-tos).

Writes eval/hard/human-wiki-<slug>.txt with the usual front matter. Text is
CC BY-SA 4.0; attribution (article + revision id) goes in each file's notes.

Usage:  python -I fetch_wiki_human.py
"""
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "eval" / "hard"
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "tracer-eval-builder/0.1 (local research script)"}
AS_OF = "2019-06-01T00:00:00Z"

TITLES = [
    "Counter-Strike: Global Offensive", "Elo rating system", "Steam (service)", "Valve Anti-Cheat",
    "Esports", "Matchmaking (video games)", "Strava", "Payroll", "Board game cafe",
    "Multi-factor authentication", "Product return", "Release notes", "Discord (software)",
    "Twitch (service)", "Spotify", "Fitbit", "Duolingo", "Dropbox", "Password manager", "Netflix", "Uber", "Airbnb", "Slack (software)", "Zoom Video Communications",
]


def get(params):
    import time; time.sleep(6)  # be polite: the API rate-limits bursts
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def lead_text(title):
    rev = get({"action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
               "rvstart": AS_OF, "rvdir": "older", "rvprop": "ids|timestamp"})
    page = rev["query"]["pages"][0]
    if "revisions" not in page:
        return None
    r = page["revisions"][0]
    parsed = get({"action": "parse", "oldid": r["revid"], "prop": "text", "section": 0, "disablelimitreport": 1})
    raw = parsed["parse"]["text"]
    paras = re.findall(r"<p>(.*?)</p>", raw, flags=re.S)
    out = []
    for p in paras:
        t = re.sub(r"<sup.*?</sup>", "", p, flags=re.S)          # citations
        t = re.sub(r"<style.*?</style>", "", t, flags=re.S)
        t = re.sub(r"<[^>]+>", "", t)
        t = html.unescape(t)
        t = re.sub(r"\[\d+\]|\[citation needed\]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t.split()) >= 25:
            out.append(t)
    text = "\n\n".join(out)
    words = text.split()
    if len(words) > 260:                                         # keep it sample-sized
        text = " ".join(words[:260]).rsplit(". ", 1)[0] + "."
    return text, r["revid"], r["timestamp"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for title in TITLES:
        slug0 = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
        if (OUT / f"human-wiki-{slug0}.txt").exists():
            continue
        try:
            res = lead_text(title)
        except Exception as e:  # noqa: BLE001
            print(f"skip {title}: {e}")
            continue
        if not res or len(res[0].split()) < 80:
            print(f"skip {title}: too short")
            continue
        text, revid, ts = res
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
        (OUT / f"human-wiki-{slug}.txt").write_text(
            "---\nexpected: human\nexpected_ai_share: 0.0\nkind: encyclopedic\n"
            f"source: Wikipedia, \"{title}\", revision {revid} ({ts[:10]}), CC BY-SA 4.0\n"
            "notes: plain factual human prose written before LLMs; false-positive check\n---\n" + text + "\n",
            encoding="utf-8")
        print(f"{slug:42s} {len(text.split()):4d} words  rev {revid} {ts[:10]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
