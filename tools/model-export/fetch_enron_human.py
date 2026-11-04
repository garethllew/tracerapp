"""Fetch genuine pre-LLM human business emails (Enron corpus, 2000-2001) for
the hard eval set, the false-positive check closest to Tracer's real use.
Uses the Hugging Face datasets-server rows API (no full download).

Picks prose emails of 80-260 words: no forwards/replies quoting other mail,
no lists or tables, mostly sentences. Writes eval/hard/human-enron-NN.txt.

Usage:  python -I fetch_enron_human.py
"""
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "eval" / "hard"
DATASET = "corbt/enron-emails"
WANT = 15


def rows(offset, length=100):
    q = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train", "offset": offset, "length": length})
    with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=60) as r:
        return [x["row"] for x in json.load(r)["rows"]]


def clean(body):
    body = body.replace("\r", "")
    # stop at quoted/forwarded material and signatures
    body = re.split(r"\n-{2,}|\n_{3,}|\nFrom:|\n>|Original Message|Forwarded by", body)[0]
    paras = [re.sub(r"\s+", " ", p).strip() for p in body.split("\n\n")]
    return "\n\n".join(p for p in paras if p)


def prose_like(t):
    words = t.split()
    if not 80 <= len(words) <= 260:
        return False
    if re.search(r"https?://|@|\t|\|{2}|={3}|\*{3}", t):
        return False
    sentences = re.findall(r"[.!?](\s|$)", t)
    caps = sum(w.isupper() and len(w) > 3 for w in words) / len(words)
    return len(sentences) >= 4 and caps < 0.05


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    got = 0
    offset = 0
    seen = set()
    while got < WANT and offset < 20000:
        for r in rows(offset):
            body = r.get("body") or r.get("text") or r.get("content") or ""
            t = clean(body)
            key = t[:80]
            if key in seen or not prose_like(t):
                continue
            seen.add(key)
            got += 1
            subj = (r.get("subject") or "").strip().replace("\n", " ")[:60]
            (OUT / f"human-enron-{got:02d}.txt").write_text(
                "---\nexpected: human\nexpected_ai_share: 0.0\nkind: email\n"
                f"source: Enron email corpus (public, 2000-2001), {DATASET} row {offset}\n"
                f"notes: genuine pre-LLM business email; subject: {subj}\n---\n" + t + "\n", encoding="utf-8")
            print(f"{got:02d} {len(t.split()):4d} words  {subj}")
            if got >= WANT:
                break
        offset += 997  # sample across the corpus rather than one mailbox


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
