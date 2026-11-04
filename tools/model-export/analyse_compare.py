"""AUC + threshold analysis of compare-results.json on VALID labels only.

Valid AI = every AI sample (classic synthetic + hard plain-voice + Gemini).
Valid human = Wikipedia 2019 leads only. The original eval/samples "human"
files were written by an LLM imitating people, so they can't count as human.

Usage:  python -I analyse_compare.py
"""
import itertools
import json
import sys
from pathlib import Path

R = json.loads((Path(__file__).parent / "work" / "compare-results.json").read_text())
MODELS = [m for m in R if m not in ("tmr", "fakespot", "tabularis")]  # clearly unusable
# Which real-human pool counts: "human-" = Wikipedia + Enron, or pass "wiki" / "enron".
HUMAN_FILTER = "human-" + (sys.argv[1] if len(sys.argv) > 1 else "")


def auc(pos, neg):
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def split(scores_of):
    pos, neg = [], []
    for i, r in enumerate(R["desklib"]["rows"]):
        if r["expected"] == "mixed":
            continue
        is_human = r["expected"] == "human" and r["set"] == "hard" and HUMAN_FILTER in r["name"]
        is_ai = r["expected"] == "ai"
        if not (is_human or is_ai):
            continue
        for s in scores_of(i):
            (neg if is_human else pos).append(s)
    return pos, neg


def report(label, doc_fn, par_fn):
    dp, dn = split(doc_fn)
    pp, pn = split(par_fn)
    # recall on AI at the strictest threshold that keeps wiki doc FPs at 0 and at <= 1/15
    srt = sorted(dn, reverse=True)
    t0, t1 = srt[0] + 1e-9, srt[1] + 1e-9
    r0 = sum(p >= t0 for p in dp) / len(dp)
    r1 = sum(p >= t1 for p in dp) / len(dp)
    print(f"{label:28s} docAUC {auc(dp, dn):.3f}  parAUC {auc(pp, pn):.3f}  "
          f"AI recall @0 human FP {r0:4.0%} (t={t0:.2f})  @1 FP {r1:4.0%}  "
          f"hardAI@0.5 {sum(p >= .5 for p in dp)}/{len(dp)} humanFP@0.5 {sum(n >= .5 for n in dn)}/{len(dn)}")


rows = {m: R[m]["rows"] for m in R}
print(f"valid AI docs: {sum(1 for r in rows['desklib'] if r['expected']=='ai')}, valid human docs: "
      f"{sum(1 for r in rows['desklib'] if r['set']=='hard' and HUMAN_FILTER in r['name'])} ({HUMAN_FILTER}*)\n")
for m in MODELS:
    report(m, lambda i, m=m: [rows[m][i]["doc"]], lambda i, m=m: rows[m][i]["paras"])
print()
for combo in [c for k in (2, 3) for c in itertools.combinations(MODELS, k)]:
    name = "+".join(combo)
    report("mean " + name,
           lambda i, c=combo: [sum(rows[m][i]["doc"] for m in c) / len(c)],
           lambda i, c=combo: [sum(v) / len(c) for v in zip(*(rows[m][i]["paras"] for m in c))])
