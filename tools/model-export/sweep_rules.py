"""Sweep window-level combination rules offline, from the eval CLI's span dumps.

  cargo run --release -p tracer-engine --bin eval -- --samples eval/hard --model tracer-int8.onnx --dump desk-hard.tsv
  cargo run --release -p tracer-engine --bin eval -- --samples eval/hard --model rasbt-bf16w.onnx --dump rasbt-hard.tsv
  python -I sweep_rules.py desk-hard.tsv [rasbt-hard.tsv]

Rebuilds analyze()'s aggregation (sentence P = mean of its windows' P,
AI share = word-weighted share of sentences >= 0.5) under each rule, so a rule
can be chosen on data before it goes into the engine. The final numbers come
from the real pipeline (eval binary), not from this script.
"""
import sys
from collections import defaultdict

AI_T = 0.5


def load(path):
    d = defaultdict(lambda: dict(spans={}))
    for line in open(path, encoding="utf-8"):
        name, exp, share, kind, a, b, w, p = line.rstrip("\n").split("\t")
        s = d[name]
        s.update(expected=exp, share=float(share))
        if kind == "sent":
            s["sw"] = [int(x) for x in p.split(",")]
        else:
            s["spans"][(kind, int(a), int(b))] = float(p)
    return dict(d)


def ai_share(sw, wins):
    """wins: {(start, end): p}. Returns word-weighted share of AI sentences."""
    tot = [0.0] * len(sw)
    cnt = [0] * len(sw)
    for (a, b), p in wins.items():
        for i in range(a, b):
            tot[i] += p
            cnt[i] += 1
    sp = [t / c if c else 0.5 for t, c in zip(tot, cnt)]
    return sum(w for w, p in zip(sw, sp) if p >= AI_T) / max(1, sum(sw))


def spans(s, kind):
    return {(a, b): p for (k, a, b), p in s["spans"].items() if k == kind}


def group_of(s, a, b):
    """The (chunked) paragraph-group span containing window a..b."""
    for (k, ga, gb), p in s["spans"].items():
        if k == "group" and ga <= a and b <= gb:
            return p
    # A window can straddle two chunks of an over-long group: use the max overlap.
    best = max(((min(b, gb) - max(a, ga), p) for (k, ga, gb), p in s["spans"].items() if k == "group"), default=(0, 0.5))
    return best[1]


def doc_score(s):
    d = spans(s, "doc")
    return sum(p * sum(s["sw"][a:b]) for (a, b), p in d.items()) / max(1, sum(s["sw"]))


def evaluate(samples, rule):
    res = {}
    for name, s in samples.items():
        res[name] = (s["expected"], s["share"], rule(s))
    return res


def summary(res):
    def count(prefix, exp):
        xs = [v for n, (e, _, v) in res.items() if e == exp and n.startswith(prefix)]
        return sum(v >= 0.5 for v in xs), len(xs)
    gem = next((v for n, (_, _, v) in res.items() if n.startswith("ai-faceit")), None)
    mixed = [abs(v - sh) for n, (e, sh, v) in res.items() if e == "mixed"]
    parts = []
    if gem is not None:
        parts.append(f"gemini {gem * 100:3.0f}%")
    for pre, exp, lab in [("ai-plain", "ai", "plainAI"), ("ai-", "ai", "AI"), ("human-enron", "human", "enronFP"),
                          ("human-wiki", "human", "wikiFP"), ("human-", "human", "humanFP")]:
        k, n = count(pre, exp)
        if n:
            parts.append(f"{lab} {k}/{n}")
    if mixed:
        parts.append(f"mixedErr {sum(mixed) / len(mixed) * 100:.0f}")
    return "  ".join(parts)


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("-")]
    desk = load(files[0])
    rules = {"desklib": lambda s: ai_share(s["sw"], spans(s, "win"))}
    if len(files) > 1:
        ras = load(files[1])
        rules.update(make_rules(desk, ras))
    for name, rule in rules.items():
        print(f"{name:40s} {summary(evaluate(desk, rule))}")
    if "-v" in sys.argv:
        for name, rule in rules.items():
            print(f"\n{name}")
            for n, (e, sh, v) in sorted(evaluate(desk, rule).items()):
                print(f"  {n:48s} {e:6s} {v * 100:4.0f}")


def make_rules(desk, ras):
    """Candidate rules over both models' spans (filled in once rasbt scores exist)."""
    return {}


if __name__ == "__main__":
    main()
