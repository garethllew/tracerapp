"""Sweep window-level combination rules offline, from the eval CLI's span dumps.

  cargo run --release -p tracer-engine --bin eval -- --samples eval/hard    --model tracer-int8.onnx --dump desk-hard.tsv
  cargo run --release -p tracer-engine --bin eval -- --samples eval/hard    --model rasbt-bf16w.onnx --dump rasbt-hard.tsv
  cargo run --release -p tracer-engine --bin eval -- --samples eval/samples --model tracer-int8.onnx --dump desk-samples.tsv
  cargo run --release -p tracer-engine --bin eval -- --samples eval/samples --model rasbt-bf16w.onnx --dump rasbt-samples.tsv
  python -I sweep_rules.py desk-hard.tsv rasbt-hard.tsv [desk-samples.tsv rasbt-samples.tsv] [-v] [--all]

Rebuilds analyze()'s aggregation (sentence P = mean of its windows' P,
AI share = word-weighted share of sentences >= 0.5) under each rule, so a rule
can be chosen on data before it goes into the engine. The final numbers come
from the real pipeline (eval binary), not from this script.

Chosen (2026-11-04): either0.95/0.5, i.e. the context gate + style lift in
tracer-engine's analyze_with (EngineConfig.context_gate = 0.5,
style_lift_threshold = 0.95). The engine applies it per sentence instead of
per window (each sentence has exactly one doc chunk and one paragraph chunk);
on eval/hard and eval/samples the eval binary reproduces this script's AI%
for every sample. Results: docs/detector-ensemble.md.

Prints one line per rule, best first: fewest human false positives (Enron +
Wikipedia + eval/samples humans), then most AI caught, then mixed error.
Without --all only rules meeting the hard-set targets that matter most
(Enron FP <= 1, Wikipedia FP <= desklib's) are listed, plus the baselines.
"""
import math
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


def doc_of(s, a, b):
    """The doc chunk containing window a..b (max overlap when it straddles two)."""
    best = max(((min(b, db) - max(a, da), p) for (k, da, db), p in s["spans"].items() if k == "doc"), default=(0, 0.5))
    return best[1]


def doc_score(s):
    d = spans(s, "doc")
    return sum(p * sum(s["sw"][a:b]) for (a, b), p in d.items()) / max(1, sum(s["sw"]))


def sig(x):
    return 1 / (1 + math.exp(-x))


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# ---- rules -----------------------------------------------------------------
#
# A rule maps (desklib sample, rasbt sample) -> AI share. Every rule here keeps
# the engine's aggregation and only changes the score each window gets.
#
#   base:  how the two models' WINDOW scores combine
#     desk          desklib only
#     max           max(d, r)                       (the WIP naive ensemble)
#     rasT{t}       r counts only when r >= t        (rasbt high threshold)
#     rasK{k}       max(d, sigmoid(logit(r) - k))    (rasbt shifted towards human)
#     liftG{t}      d, but max(d, r) where rasbt's PARAGRAPH score >= t
#     liftD{t}      d, but max(d, r) where rasbt's WHOLE-TEXT score >= t
#     liftGs{t}     d, but max(d, rg) where rg = rasbt paragraph score >= t
#
#   gate:  a window may only stay >= 0.5 when its context also reads AI;
#          otherwise its score is capped at the context score (min(w, ctx))
#     gG / gD       context = desklib paragraph / whole-text score
#     cG / cD       context = paragraph / whole-text score under the same base
#                   rule (e.g. max(dg, rg) for `max`)


def window_rule(sd, sr, base, gate=None, tc=AI_T):
    dw, rw = spans(sd, "win"), spans(sr, "win")
    out = {}
    for k, d in dw.items():
        r = rw[k]
        a, b = k
        dg, rg = group_of(sd, a, b), group_of(sr, a, b)
        dd, rd = doc_of(sd, a, b), doc_of(sr, a, b)
        w = base(d, r, dg, rg, dd, rd)
        if gate:
            ctx = gate(d, r, dg, rg, dd, rd, base)
            if ctx < tc:
                w = min(w, ctx)
        out[k] = w
    return ai_share(sd["sw"], out)


def make_bases():
    b = {
        "desk": lambda d, r, dg, rg, dd, rd: d,
        "max": lambda d, r, dg, rg, dd, rd: max(d, r),
    }
    for t in (0.6, 0.7, 0.8, 0.9, 0.95, 0.98):
        b[f"rasT{t}"] = lambda d, r, dg, rg, dd, rd, t=t: max(d, r) if r >= t else d
    for k in (0.5, 1.0, 1.5, 2.0, 3.0):
        b[f"rasK{k}"] = lambda d, r, dg, rg, dd, rd, k=k: max(d, sig(logit(r) - k))
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        b[f"liftG{t}"] = lambda d, r, dg, rg, dd, rd, t=t: max(d, r) if rg >= t else d
        b[f"liftD{t}"] = lambda d, r, dg, rg, dd, rd, t=t: max(d, r) if rd >= t else d
        b[f"liftGs{t}"] = lambda d, r, dg, rg, dd, rd, t=t: max(d, rg) if rg >= t else d
    return b


def make_gates():
    def ctx_g(d, r, dg, rg, dd, rd, base):
        return base(dg, rg, dg, rg, dd, rd)

    def ctx_d(d, r, dg, rg, dd, rd, base):
        return base(dd, rd, dg, rg, dd, rd)

    return {
        None: None,
        "gG": lambda d, r, dg, rg, dd, rd, base: dg,
        "gD": lambda d, r, dg, rg, dd, rd, base: dd,
        "cG": ctx_g,
        "cD": ctx_d,
        "cGD": lambda *a: max(ctx_g(*a), ctx_d(*a)),
    }


def either_rule(sd, sr, tr, td, use_win=False):
    """The explicit form of liftGs{t}+cD: a window reads AI when
       (a) desklib's window AND desklib's whole-text (doc chunk) score are >= td, or
       (b) rasbt's paragraph-group score is >= tr (then it takes that score).
    Otherwise it's capped at desklib's whole-text score (shown human)."""
    out = {}
    for k, d in spans(sd, "win").items():
        a, b = k
        dd, rg = doc_of(sd, a, b), group_of(sr, a, b)
        w = d if dd >= td else min(d, dd)
        if rg >= tr:
            w = max(w, spans(sr, "win")[k] if use_win else rg)
        out[k] = w
    return ai_share(sd["sw"], out)


def make_rules(desk, ras):
    rules = {}
    for tr in (0.9, 0.95, 0.97, 0.98, 0.99, 0.995):
        for td in (0.4, 0.5):
            rules[f"either{tr}/{td}"] = lambda sd, tr=tr, td=td: either_rule(sd, ras[sd["name"]], tr, td)
            rules[f"eitherW{tr}/{td}"] = lambda sd, tr=tr, td=td: either_rule(sd, ras[sd["name"]], tr, td, True)
    for bn, base in make_bases().items():
        for gn, gate in make_gates().items():
            for tc in ((AI_T,) if gate is None else (0.3, 0.4, 0.5)):
                name = bn + (f"+{gn}@{tc}" if gn else "")
                rules[name] = lambda sd, base=base, gate=gate, tc=tc: window_rule(sd, ras[sd["name"]], base, gate, tc)
    return rules


# ---- reporting -------------------------------------------------------------


def evaluate(samples, rule):
    return {name: (s["expected"], s["share"], rule(s)) for name, s in samples.items()}


def metrics(res):
    def count(prefix, exp):
        xs = [v for n, (e, _, v) in res.items() if e == exp and n.startswith(prefix)]
        return sum(v >= 0.5 for v in xs), len(xs)

    m = {}
    gem = next((v for n, (_, _, v) in res.items() if n.startswith("ai-faceit")), None)
    if gem is not None:
        m["gemini"] = gem
    for pre, exp, lab in [("ai-plain", "ai", "plainAI"), ("human-enron", "human", "enronFP"), ("human-wiki", "human", "wikiFP")]:
        k, n = count(pre, exp)
        if n:
            m[lab] = (k, n)
    clear = [(e, v) for n, (e, _, v) in res.items() if e in ("ai", "human") and not n.startswith(("edge/", "ai-plain", "ai-faceit", "human-enron", "human-wiki"))]
    if clear:
        m["clear"] = (sum((v >= 0.5) == (e == "ai") for e, v in clear), len(clear))
        m["humanFP"] = (sum(v >= 0.5 for e, v in clear if e == "human"), sum(e == "human" for e, _ in clear))
    mixed = [abs(v - sh) for n, (e, sh, v) in res.items() if e == "mixed"]
    if mixed:
        m["mixedErr"] = sum(mixed) / len(mixed) * 100
    return m


def fmt(m):
    parts = []
    if "gemini" in m:
        parts.append(f"gem {m['gemini'] * 100:3.0f}%")
    for k in ("plainAI", "enronFP", "wikiFP", "clear", "humanFP"):
        if k in m:
            parts.append(f"{k} {m[k][0]:2d}/{m[k][1]}")
    if "mixedErr" in m:
        parts.append(f"mixErr {m['mixedErr']:4.1f}")
    return "  ".join(parts)


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("-")]
    sets = []
    for i in range(0, len(files), 2):
        desk = load(files[i])
        for n, s in desk.items():
            s["name"] = n
        ras = load(files[i + 1]) if i + 1 < len(files) else None
        sets.append((desk, ras))

    names = None
    per_set = []
    for desk, ras in sets:
        rules = {"desk(engine)": lambda s: ai_share(s["sw"], spans(s, "win"))}
        if ras:
            rules.update(make_rules(desk, ras))
        names = list(rules)
        per_set.append({n: metrics(evaluate(desk, r)) for n, r in rules.items()})

    base_wiki = per_set[0]["desk(engine)"].get("wikiFP", (99, 0))[0]

    def key(n):
        ms = [ps[n] for ps in per_set]
        fp = sum(m.get(k, (0, 0))[0] for m in ms for k in ("enronFP", "wikiFP", "humanFP"))
        hit = sum(m.get("plainAI", (0, 0))[0] for m in ms) + sum(m.get("gemini", 0) >= 0.5 for m in ms)
        err = sum(m.get("mixedErr", 0) for m in ms)
        return (fp, -hit, err)

    def ok(n):
        m = per_set[0][n]
        return m.get("enronFP", (0, 0))[0] <= 1 and m.get("wikiFP", (0, 0))[0] <= base_wiki

    show = [n for n in sorted(names, key=key) if "--all" in sys.argv or ok(n) or n in ("desk(engine)", "desk", "max")]
    for n in show:
        print(f"{n:22s} " + "  |  ".join(fmt(ps[n]) for ps in per_set))

    if "-v" in sys.argv:
        for desk, ras in sets:
            rules = {"desk(engine)": lambda s: ai_share(s["sw"], spans(s, "win"))}
            if ras:
                rules.update(make_rules(desk, ras))
            for n in show[:12]:
                print(f"\n{n}")
                for name, (e, sh, v) in sorted(evaluate(desk, rules[n]).items()):
                    print(f"  {name:48s} {e:6s} {v * 100:4.0f}")


if __name__ == "__main__":
    main()
