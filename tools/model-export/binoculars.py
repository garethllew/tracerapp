"""Zero-shot "Binoculars" detector (Hans et al., 2024) with small CPU models.

score = log-perplexity of the text under the performer model
        / cross-perplexity between observer and performer predictions.
LOW score = AI. No training on any generator, so it should generalise to
new models and house styles better than fine-tuned classifiers.

Adds "bino-<tag>" rows to work/compare-results.json, as P(AI)-like values
(1 - sigmoid((score - centre) * 12)), so analyse_compare.py can rank it.

Usage:  python -I binoculars.py [--pair qwen0.5|qwen1.5]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_models import paragraphs, read_samples  # noqa: E402

PAIRS = {
    "qwen0.5": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen1.5": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
}


@torch.no_grad()
def score(text, tok, obs, perf, max_tokens=512):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    ids = enc["input_ids"]
    if ids.shape[1] < 3:
        return float("nan")
    lo = obs(**enc).logits[:, :-1].float()
    lp = perf(**enc).logits[:, :-1].float()
    target = ids[:, 1:]
    # log-perplexity of the text under the performer
    nll = torch.nn.functional.cross_entropy(lp.transpose(1, 2), target, reduction="mean")
    # cross-perplexity: observer's distribution scored by the performer
    x = -(torch.softmax(lo, -1) * torch.log_softmax(lp, -1)).sum(-1).mean()
    return float(nll / x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="qwen0.5", choices=PAIRS)
    a = ap.parse_args()
    obs_id, perf_id = PAIRS[a.pair]
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(obs_id)
    obs = AutoModelForCausalLM.from_pretrained(obs_id, torch_dtype=torch.float32).eval()
    perf = AutoModelForCausalLM.from_pretrained(perf_id, torch_dtype=torch.float32).eval()

    samples = read_samples()
    t = time.perf_counter()
    raw = []
    for s in samples:
        raw.append(dict(doc=score(s["body"], tok, obs, perf), paras=[score(p, tok, obs, perf) for p in paragraphs(s["body"])]))
    secs = time.perf_counter() - t

    # Centre the squashing on the midpoint between the medians of AI and real-human docs.
    ai = sorted(r["doc"] for r, s in zip(raw, samples) if s["expected"] == "ai")
    hu = sorted(r["doc"] for r, s in zip(raw, samples) if s["set"] == "hard" and s["expected"] == "human")
    centre = (ai[len(ai) // 2] + hu[len(hu) // 2]) / 2
    p = lambda v: 0.5 if math.isnan(v) else 1 / (1 + math.exp((v - centre) * 12))
    rows = [dict(**{k: s[k] for k in ("set", "name", "expected")}, doc=p(r["doc"]), paras=[p(v) for v in r["paras"]], raw=r["doc"])
            for r, s in zip(raw, samples)]
    path = HERE / "work" / "compare-results.json"
    res = json.loads(path.read_text())
    res[f"bino-{a.pair}"] = dict(rows=rows, secs=secs, centre=centre)
    path.write_text(json.dumps(res, indent=1))
    print(f"bino-{a.pair}: {len(samples)} samples in {secs:.0f}s; median raw AI {ai[len(ai)//2]:.3f} vs human {hu[len(hu)//2]:.3f}; centre {centre:.3f}")
    for r in rows:
        if r["set"] == "hard" or r["name"].startswith("ai-"):
            print(f"  {r['name'][:40]:40s} {r['expected']:6s} raw {r['raw']:.3f}  p {r['doc']:.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
