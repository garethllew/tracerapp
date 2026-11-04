"""Phase-0 spike: parity, document-level eval and CPU latency.

1. Tokenizer check: `tokenizers` (what the Rust crate uses, via tokenizer.json)
   must give the same ids as transformers' AutoTokenizer.
2. Parity: P(AI) per eval sample from PyTorch vs each ONNX variant.
3. Doc-level eval: whole-text P(AI) vs the sample's expected label.
4. Latency: load time, and a 500-word text scored as ~60-word windows
   (batched) plus a whole-document pass, at 16 and 4 threads.

Usage:  python parity_bench.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from common import HF_DIR, OUT_DIR, load_torch_model

SAMPLES = Path(__file__).resolve().parents[2] / "eval" / "samples"
VARIANTS = ["tracer-fp32.onnx", "tracer-int8.onnx"]
MAX_LEN = 512


def read_samples():
    out = []
    for p in sorted(SAMPLES.rglob("*.txt")):
        raw = p.read_text(encoding="utf-8")
        _, meta, body = raw.split("---", 2)
        m = dict(l.split(":", 1) for l in meta.strip().splitlines() if ":" in l)
        m = {k.strip(): v.split("#")[0].strip() for k, v in m.items()}
        out.append((p.relative_to(SAMPLES).as_posix(), m, body.strip()))
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def encode(tok, texts):
    tok.enable_truncation(MAX_LEN)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    enc = tok.encode_batch(texts)
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    return ids, mask


def session(name, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t = time.perf_counter()
    s = ort.InferenceSession(str(OUT_DIR / name), so, providers=["CPUExecutionProvider"])
    return s, time.perf_counter() - t


def run(sess, ids, mask):
    return sigmoid(sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0][:, 0])


def windows_of(text, words=60, stride=30):
    w = text.split()
    return [" ".join(w[i:i + words]) for i in range(0, max(1, len(w) - stride), stride)]


def main():
    samples = read_samples()
    texts = [b for _, _, b in samples]
    fast = Tokenizer.from_file(str(OUT_DIR / "tokenizer.json"))
    ref = AutoTokenizer.from_pretrained(str(HF_DIR))

    # 1. tokenizer parity
    mism = 0
    for t in texts:
        a = fast.encode(t).ids[:MAX_LEN]
        b = ref(t, truncation=True, max_length=MAX_LEN)["input_ids"]
        mism += a != b
    print(f"tokenizer.json vs AutoTokenizer: {len(texts) - mism}/{len(texts)} identical")

    # 2. parity, scored one text at a time (no padding effects)
    model = load_torch_model()
    probs = {"torch": []}
    with torch.no_grad():
        for t in texts:
            ids, mask = encode(fast, [t])
            probs["torch"].append(float(torch.sigmoid(model(torch.tensor(ids), torch.tensor(mask)))[0, 0]))
    del model
    for v in VARIANTS:
        s, _ = session(v, 16)
        probs[v] = [float(run(s, *encode(fast, [t]))[0]) for t in texts]
    print("\nmax / mean |ΔP(AI)| vs PyTorch:")
    ref_p = np.array(probs["torch"])
    for v in VARIANTS:
        d = np.abs(np.array(probs[v]) - ref_p)
        flips = int(((np.array(probs[v]) >= 0.5) != (ref_p >= 0.5)).sum())
        print(f"  {v:22s} max {d.max():.4f}  mean {d.mean():.4f}  label flips {flips}")

    # 3. doc-level eval table
    print(f"\n{'sample':38s} {'expected':9s} {'share':>5s} {'words':>5s} " + " ".join(f"{k[:12]:>12s}" for k in probs))
    for i, (name, m, body) in enumerate(samples):
        row = " ".join(f"{probs[k][i]:12.3f}" for k in probs)
        print(f"{name:38s} {m['expected']:9s} {float(m['expected_ai_share']):5.2f} {len(body.split()):5d} {row}")
    print("\ndoc-level accuracy (human<0.5, ai>=0.5; mixed + edge excluded):")
    for k in probs:
        ok = n = 0
        for i, (name, m, _) in enumerate(samples):
            if name.startswith("edge/") or m["expected"] == "mixed":
                continue
            n += 1
            ok += (probs[k][i] >= 0.5) == (m["expected"] == "ai")
        print(f"  {k:22s} {ok}/{n}")

    # 4. latency
    long_ai = " ".join(b for n, m, b in samples if m["expected"] == "ai")
    doc = " ".join(long_ai.split()[:500])
    wins = windows_of(doc)
    w_ids, w_mask = encode(fast, wins)
    d_ids, d_mask = encode(fast, [doc])
    print(f"\nlatency: 500-word text -> {len(wins)} windows (batch, padded to {w_ids.shape[1]} tok) "
          f"+ whole-doc pass ({d_ids.shape[1]} tok)")
    bench = {}
    for threads in (16, 8, 4):
        for v in VARIANTS:
            s, load = session(v, threads)
            run(s, w_ids[:1], w_mask[:1])  # warm-up
            tw, td = [], []
            for _ in range(5):
                t = time.perf_counter(); run(s, w_ids, w_mask); tw.append(time.perf_counter() - t)
                t = time.perf_counter(); run(s, d_ids, d_mask); td.append(time.perf_counter() - t)
            w, d = statistics.median(tw), statistics.median(td)
            bench[f"{v}@{threads}"] = dict(load_s=load, windows_s=w, doc_s=d, total_s=w + d)
            print(f"  {threads:2d} threads {v:22s} load {load:5.2f}s  windows {w:5.2f}s  doc {d:5.2f}s  total {w + d:5.2f}s")
            del s
    (OUT_DIR / "spike-results.json").write_text(json.dumps(dict(probs=probs, bench=bench,
        samples=[n for n, _, _ in samples]), indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
