"""Compare candidate AI-text detectors against desklib on the eval sets.

Scores every sample at two granularities: the whole text (truncated to 512
tokens) and each paragraph (>= 15 words), since Tracer's windows stay inside
paragraphs. Prints per-sample scores and a summary per model.

Usage:  python -I compare_models.py [--download] [--only name,name]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import HF_DIR, load_torch_model  # noqa: E402

CAND = HERE / "work" / "candidates"
ROOT = HERE.parents[1]
SETS = {"samples": ROOT / "eval" / "samples", "hard": ROOT / "eval" / "hard"}

MODELS = {
    "desklib": None,  # the current model (tools/model-export/work/hf)
    "fakespot": "fakespot-ai/roberta-base-ai-text-detection-v1",
    "tmr": "Oxidane/tmr-ai-text-detector",
    "vanguard": "ShantanuT01/vanguard-ai-text-detector",
    "tabularis": "tabularisai/ai-text-detection",
    "rasbt": "rasbt/ai-text-detector-modernbert",
    "wasitai": "wasitaigeneratedcom/ai-text-detector-small",
}
AI_WORDS = ("ai", "machine", "generated", "fake", "llm", "gpt", "artificial")


def read_samples():
    out = []
    for set_name, d in SETS.items():
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.txt")):
            raw = p.read_text(encoding="utf-8")
            _, meta, body = raw.split("---", 2)
            m = {k.strip(): v.split("#")[0].strip() for k, v in (l.split(":", 1) for l in meta.strip().splitlines() if ":" in l)}
            if p.parent.name == "edge":
                continue
            out.append(dict(set=set_name, name=p.name, expected=m["expected"], body=body.strip()))
    return out


def paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if len(p.split()) >= 15]


class MultiHeadDetector(PreTrainedModel):
    """tropa-mini's reference class (model card): DeBERTa + masked mean pool + Linear(H, n)."""
    config_class = AutoConfig
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = torch.nn.Linear(config.hidden_size, getattr(config, "detector_num_labels", 1))

    def forward(self, input_ids, attention_mask):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).float()
        return self.classifier((h * m).sum(1) / m.sum(1))


class Scorer:
    def __init__(self, name):
        self.name = name
        if name == "desklib":
            self.tok = AutoTokenizer.from_pretrained(str(HF_DIR))
            self.model = load_torch_model()
            self.mode, self.desklib = "sigmoid", True
            return
        self.desklib = False
        path = CAND / name
        cfg = AutoConfig.from_pretrained(path)
        arch = (cfg.architectures or [""])[0]
        self.tok = AutoTokenizer.from_pretrained(path)
        if "Desklib" in arch and getattr(cfg, "detector_num_labels", 1) > 1:
            # tropa-mini: desklib's architecture with a 4-way head
            # (human / ai / ai_edited / humanized); score = 1 - P(human).
            self.model = MultiHeadDetector.from_pretrained(str(path)).eval()
            self.mode, self.desklib = "one-minus-human", True
            return
        if "Desklib" in arch:
            from common import DesklibAIDetectionModel
            self.model = DesklibAIDetectionModel.from_pretrained(str(path)).eval()
            self.mode, self.desklib = "sigmoid", True
            return
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        labels = {int(k): str(v) for k, v in (cfg.id2label or {}).items()}
        if cfg.num_labels == 1:
            self.mode, self.ai_idx = "sigmoid", None
        else:
            self.mode = "softmax"
            hits = [i for i, l in labels.items() if any(w in l.lower() for w in AI_WORDS) and "human" not in l.lower()]
            self.ai_idx = hits[0] if hits else 1
        print(f"  {name}: arch={arch} labels={labels} -> {self.mode} ai_idx={getattr(self, 'ai_idx', None)}")

    @torch.no_grad()
    def __call__(self, text):
        enc = self.tok(text, truncation=True, max_length=512, return_tensors="pt")
        if self.desklib:
            logit = self.model(enc["input_ids"], enc["attention_mask"])
            if self.mode == "one-minus-human":
                return float(1 - torch.softmax(logit, -1)[0, 0])
            return float(torch.sigmoid(logit)[0, 0])
        out = self.model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}).logits
        if self.mode == "sigmoid":
            return float(torch.sigmoid(out)[0, 0])
        return float(torch.softmax(out, -1)[0, self.ai_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = [n for n in MODELS if not a.only or n in a.only.split(",")]
    if a.download:
        for n in names:
            if MODELS[n]:
                snapshot_download(MODELS[n], local_dir=str(CAND / n),
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "spm*"])
                print("downloaded", n)
        return

    samples = read_samples()
    results = {}
    for n in names:
        torch.set_num_threads(8)
        s = Scorer(n)
        t = time.perf_counter()
        rows = []
        for smp in samples:
            doc = s(smp["body"])
            paras = [s(p) for p in paragraphs(smp["body"])]
            rows.append(dict(**{k: smp[k] for k in ("set", "name", "expected")}, doc=doc, paras=paras))
        results[n] = dict(rows=rows, secs=time.perf_counter() - t)
        del s
    (HERE / "work" / "compare-results.json").write_text(json.dumps(results, indent=1))

    # per-sample doc scores for the hard set + the AI/human samples
    print(f"\n{'sample':40s} {'exp':6s} " + " ".join(f"{n[:9]:>9s}" for n in names))
    for i, smp in enumerate(samples):
        if smp["expected"] == "mixed":
            continue
        print(f"{smp['set'][0]}:{smp['name'][:38]:38s} {smp['expected']:6s} " + " ".join(f"{results[n]['rows'][i]['doc']:9.2f}" for n in names))

    print("\nSummary (doc-level P(AI) >= 0.5 counts as AI; paragraph = each paragraph scored alone)")
    print(f"{'model':10s} {'hardAI doc':>10s} {'hardAI par':>10s} {'wiki FP doc':>11s} {'wiki FP par':>11s} {'orig AI':>8s} {'orig hum FP':>11s} {'secs':>6s}")
    for n in names:
        rows = results[n]["rows"]
        def frac(sel, key):
            vals = []
            for r in rows:
                if sel(r):
                    vals += [r["doc"]] if key == "doc" else r["paras"]
            return f"{sum(v >= 0.5 for v in vals)}/{len(vals)}"
        print(f"{n:10s} {frac(lambda r: r['set']=='hard' and r['expected']=='ai', 'doc'):>10s} "
              f"{frac(lambda r: r['set']=='hard' and r['expected']=='ai', 'par'):>10s} "
              f"{frac(lambda r: r['set']=='hard' and r['expected']=='human', 'doc'):>11s} "
              f"{frac(lambda r: r['set']=='hard' and r['expected']=='human', 'par'):>11s} "
              f"{frac(lambda r: r['set']=='samples' and r['expected']=='ai', 'doc'):>8s} "
              f"{frac(lambda r: r['set']=='samples' and r['expected']=='human', 'doc'):>11s} "
              f"{results[n]['secs']:6.1f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
