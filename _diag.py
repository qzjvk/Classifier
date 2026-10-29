"""Throwaway error analysis: where does the model get val wrong?

For each domain: error count, confusion, length stats (errors vs correct),
and a sample of misclassified inputs. Usage:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python _diag.py [domain ...]
domains: logic, skill1, skill2   (default: all)
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

from classifier import Classifier, encode, MAXLEN, VOCAB

HERE = os.path.dirname(__file__)
DEV = "cuda"

FILES = {
    "logic": "val.jsonl",
    "skill1": "skill_val.jsonl",
    "skill2": "skill_val_2.jsonl",
}
NAMES = {0: "TAUT", 1: "CONTRAD", 2: "CONTINGENT", 3: "POS", 4: "NEG",
         5: "NEU", 6: "TRUE", 7: "FALSE"}


def load_model():
    state = torch.load(os.path.join(HERE, "classifier.pt"), map_location=DEV)
    n = state["head.weight"].shape[0]
    n_slots = state["l2.memory"].shape[0]
    m = Classifier(n_classes=n, n_slots=n_slots).to(DEV)
    m.load_state_dict(state)
    m.eval()
    return m


def read(name):
    out = []
    with open(os.path.join(HERE, "dataset", name)) as f:
        for line in f:
            d = json.loads(line)
            out.append((d["text"], int(d["label"])))
    return out


def to_tensor(pairs):
    x = np.zeros((len(pairs), MAXLEN), dtype=np.int64)
    for i, (s, _) in enumerate(pairs):
        x[i] = encode(s)
    y = np.array([l for _, l in pairs], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def analyze(m, name, pairs):
    x, y = to_tensor(pairs)
    lens = np.array([len(s) for s, _ in pairs])
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            preds.append(m(x[i:i + 4096].to(DEV), use_l2=True).argmax(1).cpu())
    pred = torch.cat(preds)
    ok = pred == y
    n_err = int((~ok).sum())
    print(f"== {name}: {n_err} / {len(y)} errors ==")
    if n_err == 0:
        print("  (no errors)")
        print()
        return
    conf = Counter((int(y[i]), int(pred[i])) for i in range(len(y)) if not ok[i])
    for (a, b), c in conf.most_common(8):
        print(f"  {NAMES.get(a, a):10} -> {NAMES.get(b, b):10}  {c}")
    el, cl = lens[~ok], lens[ok]
    print(f"  len: errors  mean {el.mean():.1f}  p50 {np.median(el):.0f}  "
          f"p90 {np.percentile(el, 90):.0f}  max {el.max()}")
    print(f"  len: correct mean {cl.mean():.1f}  p50 {np.median(cl):.0f}  "
          f"p90 {np.percentile(cl, 90):.0f}  max {cl.max()}")
    # bucket error rate by length
    for lo, hi in [(1, 15), (15, 30), (30, 45), (45, 60), (60, 97)]:
        sel = (lens >= lo) & (lens < hi)
        if sel.sum() == 0:
            continue
        err_rate = (~ok[sel]).float().mean()
        print(f"  len {lo:3}-{hi - 1:3}: err {err_rate:.3f} ({int((~ok[sel]).sum())}/{int(sel.sum())})")
    # samples: longest errors + a few short ones
    errs = [i for i in range(len(y)) if not ok[i]]
    errs.sort(key=lambda i: lens[i], reverse=True)
    short = [i for i in errs if lens[i] < 30][:5]
    shown = 0
    for i in errs[:10]:
        s = pairs[i][0]
        if len(s) > 60:
            s = s[:60] + "…"
        print(f"  ERR len={lens[i]:3} y={NAMES.get(y[i], y[i]):10} "
              f"p={NAMES.get(pred[i], pred[i]):10} {s!r}")
        shown += 1
    for i in short:
        print(f"  ERR len={lens[i]:3} y={NAMES.get(y[i], y[i]):10} "
              f"p={NAMES.get(pred[i], pred[i]):10} {pairs[i][0]!r}")
    print()


def main():
    which = [w for w in sys.argv[1:] if w in FILES] or list(FILES)
    m = load_model()
    for name in which:
        analyze(m, name, read(FILES[name]))


if __name__ == "__main__":
    main()