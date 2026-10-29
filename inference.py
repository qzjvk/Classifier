"""Inference: load classifier.pt and classify text -> label + breakdown.

Usage:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python inference.py "p | ~p"
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python inference.py --json "the movie was great"
    echo "p & ~p" | CUDA_VISIBLE_DEVICES=2 .venv/bin/python inference.py -

Default: one line per text -> text, label, name, confidence, and what the L1
base parser alone decided (l1-agrees / l2-flipped). --json prints the full
rich payload (probabilities, L1 breakdown, margin, latency).
"""

import argparse
import json
import os
import sys
import time

import torch

from classifier import Classifier

HERE = os.path.dirname(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*", help="text to classify; '-' reads stdin")
    ap.add_argument("--json", action="store_true",
                    help="print the full rich JSON payload per text")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(os.path.join(HERE, "classifier.pt"), map_location=device)
    n = state["head.weight"].shape[0]
    n_slots = state["l2.memory"].shape[0]
    model = Classifier(n_classes=n, n_slots=n_slots).to(device)
    model.load_state_dict(state)
    model.eval()

    if args.texts and args.texts[0] != "-":
        texts = args.texts
    else:
        texts = [l.strip() for l in sys.stdin if l.strip()]

    for t in texts:
        t0 = time.perf_counter()
        r = model.predict_rich(t)
        torch.cuda.synchronize()
        r["ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if args.json:
            print(json.dumps(r))
        else:
            flip = "l2-flipped" if r["l2_changed"] else "l1-agrees"
            print(f"{t}\t{r['label']}\t{r['name']}\t"
                  f"conf={r['confidence']:.2f}\tl1={r['l1']['name']} {flip}")


if __name__ == "__main__":
    main()