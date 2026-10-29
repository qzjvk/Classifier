"""Throwaway: L1 variant comparison on the logic task (phase 1 only).

Compares the current 2-conv L1 (RF 5) against a 4-layer dilated causal
conv stack (RF 17) with different training recipes, all on the real
./dataset/ logic splits. Reports 3-class val accuracy + full-model size.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier import Classifier, L1ParsingModel, encode, MAXLEN, VOCAB, PAD
from _diag import read  # reuse jsonl reader

HERE = os.path.dirname(__file__)
DEV = "cuda"
BATCH = 512


class L1TCN(nn.Module):
    """Dilated causal conv stack. RF = 1 + sum(2*dilation).

    256-wide middle keeps the full model under the 12MB budget; a final
    linear projects back to the 512-d h1 the head/experts expect."""

    def __init__(self, emb_dim=256, hidden=256, dilations=(1, 1, 2, 4)):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 3, dilation=d) for d in dilations])
        self.proj = nn.Linear(hidden, 512)

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        for c in self.convs:
            h = F.relu(c(F.pad(h, (2 * c.dilation[0], 0))))
        mask = (x != PAD).float().unsqueeze(1)
        h = (h * mask).sum(2) / mask.sum(2).clamp(min=1.0)
        return self.proj(h)


def to_tensor(pairs):
    x = np.zeros((len(pairs), MAXLEN), dtype=np.int64)
    for i, (s, _) in enumerate(pairs):
        x[i] = encode(s)
    y = np.array([l for _, l in pairs], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def evaluate(l1, head, xv, yv):
    l1.eval()
    head.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(xv), 4096):
            xb, yb = xv[i:i + 4096].to(DEV), yv[i:i + 4096].to(DEV)
            correct += (head(l1(xb)).argmax(1) == yb).sum().item()
    return correct / len(xv)


def full_model_bytes(l1):
    m = Classifier(n_classes=8, n_slots=3)
    m.l1 = l1
    return sum(p.numel() * p.element_size() for p in m.parameters())


def run_variant(name, l1, x, y, xv, yv, epochs, lr, decay):
    torch.manual_seed(0)
    l1 = l1.to(DEV)
    head = nn.Linear(512, 3).to(DEV)
    opt = torch.optim.Adam(list(l1.parameters()) + list(head.parameters()), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.1) if decay else None)
    size = full_model_bytes(l1)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = x[idx].to(DEV), y[idx].to(DEV)
            opt.zero_grad()
            loss = F.cross_entropy(head(l1(xb)), yb)
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        print(f"  {name} ep{ep} loss={loss.item():.4f} ({time.time() - t0:.0f}s)",
              flush=True)
    acc = evaluate(l1, head, xv, yv)
    print(f"  {name}: logic val (3cls) = {acc:.4f}  "
          f"full-model size = {size / 1e6:.2f}MB", flush=True)
    del l1, head, opt, sched
    torch.cuda.empty_cache()
    return acc, size


def main():
    print("loading data...", flush=True)
    train, val = read("train.jsonl"), read("val.jsonl")
    x, y = to_tensor(train)
    xv, yv = to_tensor(val)
    print(f"  {len(train)} train / {len(val)} val", flush=True)

    results = []
    for name, l1, epochs, lr, decay in [
        ("A cur-2conv 4ep", L1ParsingModel(), 4, 1e-3, False),
        ("B cur-2conv 10ep+cos", L1ParsingModel(), 10, 1e-3, True),
        ("C TCN-4conv 4ep", L1TCN(), 4, 1e-3, False),
        ("D TCN-4conv 10ep+cos", L1TCN(), 10, 1e-3, True),
    ]:
        print(f"== {name} ==", flush=True)
        results.append((name,) + run_variant(name, l1, x, y, xv, yv, epochs, lr, decay))
    print("\n== summary ==")
    for name, acc, size in results:
        print(f"  {name:22} val={acc:.4f}  size={size / 1e6:.2f}MB")


if __name__ == "__main__":
    main()