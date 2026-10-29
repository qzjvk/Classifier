"""Training pipeline: L1 (logic) -> skills (replay) -> SVD -> bench/save.

Run: CUDA_VISIBLE_DEVICES=2 .venv/bin/python train.py   (GTX 1070)

Data: if ./dataset/{train,val,skill*.jsonl,skill_val*.jsonl} exist (lines:
{"text","label"}), they are used; otherwise synthetic logic + arithmetic
data is generated as a stand-in. Multiple skill files (skill.jsonl,
skill_2.jsonl, ...) train multiple coexisting skills — each gets its own L2
expert + memory slot, so they don't overwrite each other.
"""

import os
import re
import json
from collections import Counter
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier import Classifier, encode, MAXLEN, VOCAB, LABELS
from data import make_splits, TAUT, CONTRAD, CONTINGENT, TRUE, FALSE

DEVICE = "cuda"
BATCH = 512
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
ROUTER_W = 0.1  # weight on the domain-router CE loss


def to_tensor(pairs):
    x = np.zeros((len(pairs), MAXLEN), dtype=np.int64)
    for i, (s, _) in enumerate(pairs):
        x[i] = encode(s)
    y = np.array([l for _, l in pairs], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def _skill_files(prefix):
    """Find <prefix>.jsonl, <prefix>_2.jsonl, <prefix>_3.jsonl, ... in order
    (no suffix = skill 1)."""
    files = {}
    pat = re.compile(re.escape(prefix) + r"(_(\d+))?\.jsonl")
    for name in os.listdir(DATASET_DIR):
        m = pat.fullmatch(name)
        if m:
            files[int(m.group(2)) if m.group(2) else 1] = name
    return [files[i] for i in sorted(files)]


def load_dataset():
    """Return (train, val, skills, skills_val).

    skills / skills_val are lists of (name, pairs), one per skill file
    (skill.jsonl = skill 1, skill_2.jsonl = skill 2, ...). Real dataset
    (JSONL) if the base task + at least one skill exist, else the synthetic
    logic + arithmetic stand-in (one skill)."""
    def read_jsonl(name):
        path = os.path.join(DATASET_DIR, name)
        if not os.path.exists(path):
            return None
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append((d["text"], int(d["label"])))
        return out

    train = read_jsonl("train.jsonl")
    val = read_jsonl("val.jsonl")
    skill_files = _skill_files("skill")
    val_files = _skill_files("skill_val")
    if train and val and skill_files and val_files:
        skills = [(f, read_jsonl(f)) for f in skill_files]
        skills_val = [(f, read_jsonl(f)) for f in val_files]
        print(f"using real dataset from ./dataset/ "
              f"({len(skills)} skill(s): {[f for f, _ in skills]})")
        return train, val, skills, skills_val
    print("no ./dataset/ found -> generating synthetic stand-in")
    train, val, skill_train, skill_val = make_splits()
    return train, val, [("skill", skill_train)], [("skill_val", skill_val)]


def evaluate(model, x, y, use_l2=True):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            xb, yb = x[i:i + 4096].to(DEVICE), y[i:i + 4096].to(DEVICE)
            correct += (model(xb, use_l2=use_l2).argmax(1) == yb).sum().item()
    return correct / len(x)


def router_accuracy(model, x, dom):
    """Fraction of rows the domain router assigns to the correct slot."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            g = model.l2._route(x[i:i + 4096].to(DEVICE))
            correct += (g.argmax(1) == dom[i:i + 4096].to(DEVICE)).sum().item()
    return correct / len(x)


def train_l1(model, x, y, n_classes, epochs=4, lr=1e-3):
    """Train L1 + n-class head on logic. L2 stays identity (frozen)."""
    model.head = nn.Linear(512, n_classes).to(DEVICE)
    model.train()
    for p in model.l2.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(model.l1.parameters()) + list(model.head.parameters()), lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = x[idx].to(DEVICE), y[idx].to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb, use_l2=False), yb)
            loss.backward()
            opt.step()
        print(f"  L1 ep{ep} loss={loss.item():.4f} ({time.time()-t0:.1f}s)")
    print(f"  L1 train acc={evaluate(model, x, y, use_l2=False):.4f}")
    return model


def train_skill(model, xs, ys, sdom, rx, ry, rdom, n_skill_classes,
                n_logic, epochs=20, lr=1e-3):
    """Train skill (domain k) into its own expert, with replay of the base
    task + all previous skills. Head expands n -> n + n_skill (old rows
    copied, new rows ~0-init). L1 frozen. Each batch is 50% new skill + 50%
    replay; the router is trained by domain CE. Teacher-forced expert
    selection means each expert is updated only by its own domain.
    The head's base-task rows (0..n_logic) are frozen: the softmax over all
    classes otherwise lets skill gradients pull the base rows off h1, and the
    base expert then absorbs a large compensating residual (which breaks the
    L1-only path). Freezing keeps base rows tuned for h1 alone."""
    old = model.head
    n_old = old.out_features
    new = nn.Linear(512, n_old + n_skill_classes).to(DEVICE)
    with torch.no_grad():
        new.weight[:n_old, :] = old.weight
        new.bias[:n_old] = old.bias
        nn.init.normal_(new.weight[n_old:, :], 0, 1e-3)
        nn.init.zeros_(new.bias[n_old:])
    model.head = new
    model.l1.requires_grad_(False)
    for p in model.l2.parameters():
        p.requires_grad_(True)
    model.train()
    opt = torch.optim.Adam(list(model.l2.parameters()) + list(model.head.parameters()), lr=lr)
    half = BATCH // 2
    n_replay = len(rx)
    t0 = time.time()
    for ep in range(epochs):
        perm_s = torch.randperm(len(xs))
        perm_r = torch.randperm(n_replay)
        for i in range(0, len(xs), half):
            si = perm_s[i:i + half]
            ri = perm_r[i:i + half]
            xb = torch.cat([xs[si], rx[ri]]).to(DEVICE)
            yb = torch.cat([ys[si], ry[ri]]).to(DEVICE)
            db = torch.cat([sdom[si], rdom[ri]]).to(DEVICE)
            opt.zero_grad()
            logits, router_logits = model.forward_train(xb, db, use_l2=True)
            loss = (F.cross_entropy(logits, yb)
                    + ROUTER_W * F.cross_entropy(router_logits, db))
            loss.backward()
            # freeze the head's base-task rows: zero their gradients so skill
            # + router gradients can't pull them off h1 (see docstring).
            if model.head.weight.grad is not None:
                model.head.weight.grad[:n_logic].zero_()
                model.head.bias.grad[:n_logic].zero_()
            opt.step()
        print(f"  skill ep{ep} loss={loss.item():.4f} ({time.time()-t0:.1f}s)")
    return model


def _size(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def svd_compress(model, target_bytes=12 * 1024 * 1024):
    """SVD low-rank pass over L1's conv weights. No-op when already under
    the budget. Note: real param savings require factorized (unmaterialized)
    U_r/Vt_r storage; a full-matrix reconstruction saves nothing, so this
    only runs if an explicit tighter budget is requested."""
    if _size(model) <= target_bytes:
        print(f"  SVD: model {_size(model)/1e6:.2f}MB <= {target_bytes/1e6:.0f}MB, skipping")
        return model
    for i, mod in reversed(list(enumerate(model.l1.convs))):
        if _size(model) <= target_bytes:
            break
        w = mod.weight
        out_c, in_c, k = w.shape
        mat = w.reshape(out_c, in_c * k)
        U, S, Vt = torch.linalg.svd(mat, full_matrices=False)
        r = min(out_c, in_c * k)
        for cand in range(r // 2, 0, -32):
            w_new = (U[:, :cand] * S[:cand]).mm(Vt[:cand])
            saved = (w.numel() - w_new.numel()) * w.element_size()
            if _size(model) - saved <= target_bytes:
                r = cand
                break
        w_new = (U[:, :r] * S[:r]).mm(Vt[:r])
        mod.weight = nn.Parameter(w_new.reshape(out_c, in_c, k))
        print(f"  SVD: conv{i} rank {min(out_c, in_c * k)} -> {r}")
    print(f"  SVD: final size {_size(model)/1e6:.2f}MB")
    return model


def bench_latency(model, n=2000):
    model.eval()
    x = torch.randint(1, len(VOCAB), (1, MAXLEN), device=DEVICE)
    with torch.no_grad():
        for _ in range(100):
            model(x, use_l2=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x, use_l2=True)
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"  latency: {dt*1000:.1f} us/request (batch 1, {n} iters)")
    return dt


def main():
    torch.manual_seed(0)
    print("loading data...")
    t0 = time.time()
    train, val, skills, skills_val = load_dataset()
    n_logic = int(max(l for _, l in train) + 1)
    skill_info = []
    for name, pairs in skills:
        classes = sorted(set(l for _, l in pairs))
        skill_info.append((name, pairs, classes))
    n_classes_total = n_logic + sum(len(c) for _, _, c in skill_info)
    n_slots = 1 + len(skill_info)
    print(f"  {len(train)} train / {len(val)} val / "
          + " / ".join(f"{len(p)} {len(c)}cls {n}"
                       for n, p, c in skill_info)
          + f"  ({n_logic} logic + {n_slots - 1} skills, "
          f"{n_classes_total} classes, {n_slots} slots) in {time.time()-t0:.1f}s")
    x, y = to_tensor(train)
    xv, yv = to_tensor(val)
    dom = torch.zeros(len(x), dtype=torch.long)      # logic = domain 0
    xdv = torch.zeros(len(xv), dtype=torch.long)

    model = Classifier(n_classes=n_logic, n_slots=n_slots).to(DEVICE)
    print(f"  model: {sum(p.numel() for p in model.parameters())/1e6:.3f}M params, "
          f"{_size(model)/1e6:.2f}MB")

    print("L1: parsing model, trained on logic...")
    model = train_l1(model, x, y, n_logic)
    print(f"  logic val (L1 only): {evaluate(model, xv, yv, use_l2=False):.4f}")

    # Base expert (slot 0) is identity at init; L1 handles the base task.
    # Phase 3's replay keeps it that way, so no distill phase is needed.
    print(f"  logic val (L1+L2, base identity): {evaluate(model, xv, yv, use_l2=True):.4f}")

    # Skills, sequential, each with replay of the base task + prior skills.
    replay_x, replay_y, replay_dom = x, y, dom
    for k, (name, pairs, classes) in enumerate(skill_info, start=1):
        xs, ys = to_tensor(pairs)
        xsv, ysv = to_tensor(skills_val[k - 1][1])
        sdom = torch.full((len(xs),), k, dtype=torch.long)
        print(f"skill {k} ({name}): {len(pairs)} train, classes {classes}, "
              f"replay {len(replay_x)}...")
        model = train_skill(model, xs, ys, sdom, replay_x, replay_y, replay_dom,
                            len(classes), n_logic=n_logic)
        print(f"  logic val (after skill {k}): {evaluate(model, xv, yv, use_l2=True):.4f}")
        print(f"  skill {k} val (L1 only):     {evaluate(model, xsv, ysv, use_l2=False):.4f}")
        print(f"  skill {k} val (L1+L2):       {evaluate(model, xsv, ysv, use_l2=True):.4f}")
        # add this skill to the replay pool for the next skill
        replay_x = torch.cat([replay_x, xs])
        replay_y = torch.cat([replay_y, ys])
        replay_dom = torch.cat([replay_dom, sdom])

    print("router domain accuracy (val):")
    print(f"  logic (dom 0):  {router_accuracy(model, xv, xdv):.4f}")
    for k, (name, pairs, classes) in enumerate(skill_info, start=1):
        xsv, ysv = to_tensor(skills_val[k - 1][1])
        sdomv = torch.full((len(xsv),), k, dtype=torch.long)
        print(f"  skill {k} (dom {k}): {router_accuracy(model, xsv, sdomv):.4f}")

    print("SVD compression...")
    model = svd_compress(model)
    print(f"  logic val (after SVD): {evaluate(model, xv, yv, use_l2=True):.4f}")
    for k, (name, pairs, classes) in enumerate(skill_info, start=1):
        xsv, ysv = to_tensor(skills_val[k - 1][1])
        print(f"  skill {k} val (after SVD): {evaluate(model, xsv, ysv, use_l2=True):.4f}")

    print("latency...")
    bench_latency(model)

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "classifier.pt"))
    print(f"saved classifier.pt ({_size(model)/1e6:.2f}MB)")


if __name__ == "__main__":
    main()