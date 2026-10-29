"""Verify trained model: coexistence, router, latency, per-op profile."""
import os, time
import torch
from classifier import Classifier, encode, MAXLEN, VOCAB
from data import make_splits

HERE = os.path.dirname(__file__)
DEV = "cuda"


def load():
    state = torch.load(os.path.join(HERE, "classifier.pt"), map_location=DEV)
    n = state["head.weight"].shape[0]
    n_slots = state["l2.memory"].shape[0]
    m = Classifier(n_classes=n, n_slots=n_slots).to(DEV)
    m.load_state_dict(state)
    m.eval()
    return m


def to_tensor(pairs):
    import numpy as np
    x = np.zeros((len(pairs), MAXLEN), dtype=np.int64)
    for i, (s, _) in enumerate(pairs):
        x[i] = encode(s)
    y = np.array([l for _, l in pairs], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def acc(m, x, y, use_l2=True):
    m.eval()
    c = 0
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            c += (m(x[i:i+4096].to(DEV), use_l2=use_l2).argmax(1)
                  == y[i:i+4096].to(DEV)).sum().item()
    return c / len(x)


def bench(m, x, use_l2, n=3000, warm=200):
    with torch.no_grad():
        for _ in range(warm):
            m(x, use_l2=use_l2)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            m(x, use_l2=use_l2)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def main():
    m = load()
    # load val sets
    import json
    def rd(p):
        out = []
        with open(os.path.join(HERE, "dataset", p)) as f:
            for line in f:
                d = json.loads(line)
                out.append((d["text"], int(d["label"])))
        return out
    xv, yv = to_tensor(rd("val.jsonl"))
    x1, y1 = to_tensor(rd("skill_val.jsonl"))
    x2, y2 = to_tensor(rd("skill_val_2.jsonl"))

    print("== accuracy (L1+L2) ==")
    print(f"logic : {acc(m, xv, yv):.4f}")
    print(f"skill1: {acc(m, x1, y1):.4f}")
    print(f"skill2: {acc(m, x2, y2):.4f}")
    print("== accuracy (L1 only) ==")
    print(f"logic : {acc(m, xv, yv, use_l2=False):.4f}")
    print(f"skill1: {acc(m, x1, y1, use_l2=False):.4f}")
    print(f"skill2: {acc(m, x2, y2, use_l2=False):.4f}")

    print("== router accuracy ==")
    for name, xx, dom in [("logic", xv, 0), ("skill1", x1, 1), ("skill2", x2, 2)]:
        d = torch.full((len(xx),), dom, dtype=torch.long)
        c = 0
        with torch.no_grad():
            for i in range(0, len(xx), 4096):
                g = m.l2._route(xx[i:i+4096].to(DEV))
                c += (g.argmax(1) == d[i:i+4096].to(DEV)).sum().item()
        print(f"{name}: {c/len(xx):.4f}")

    print("== latency (us, batch 1) ==")
    x1b = torch.randint(1, len(VOCAB), (1, MAXLEN), device=DEV)
    print(f"L1 only : {bench(m, x1b, False):8.1f}")
    print(f"L1 + L2 : {bench(m, x1b, True):8.1f}")

    print("== per-op profile (L1+L2, 50 iters) ==")
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        for _ in range(20):
            m(x1b, use_l2=True)
        torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            for _ in range(50):
                m(x1b, use_l2=True)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("== predict_rich samples ==")
    for t in ["p | ~p", "the movie was great", "3+4=7", "i hated the trip", "5*6<30"]:
        r = m.predict_rich(t)
        print(f"  {t!r:30} -> {r['name']:10} conf={r['confidence']:.2f} "
              f"l1={r['l1']['name']} changed={r['l2_changed']}")

    print("== head logits + base-expert residual on logic ==")
    with torch.no_grad():
        for t in ["p | ~p", "p & ~p", "q -> r"]:
            xi = torch.tensor([encode(t)], device=DEV)
            h1 = m.l1(xi)
            logits_l1 = m.head(h1)[0]
            h2 = m.l2(h1, xi)
            logits_l2 = m.head(h2)[0]
            r0 = (h2 - h1)[0]
            print(f"  {t!r:12} L1-only argmax={int(logits_l1.argmax())} "
                  f"L1+L2 argmax={int(logits_l2.argmax())} "
                  f"|residual|={r0.abs().max().item():.4f}")


if __name__ == "__main__":
    main()