"""Throwaway latency diagnosis: L1-only vs L1+L2, per-op profile."""
import os, sys, time
import torch

from classifier import Classifier, encode, MAXLEN, VOCAB

HERE = os.path.dirname(__file__)


def load(device):
    state = torch.load(os.path.join(HERE, "classifier.pt"), map_location=device)
    n = state["head.weight"].shape[0]
    n_slots = state["l2.memory"].shape[0]
    model = Classifier(n_classes=n, n_slots=n_slots).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def bench(model, x, use_l2, n=3000, warm=200):
    with torch.no_grad():
        for _ in range(warm):
            model(x, use_l2=use_l2)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x, use_l2=use_l2)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6  # us


def main():
    device = "cuda"
    model = load(device)
    x = torch.randint(1, len(VOCAB), (1, MAXLEN), device=device)

    print("== batch-1 latency (us) ==")
    print(f"L1 only   : {bench(model, x, False):8.1f}")
    print(f"L1 + L2   : {bench(model, x, True):8.1f}")

    # predict_rich path: two forwards
    with torch.no_grad():
        for _ in range(200):
            model.predict_rich("p | ~p")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(300):
            model.predict_rich("p | ~p")
        torch.cuda.synchronize()
    print(f"predict_rich: {(time.perf_counter()-t0)/300*1e6:8.1f}  (2 forwards)")

    print("\n== per-op profile (L1+L2, 50 iters) ==")
    with torch.no_grad():
        for _ in range(20):
            model(x, use_l2=True)
        torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            for _ in range(50):
                model(x, use_l2=True)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    # CPU comparison
    print("== CPU latency (us) ==")
    mc = model.cpu()
    xc = x.cpu()
    def bench_c(m, xx, use_l2, n=500, warm=50):
        with torch.no_grad():
            for _ in range(warm):
                m(xx, use_l2=use_l2)
            t0 = time.perf_counter()
            for _ in range(n):
                m(xx, use_l2=use_l2)
            return (time.perf_counter() - t0) / n * 1e6
    print(f"CPU L1 only : {bench_c(mc, xc, False):8.1f}")
    print(f"CPU L1 + L2 : {bench_c(mc, xc, True):8.1f}")


if __name__ == "__main__":
    main()