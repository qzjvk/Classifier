"""Throwaway smoke test for the multi-slot L2."""
import torch
from classifier import Classifier, encode, MAXLEN, VOCAB

torch.manual_seed(0)
n_classes, n_slots = 8, 3
m = Classifier(n_classes=n_classes, n_slots=n_slots)
print("params", sum(p.numel() for p in m.parameters()),
      "bytes", sum(p.numel()*p.element_size() for p in m.parameters())/1e6, "MB")

x = torch.randint(1, len(VOCAB), (5, MAXLEN))
# inference
print("fwd use_l2", m(x, use_l2=True).shape)
print("fwd no_l2 ", m(x, use_l2=False).shape)
# batch-1 fast path
print("fwd b1    ", m(x[:1], use_l2=True).shape)
# training path (teacher-forced)
dom = torch.tensor([0, 1, 2, 1, 0])
logits, g = m.forward_train(x, dom, use_l2=True)
print("train logits", logits.shape, "router", g.shape)
loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 8, (5,))) \
     + 0.1 * torch.nn.functional.cross_entropy(g, dom)
loss.backward()
print("loss", loss.item(), "grad ok:",
      all(p.grad is not None for p in m.parameters() if p.requires_grad))

# identity at init: L2 residual should be ~0 (res zero-init)
m2 = Classifier(n_classes=n_classes, n_slots=n_slots).eval()
with torch.no_grad():
    h1 = m2.l1(x)
    h2 = m2.l2(h1, x)
print("init residual max abs:", (h2 - h1).abs().max().item())

# predict_rich
r = m2.predict_rich("the movie was great")
print("predict_rich keys:", sorted(r.keys()), "label", r["label"])

# state dict round-trip
sd = m.state_dict()
m3 = Classifier(n_classes=n_classes, n_slots=n_slots)
m3.load_state_dict(sd)
print("round-trip ok; n_slots from sd:", sd["l2.memory"].shape)
print("expert count:", len(m3.l2.experts))