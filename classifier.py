"""Two-layer classifier: L1 parsing model (logic) + L2 skills parser.

Not an LLM: plain NN, no transformer / attention / BERT-style encoder.
L1: causal conv stack over char embeddings (unidirectional).
L2: skills parser. A small router over the raw chars picks one of K
    per-skill experts (a tiny top-1 MoE). Each expert is an MLP over
    [L1 rep; slot memory; its own raw-char features] that adds a zero-init
    residual to L1's representation. Multi-slot memory: each skill owns a
    memory vector + expert + char encoder, so training a new skill never
    overwrites an old one.
Head: single linear layer -> labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# char vocabulary: logic + arithmetic symbols, digits, full lowercase
# alphabet + light punctuation (text skill)
VOCAB = [
    " ", "(", ")", "&", "|", "~", "^", ">", "=", "+", "-", "*", "/",
    "<", "!", ",", ".", "?", "'",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
PAD = 0  # " " is pad id; inputs never contain a meaningful space
MAXLEN = 96
LABELS = ["TAUT", "CONTRAD", "CONTINGENT", "POS", "NEG", "NEU", "TRUE", "FALSE"]


def _label_name(i: int) -> str:
    """Display name for a class index; CLASS_i past the known LABELS."""
    return LABELS[i] if i < len(LABELS) else f"CLASS_{i}"


def encode(text: str) -> list[int]:
    ids = [CHAR2ID[c] for c in text[:MAXLEN]]
    ids += [PAD] * (MAXLEN - len(ids))
    return ids


class L1ParsingModel(nn.Module):
    """Parsing model trained on logic. Causal convs only (no attention).

    Dilated causal conv stack (TCN-style): receptive field
    1 + sum(2*dilation) = 17 chars at the default dilations (1, 1, 2, 4).
    256-wide layers keep the full model under the 12MB budget; a final
    linear projects the pooled rep to the 512-d h1 the head and experts
    expect. The old 2-conv RF-5 stack collapsed on long, deeply nested
    formulas (the hard part of the logic task); on the real dataset the
    4ep phase-1 run gets 95.2% val with RF-5 vs 97.4% with this stack
    (see _exp_l1.py for the ablation), ~1MB smaller in the process."""

    def __init__(self, emb_dim=256, hidden=256, out_dim=512,
                 dilations=(1, 1, 2, 4)):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, kernel_size=3, dilation=d)
             for d in dilations])
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # x: (B, L) char ids -> (B, out_dim); causal via left-only padding
        h = self.emb(x).transpose(1, 2)            # (B, H, L)
        for c in self.convs:
            h = F.relu(c(F.pad(h, (2 * c.dilation[0], 0))))  # (B, H, L)
        mask = (x != PAD).float().unsqueeze(1)     # (B, 1, L)
        h = (h * mask).sum(2) / mask.sum(2).clamp(min=1.0)
        return self.proj(h)


class _SkillTextEncoder(nn.Module):
    """Tiny char conv encoder owned by one expert.

    L1 is a *logic* parser, frozen during skill training: its pooled rep h1
    never learned what digits or sentiment words mean. This encoder gives an
    expert direct access to the raw characters, so a skill can be learned from
    h1 + text instead of asking an adapter to recover char detail from h1.
    Receptive field 3 + 2*2 = 7 chars, no attention.
    """

    def __init__(self, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), emb_dim, padding_idx=PAD)
        self.conv1 = nn.Conv1d(emb_dim, hidden, 3)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, dilation=2)

    def forward(self, x):
        # x: (B, L) char ids -> (B, hidden)
        h = self.emb(x).transpose(1, 2)
        h = F.relu(self.conv1(F.pad(h, (2, 0))))
        h = F.relu(self.conv2(F.pad(h, (4, 0))))
        mask = (x != PAD).float().unsqueeze(1)
        return (h * mask).sum(2) / mask.sum(2).clamp(min=1.0)


class _Expert(nn.Module):
    """One skill's correction MLP. Zero-init residual => identity at init.

    Input is [h1; slot memory; raw-char features], so the expert can use both
    L1's knowledge and char detail L1 never learned (e.g. digits)."""

    def __init__(self, in_dim, mem_dim, hidden, out_dim, text_dim=128):
        super().__init__()
        self.text = _SkillTextEncoder(hidden=text_dim)
        self.fc1 = nn.Linear(in_dim + mem_dim + text_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.res = nn.Linear(out_dim, in_dim)
        nn.init.zeros_(self.res.weight)
        nn.init.zeros_(self.res.bias)

    def forward(self, h, m, x):
        # h: (B, in_dim) L1 rep; m: (B, mem_dim) slot memory; x: (B, L) chars
        z = torch.cat([h, m, self.text(x)], dim=1)
        z = F.relu(self.fc1(z))
        return self.res(F.relu(self.fc2(z)))  # residual (B, in_dim)


class L2SkillsParser(nn.Module):
    """Skills parser: per-skill experts + multi-slot memory + domain router.

    A small router over the raw chars (independent of L1) picks one of K
    experts. Expert k is conditioned on memory slot k and its own raw-char
    encoder (L1 never saw digits/words it was not trained on). Because each
    skill owns its expert + memory slot + text encoder, training skill j never
    overwrites skill k (k != j) — skills coexist. All residuals are zero-init,
    so at init L2 is exactly the identity: h1 + 0.
    """

    def __init__(self, in_dim=512, mem_dim=128, hidden=192, out_dim=512,
                 n_slots=2, text_dim=128):
        super().__init__()
        self.n_slots = n_slots
        self.memory = nn.Parameter(torch.zeros(n_slots, mem_dim))
        self.experts = nn.ModuleList(
            [_Expert(in_dim, mem_dim, hidden, out_dim, text_dim=text_dim)
             for _ in range(n_slots)])
        # domain router over raw chars (robust: does not depend on L1's h1).
        # A char histogram + one linear: 2 kernels at batch 1 (vs ~6 for an
        # embedding + masked pool), and the histogram is a fixed, cheap feature
        # that already separates the domains (logic/arithmetic vs text).
        self.router_fc = nn.Linear(len(VOCAB), n_slots)

    def _route(self, x):
        # x: (B, L) char ids -> (B, K) router logits
        hist = F.one_hot(x, len(VOCAB)).sum(1).float()  # (B, |V|)
        return self.router_fc(hist)

    def _residual(self, h1, x, slot):
        # slot: (B,) long expert index; returns h1 + routed residual
        r = torch.zeros_like(h1)
        for k in range(self.n_slots):
            m = (slot == k)
            if m.any():
                r[m] = self.experts[k](
                    h1[m], self.memory[k:k + 1].expand(m.sum(), -1), x[m])
        return h1 + r

    def forward(self, h1, x):
        """Inference: router argmax picks one expert per row."""
        slot = self._route(x).argmax(1)
        if slot.numel() == 1:  # batch-1 fast path (the serving case)
            k = slot.item()
            return h1 + self.experts[k](
                h1, self.memory[k:k + 1].expand(h1.size(0), -1), x)
        return self._residual(h1, x, slot)

    def forward_train(self, h1, x, domain):
        """Training: teacher-forced expert selection by domain.

        domain: (B,) long, the true slot per row. Each expert sees only its
        own domain's rows, so an expert's weights are updated only by its own
        skill. Returns (h1 + residual, router logits) — the router logits are
        for the domain-classifier CE loss (trained separately)."""
        g = self._route(x)
        return self._residual(h1, x, domain), g


class Classifier(nn.Module):
    def __init__(self, n_classes=3, n_slots=2):
        super().__init__()
        self.l1 = L1ParsingModel()
        self.l2 = L2SkillsParser(n_slots=n_slots)
        self.head = nn.Linear(512, n_classes)

    def forward(self, x, use_l2=True):
        h = self.l1(x)
        h = self.l2(h, x) if use_l2 else h
        return self.head(h)

    def forward_train(self, x, domain, use_l2=True):
        """Training forward: teacher-forced L2 routing.

        Returns (task_logits, router_logits); router_logits is None when
        use_l2=False."""
        h = self.l1(x)
        if use_l2:
            h, g = self.l2.forward_train(h, x, domain)
        else:
            g = None
        return self.head(h), g

    @torch.no_grad()
    def predict(self, text: str, use_l2=True) -> int:
        x = torch.tensor([encode(text)], dtype=torch.long,
                          device=self.l1.emb.weight.device)
        return int(self(x, use_l2=use_l2).argmax(1).item())

    @torch.no_grad()
    def predict_rich(self, text: str) -> dict:
        """Full prediction: final label + probabilities + L1-vs-L2 breakdown.

        Runs L1 once: the base (L1) decision is head(l1), the final (L1+L2)
        decision is head(l2(l1, x)). l2_changed says whether the skill layer
        flipped the call."""
        x = torch.tensor([encode(text)], dtype=torch.long,
                          device=self.l1.emb.weight.device)
        h1 = self.l1(x)
        full = self.head(self.l2(h1, x))[0]
        l1 = self.head(h1)[0]
        p_full = F.softmax(full, dim=0)
        p_l1 = F.softmax(l1, dim=0)
        label, l1_label = int(full.argmax()), int(l1.argmax())
        c = full.numel()
        top2 = torch.topk(p_full, min(2, c)).values
        margin = float(top2[0] - top2[1]) if c >= 2 else float(top2[0])
        return {
            "text": text,
            "label": label,
            "name": _label_name(label),
            "confidence": round(float(p_full.max()), 4),
            "margin": round(margin, 4),
            "probabilities": {_label_name(i): round(float(p_full[i]), 4)
                              for i in range(c)},
            "l1": {"label": l1_label, "name": _label_name(l1_label),
                   "confidence": round(float(p_l1.max()), 4)},
            "l2_changed": label != l1_label,
        }