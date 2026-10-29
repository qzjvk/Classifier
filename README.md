# Classifier

Two-layer NN classifier. Not an LLM: no transformer, no attention, no
BERT-style encoder. Plain feedforward net with a classification head.
Output is a label + a small per-input breakdown, not a wall of text.
Trained and served on a GTX 1070.

**Current state:** base task = propositional logic (TAUT/CONTRAD/CONTINGENT),
skill 1 = English text sentiment (POS/NEG/NEU), skill 2 = arithmetic
(TRUE/FALSE), trained from `./dataset/` — logic 97.6% val, text 100% val,
arithmetic 99.4% val. Every prediction returns a rich payload
(probabilities + L1-vs-L2 breakdown) via `serve.py` / `inference.py`.
Skills coexist: each owns a memory slot + expert + raw-char encoder, so
training skill j never overwrites skill k — see "Extending the model"
below.

## File map

```
Classifier/
├── classifier.py    # model: VOCAB, encode(), L1, _SkillTextEncoder, L2 (router+experts), Classifier
├── data.py          # generators (logic, text sentiment, arithmetic) + ./dataset/ writer
├── train.py         # training pipeline (L1 -> skills w/ replay -> SVD -> bench)
├── inference.py     # CLI: text -> label + breakdown (--json for full payload)
├── serve.py         # web demo server (stdlib http, port 8377)
├── index.html       # web demo page
├── requirements.txt # torch 2.6.0 (cu124 build, has sm_61 kernels) + numpy
├── classifier.pt    # trained weights (full state dict, 10.79MB)
├── dataset/         # logic base + text + arithmetic skills (regenerate: data.py --write)
├── _verify.py       # diagnostic: accuracy, router, latency, per-op profile on the saved model
├── _bench.py        # diagnostic: L1-only vs L1+L2 latency, CUDA profile
├── _smoke.py        # diagnostic: model smoke test (shapes, grads, identity at init)
├── _diag.py         # diagnostic: per-domain error analysis (confusion, length buckets, samples)
├── _exp_l1.py       # ablation: L1 variants (2-conv RF-5 vs dilated TCN) on the logic task
└── .venv/           # NOT committed: python 3.13 + torch 2.6.0+cu124 (has sm_61 kernels)
```

Everything runs from `.venv/bin/python` — the system python3 is the vllm
venv whose torch (2.13+cu130) has **no Pascal kernels** and cannot use the
1070. Do not install a different torch into `.venv` without checking sm_61.

## Architecture

```
input (text, <=96 chars)
    |
    v
L1  parsing model (trained on logic)
     char embedding (55 x 256, pad id 0)
     4x causal conv1d 256 -> 256, k=3, dilation 1, 1, 2, 4
        (left-only pad, unidirectional; receptive field: 17 chars)
     masked mean pool -> 256-d
     Linear 256 -> 512 -> rep h1
    |
    v
L2  skills parser (per-skill experts + multi-slot memory + domain router)
     router: char histogram (55) -> Linear(55 -> K)   (K = n_slots)
             argmax picks one expert per row (batch-1 fast path: 1 kernel)
     expert k: own tiny raw-char encoder:
               char embedding (55 x 64)
               causal conv1d 64 -> 128, k=3
               causal conv1d 128 -> 128, k=3 dilation 2
               masked mean pool -> 128-d text features (RF 7 chars)
               MLP concat [h1; memory_slot_k; text] (512 + 128 + 128)
               768 -> 192 -> 512 (ReLU)
               zero-init residual Linear 512 -> 512, added to h1
    |
    v
head  Linear(512 -> C)  -> logits -> label
```

- **L1** (`L1ParsingModel` in `classifier.py`) is the parsing model, trained
  on logic. Causal convs only: no attention, no bidirectional pass. The
  4-layer dilated stack (dilations 1, 1, 2, 4) gives position *t* a
  receptive field of 17 chars — long enough to track deeply nested
  formulas, which are the hard part of the logic task. Pooled to 256-d,
  then projected to the fixed 512-d h1.
- **L2** (`L2SkillsParser`) is the skills parser. A small **domain router**
  over the raw chars (char histogram + one linear, independent of L1's h1)
  picks one of K experts (K = n_slots). Expert k is an MLP (768 -> 192 ->
  512) conditioned on memory slot k **and its own raw-char encoder**,
  producing a zero-init residual added to h1. All residuals are zero-init,
  so at init L2 is exactly the identity: `h1 + 0` — L1's knowledge is
  retained by construction.
- **Per-expert raw-char encoder** (`_SkillTextEncoder`): L1 is a *logic*
  parser, frozen during skill training — its h1 never learned what digits or
  sentiment words mean. Each expert owns a tiny char conv stack (RF 7) so a
  skill can be learned from h1 + text instead of asking an adapter to
  recover char detail from h1. This is what carries the skills: L1 alone
  gets 0.0% on text and arithmetic (its h1 + frozen base rows know no
  digits or sentiment words), L1+L2 gets 100% and 99.4%.
- **Multi-slot memory**: `l2.memory` is (K x 128) — one 128-d slot per
  expert. Each skill owns its slot + expert + text encoder, so training
  skill j never overwrites skill k (k != j): skills coexist. The base task
  (logic) is slot 0; its expert is identity at init and the replay keeps it
  compensating for the shared head.
- **Head**: one linear layer, C classes (base + all skills). C is derived
  from the data at training time and read back from `classifier.pt` at
  inference time (`inference.py` / `serve.py` rebuild the head from the
  state dict). During skill training the head's base-task rows are frozen
  so skill gradients can't pull them off h1 (see "Training pipeline").
- **Size**: 2.70M params, 10.79MB fp32. Compact by design (12MB budget).

### Inference path

`Classifier.forward(x, use_l2=True)`: `x` is a (B, 96) int64 tensor of char
ids (`encode()`). `use_l2=False` bypasses L2 (router + experts) and returns
L1 + full-head output — a diagnostic for what L1 knows without the skill
experts (note: with the multi-class head, skill rows can fire on base-task
inputs and drag this below the 3-class phase-1 number; with the current
weights they rarely do).
`predict(text)` is the single-string argmax wrapper. `predict_rich(text)`
runs the forward twice (L1-only and L1+L2, sharing L1's h1) and returns the
full payload: final label, confidence, top-2 margin, per-class
probabilities, the L1-alone decision, and `l2_changed` (whether the skill
layer flipped the call). This is what `serve.py` and `inference.py --json`
expose.

## Data and labels

`encode()` maps text through `VOCAB` (55 chars: logic + arithmetic symbols,
digits, full lowercase alphabet a–z, light punctuation `. ? '`; `" "` is the
pad id). Constraints:

- **Unknown char** → `KeyError` in `encode()`. If real data uses new chars,
  extend `VOCAB` in `classifier.py` first (embedding is indexed by position).
- **MAXLEN = 96**: longer inputs are silently truncated. Change the constant
  in `classifier.py` if real inputs are longer (sequence length is fixed).
- **Label layout** (enforced by `main()` in `train.py`): base-task labels
  must be `0..n_logic-1`; each skill's labels are the next contiguous block.
  `n_logic = max(train labels)+1`. The dataset uses TAUT=0, CONTRAD=1,
  CONTINGENT=2 (logic base), POS=3, NEG=4, NEU=5 (text skill), TRUE=6,
  FALSE=7 (arithmetic skill). `LABELS` in `classifier.py` is display-only
  (name per index).

## Real dataset

`./dataset/` holds the logic base task + two skills (one JSON object per
line):

```
dataset/train.jsonl       # base task: logic, labels 0..2 (300k lines)
dataset/val.jsonl         # logic val (20k)
dataset/skill.jsonl       # text sentiment, labels 3..5 (60k lines)
dataset/skill_val.jsonl   # text sentiment val (10k)
dataset/skill_2.jsonl     # arithmetic, labels 6..7 (60k lines)
dataset/skill_val_2.jsonl # arithmetic val (10k)
```

Generated deterministically by `data.py` (seed 0). The text skill uses
templates over disjoint per-class lexicons (~28% of NEG lines negate a
positive word, e.g. "the trip was not good", so the task goes beyond
bag-of-words); the arithmetic skill is exact `a op b ? c` expressions.
Regenerate with `.venv/bin/python data.py --write`.
`load_dataset()` in `train.py` auto-detects the base-task files + all
`skill_*.jsonl` pairs (all must exist, otherwise the synthetic stand-in is
generated). To use your own data, keep the same format and label layout;
class counts are derived automatically, no code changes needed for a
different number of classes or skills.

## Training pipeline

`train.py`, in order (timings on the 1070):

| phase | what | frozen | hyperparams | time |
|---|---|---|---|---|
| 1. L1 | L1 + head on base task (CE) | L2 | 4 ep, Adam 1e-3, batch 512 | ~4 min |
| 2. Skill k (each) | L2 + head on skill k **with 50/50 replay of base task + prior skills** (CE); head expands n→n+n_skill (old rows copied, new rows ~0-init); **head's base-task rows frozen**; router trained by domain CE (weight 0.1) | L1 | 20 ep, Adam 1e-3, batch 512 | ~4 min/skill |
| 3. SVD | low-rank pass over L1 convs | — | budget 12MB (no-op if under) | instant |
| 4. Bench + save | latency (batch 1) + `torch.save` state dict | — | — | ~10 s |

Total ~13 min for 2 skills (last full run on this box).
Deterministic: `torch.manual_seed(0)` + seeded data (`data.py --write` for
`./dataset/`, or `make_splits(seed=0)` for the stand-in).

Key decisions:

- **Replay is required in every skill phase.** The head and L2 are shared;
  skill-only gradients destroy the base task (logic val collapses to 0%
  without replay). 50/50 replay of the base task + prior skills preserves
  them.
- **Per-skill experts + multi-slot memory + per-expert text encoder.** Each
  skill owns its expert + memory slot + raw-char encoder, so training skill
  j never overwrites skill k (k != j). The domain router is teacher-forced
  during training (each expert sees only its own domain's rows) and uses
  argmax at inference.
- **Head's base-task rows are frozen in skill phases.** The softmax over all
  classes otherwise lets skill gradients pull the base rows off h1, and the
  base expert absorbs a large compensating residual (which breaks the
  L1-only path). Freezing keeps base rows tuned for h1 alone.
- **L1 is frozen in skill phases.** L1 stays the pure "logic parser"; all
  adaptation happens in L2 + head.
- **SVD is a no-op today**: the model is already under the 12MB budget.
  Note that the current implementation reconstructs a full matrix
  (`(U_r·S_r)·Vt_r`), which saves zero params — real compression needs
  factorized (unmaterialized) storage. Treat `svd_compress` as a stub until
  that's implemented.

## Run

```bash
# recreate the venv (.venv is not committed)
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt

# train (GTX 1070 = torch device index 2 on this box)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python train.py

# infer
CUDA_VISIBLE_DEVICES=2 .venv/bin/python inference.py "p | ~p" "the movie was great"
CUDA_VISIBLE_DEVICES=2 .venv/bin/python inference.py --json "the trip was not good"
echo "p & ~p" | CUDA_VISIBLE_DEVICES=2 .venv/bin/python inference.py -

# web demo (http://127.0.0.1:8377)
CUDA_VISIBLE_DEVICES=2 .venv/bin/python serve.py

# diagnostics on the saved model
CUDA_VISIBLE_DEVICES=2 .venv/bin/python _verify.py
```

`serve.py` is stdlib-only: loads `classifier.pt` once, `POST /predict` with
`{"text": ...}` → the `predict_rich` payload (`label`, `name`, `confidence`,
`margin`, `probabilities`, `l1`, `l2_changed`, `ms`), serves `index.html`.
The demo renders the label, a per-class probability bar, and the L1-vs-L2
breakdown. Inference is behind a lock (CUDA is not thread-safe); it runs the
forward twice per request (L1-only + L1+L2, sharing L1's h1).

## Environment gotchas

- **GPU index mismatch**: on this box the 1070 is `nvidia-smi` index **1**
  but torch index **2** (the two orderings differ). Always use
  `CUDA_VISIBLE_DEVICES=2`. The 3060s (torch 0/1) run the agent — don't touch.
- **Shared 1070**: other workloads (e.g. a game client) can run on the 1070
  alongside training/inference. Accuracy is unaffected, but batch-1 latency
  degrades ~3x (0.5ms → ~1.4ms) while a co-tenant is active — check
  `nvidia-smi` before trusting latency numbers.
- **OOM**: the 1070's pool fragments after L1 training; run with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to be safe.
- **Hardcoded 512**: the h1 dim is fixed at 512 — it's the default in
  `classifier.py` (L1 `out_dim`, L2 in/out, head input) and hardcoded in the
  head constructions in `train.py` (`nn.Linear(512, n)`). If you change L1's
  output width, update both. The expert MLP input is tied to it too:
  512 (h1) + 128 (memory) + 128 (text) = 768, via the `L2SkillsParser`
  defaults (`in_dim`, `mem_dim`, `text_dim`). L1's internal conv width
  (256) is free to change as long as `out_dim` stays 512.
- **No numpy-free assumption**: `.venv` has numpy (used by `data.py` truth
  tables and `to_tensor`).

## Results

(dataset in `./dataset/`: logic base task + text sentiment + arithmetic;
seed 0, current pipeline, last full run on this box)

| stage                        | logic val | text val | arithmetic val |
|------------------------------|-----------|----------|----------------|
| L1 only (after phase 1, 3-class head) | 97.37% | — | — |
| after skill 1 (20 ep, replay) | 97.52%   | 100%     | —              |
| after skill 2 (20 ep, replay) | 97.55%   | 100%     | 99.39%         |
| after SVD (skipped, in budget)| 97.55%   | 100%     | 99.39%         |

Router domain accuracy (val): 100% on all three domains (logic, text,
arithmetic).

L1-only diagnostic (L1 + full 8-class head, no skill experts — not the
serving path): logic 97.4%, text 0.0%, arithmetic 0.0%. With the current
weights the skill head rows rarely fire on base-task inputs, so this is
~equal to the 3-class phase-1 number.

Model size: **10.79MB** (2.70M params, fp32). Latency (batch 1, GTX 1070):
**~580µs/request** for one L1+L2 forward pass (~340µs L1-only), measured
while another workload shared the GPU — see the shared-GPU gotcha. The rich
payload runs two passes (L1-only + L1+L2, sharing L1's h1), so ~1ms plus
HTTP overhead through the web server.

Notes:
- **Skills coexist**: skill 1 (text) is 100% after skill 2 (arithmetic)
  training — not overwritten. Each skill owns its expert + memory slot +
  text encoder, so training skill j never touches skill k.
- **Base task intact**: logic is 97.55% after both skills (50/50 replay
  holds it; without replay, logic val collapses to 0%).
- **L2 retains what L1 doesn't know**: text 100% vs 0.0% for L1 alone;
  arithmetic 99.4% vs 0.0% for L1 alone. The per-expert raw-char encoder
  gives skills direct access to digits and words the frozen logic parser
  never learned.
- **Ground truth is computed exactly** (truth tables / templates by
  construction), no LLM in the loop.

## Extending the model

### Add a new skill (task)

1. Put skill data in `dataset/skill_k.jsonl` + `dataset/skill_val_k.jsonl`
   with labels starting right after the previous skill's labels (base is
   0..2, skill 1 is 3..5, skill 2 is 6..7, ...). Or extend `data.py` for a
   synthetic one and re-run `.venv/bin/python data.py --write`.
   **All** `dataset/*.jsonl` files (base task + every `skill_*.jsonl` pair)
   must exist, else `load_dataset()` silently falls back to the synthetic
   stand-in — regenerate the base-task files too if you changed the base
   task.
2. If the new text uses characters outside `VOCAB`, extend `VOCAB` in
   `classifier.py` first (the embedding is indexed by position; a new char
   changes the embedding shape, so a full retrain is required).
3. Re-run the full pipeline (`train.py`): phase 1 retrains L1, then each
   skill is trained with 50/50 replay of the base task + prior skills. The
   head expands, the router gains a slot, and the new skill gets its own
   expert + memory slot + text encoder automatically.
4. Verify: `inference.py` on a few examples (watch `l2_changed` and the
   confidence), `_verify.py` for accuracy/router/latency, and the web demo.
   Confirm the base-task val did not drop — replay should hold it near the
   pre-skill number.

Gotchas:
- **Skills coexist by design**: each skill owns its expert + memory slot +
  text encoder, so training skill j never overwrites skill k (k != j). The
  router is teacher-forced during training and uses argmax at inference; if
  two skills are very similar, the router may misroute (watch router
  accuracy in the training log).
- `LABELS` in `classifier.py` is display-only; add names for any new class
  indices or they render as `CLASS_n`.
- The head width is tied to the 512 hidden dim — see "Environment gotchas".