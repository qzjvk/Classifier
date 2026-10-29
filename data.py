"""Deterministic training data: propositional logic + text sentiment skill.

L1 task: classify a propositional formula as tautology / contradiction /
contingent (ground truth by exhaustive truth table).
Skill task: sentiment of a short English sentence -> POS/NEG/NEU (ground
truth by construction: templates over disjoint per-class lexicons).
gen_arithmetic is the stand-in skill (labels 3..4, or shifted by offset to
higher slots), kept so the pipeline stays verifiable without ./dataset/.
No LLM needed: ground truth is computed exactly.
"""

import json
import os
import random
import numpy as np

TAUT, CONTRAD, CONTINGENT = 0, 1, 2
TRUE, FALSE = 3, 4
POS, NEG, NEU = 3, 4, 5  # text skill labels; same slots as the stand-in

BINARY = {"&": lambda a, b: a & b, "|": lambda a, b: a | b,
          "^": lambda a, b: a ^ b, "->": lambda a, b: ~a | b}
BINARY_STR = {"&": " & ", "|": " | ", "^": " ^ ", "->": " -> "}


def _truth_table(expr, k):
    """Evaluate expression tree over all 2**k assignments -> bool array."""
    n = 1 << k
    if isinstance(expr, int):  # variable index
        bit = 1 << expr
        return np.array([(m & bit) != 0 for m in range(n)])
    if isinstance(expr, tuple):
        op, children = expr
        if op == "~":
            return ~_truth_table(children[0], k)
        vals = [_truth_table(c, k) for c in children]
        return BINARY[op](*vals)
    raise TypeError(expr)


def _render(expr, k):
    if isinstance(expr, int):
        return chr(ord("p") + expr)
    op, children = expr
    if op == "~":
        s = _render(children[0], k)
        return "~" + (f"({s})" if isinstance(children[0], tuple) else s)
    parts = [_render(c, k) for c in children]
    if len(parts) == 2:
        a, b = parts
        if isinstance(children[0], tuple) and children[0][0] in BINARY:
            a = f"({a})"
        if isinstance(children[1], tuple) and children[1][0] in BINARY:
            b = f"({b})"
        return a + BINARY_STR[op] + b
    raise TypeError


def _random_tree(rng, k, depth):
    if depth == 0 or rng.random() < 0.35:
        return rng.randrange(k)
    if rng.random() < 0.2:
        return ("~", [_random_tree(rng, k, depth - 1)])
    op = rng.choice(list(BINARY))
    return (op, [_random_tree(rng, k, depth - 1), _random_tree(rng, k, depth - 1)])


def _tautology(rng, k):
    v = [rng.randrange(k) for _ in range(min(3, k))]
    a, b = v[0], v[1 % k]
    templates = [
        lambda: ("|", [a, ("~", [a])]),
        lambda: ("|", [("->", [a, b]), ("->", [b, a])]),
        lambda: ("->", [("&", [a, ("~", [a])]), b]),
        lambda: ("->", [a, ("|", [b, ("~", [b])])]),
        lambda: ("->", [("&", [a, b]), a]),
    ]
    return templates[rng.randrange(len(templates))]()


def _contradiction(rng, k):
    v = [rng.randrange(k) for _ in range(min(3, k))]
    a, b = v[0], v[1 % k]
    templates = [
        lambda: ("&", [a, ("~", [a])]),
        lambda: ("~", [("|", [a, ("~", [a])])]),
        lambda: ("&", [("->", [a, b]), ("&", [a, ("~", [b])])]),
        lambda: ("&", [("&", [a, ("~", [a])]), ("&", [b, ("~", [b])])]),
        lambda: ("&", [("&", [("~", [("|", [a, b])]), a]), b]) if k >= 2 else ("&", [a, ("~", [a])]),
    ]
    return templates[rng.randrange(len(templates))]()


def gen_logic(n, rng, maxlen=96):
    """Return (formula, label) list. ~50% random, 25% taut, 25% contrad."""
    out = []
    while len(out) < n:
        k = rng.randrange(2, 7)  # 2..6 variables
        r = rng.random()
        if r < 0.5:
            expr = _random_tree(rng, k, rng.randrange(2, 6))
        elif r < 0.75:
            expr = _tautology(rng, k)
        else:
            expr = _contradiction(rng, k)
        s = _render(expr, k)
        if len(s) > maxlen:
            continue
        tt = _truth_table(expr, k)
        label = TAUT if tt.all() else CONTRAD if not tt.any() else CONTINGENT
        out.append((s, label))
    return out


def gen_arithmetic(n, rng, max_op=9, offset=0):
    """Return (expr, TRUE/FALSE + offset) list. Single-digit operands by
    default so the skill is within a small conv net's receptive field.
    offset shifts the labels (e.g. 3 places this skill at 6/7 after the
    text-sentiment skill at 3/4/5)."""
    out = []
    hi = max_op + 1
    while len(out) < n:
        a = rng.randrange(1, hi)
        b = rng.randrange(1, hi)
        op = rng.choice(["+", "-", "*", "/"])
        if op == "/":
            if a % b != 0:
                continue
            r = a // b
        elif op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        else:
            r = a * b
        rel = rng.choice(["=", "<", ">"])
        if rng.random() < 0.5:
            if rel == "=":
                c = r if rng.random() < 0.5 else r + rng.choice([-2, -1, 1, 2])
            else:
                c = r + rng.choice([-3, -2, -1, 0, 1, 2, 3])
        else:
            c = rng.randrange(0, hi * 2)
        s = f"{a}{op}{b}{rel}{c}"
        ok = (c == r) if rel == "=" else ((r < c) if rel == "<" else (r > c))
        out.append((s, (TRUE if ok else FALSE) + offset))
    return out


# --- text sentiment skill --------------------------------------------------
# Lexicons are disjoint per class, so the label is exact by construction.
# Lowercase only; every char stays in classifier.VOCAB.
_NOUNS = ["movie", "book", "food", "song", "day", "trip", "game",
          "meal", "show", "film", "place", "weather", "service",
          "price", "holiday", "dinner", "concert", "experience"]
_POS_ADJ = ["great", "good", "wonderful", "excellent", "amazing",
            "fantastic", "lovely", "perfect", "nice", "superb"]
_NEG_ADJ = ["bad", "terrible", "awful", "horrible", "poor",
            "dreadful", "disgusting", "lousy", "dismal", "wretched"]
_NEU_ADJ = ["okay", "fine", "average", "normal", "plain",
            "regular", "usual", "simple", "common", "mediocre"]
_POS_VERB = ["loved", "enjoyed", "liked", "admired", "appreciated"]
_NEG_VERB = ["hated", "disliked", "despised", "avoided"]
_NEU_VERB = ["saw", "watched", "tried", "visited", "sampled"]
_DAYS = ["monday", "tuesday", "wednesday", "thursday",
         "friday", "saturday", "sunday"]


def gen_text(n, rng, maxlen=96):
    """Return (sentence, label) list: POS/NEG/NEU sentiment.

    Templates over disjoint lexicons; a minority of NEG lines negate a
    positive word ("was not good") so the task goes beyond bag-of-words."""
    out = []
    while len(out) < n:
        noun = rng.choice(_NOUNS)
        r = rng.random()
        if r < 0.34:
            label = POS
            s = rng.choice([
                f"the {noun} was {rng.choice(_POS_ADJ)}",
                f"i {rng.choice(_POS_VERB)} the {noun}",
                f"what a {rng.choice(_POS_ADJ)} {noun}",
                f"the {noun} is {rng.choice(_POS_ADJ)}",
                f"the {noun} was {rng.choice(_POS_ADJ)} and {rng.choice(_POS_ADJ)}",
            ])
        elif r < 0.67:
            label = NEG
            s = rng.choice([
                f"the {noun} was {rng.choice(_NEG_ADJ)}",
                f"i {rng.choice(_NEG_VERB)} the {noun}",
                f"what a {rng.choice(_NEG_ADJ)} {noun}",
                f"the {noun} is {rng.choice(_NEG_ADJ)}",
                f"the {noun} was {rng.choice(_NEG_ADJ)} and {rng.choice(_NEG_ADJ)}",
                f"the {noun} was not {rng.choice(_POS_ADJ)}",
                f"i did not {rng.choice(_POS_VERB)} the {noun}",
            ])
        else:
            label = NEU
            s = rng.choice([
                f"the {noun} was {rng.choice(_NEU_ADJ)}",
                f"i {rng.choice(_NEU_VERB)} the {noun}",
                f"the {noun} is {rng.choice(_NEU_ADJ)}",
                f"the {noun} is {rng.randrange(5, 30)} dollars",
                f"the {noun} was {rng.randrange(2, 50)} miles away",
                f"i {rng.choice(_NEU_VERB)} the {noun} on {rng.choice(_DAYS)}",
            ])
        if rng.random() < 0.5:
            s += "."
        if len(s) <= maxlen:
            out.append((s, label))
    return out


def write_dataset(out_dir, seed=0):
    """Write the JSONL files train.py auto-detects: logic base task
    (labels 0..2), text sentiment skill (labels 3..5), and an arithmetic
    skill (labels 6..7). Multiple skill files (skill.jsonl, skill_2.jsonl,
    ...) let several skills coexist; each gets its own L2 expert + memory
    slot."""
    os.makedirs(out_dir, exist_ok=True)
    splits = {
        "train.jsonl": gen_logic(300_000, random.Random(seed)),
        "val.jsonl": gen_logic(20_000, random.Random(seed + 1)),
        "skill.jsonl": gen_text(60_000, random.Random(seed + 2)),
        "skill_val.jsonl": gen_text(10_000, random.Random(seed + 3)),
        "skill_2.jsonl": gen_arithmetic(60_000, random.Random(seed + 4), offset=3),
        "skill_val_2.jsonl": gen_arithmetic(10_000, random.Random(seed + 5), offset=3),
    }
    for name, pairs in splits.items():
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            for s, l in pairs:
                f.write(json.dumps({"text": s, "label": l}) + "\n")
        print(f"wrote {path} ({len(pairs)} lines)")


def make_splits(seed=0):
    train = gen_logic(300_000, random.Random(seed))
    val = gen_logic(20_000, random.Random(seed + 1))
    skill_train = gen_arithmetic(60_000, random.Random(seed + 2))
    skill_val = gen_arithmetic(10_000, random.Random(seed + 3))
    return train, val, skill_train, skill_val


if __name__ == "__main__":
    import sys
    from collections import Counter
    if "--write" in sys.argv:
        write_dataset(os.path.join(os.path.dirname(__file__), "dataset"))
    train, val, st, sv = make_splits()
    print("logic train:", len(train), "val:", len(val))
    print("skill train:", len(st), "val:", len(sv))
    print("train labels:", Counter(l for _, l in train))
    print("val labels:", Counter(l for _, l in val))
    print("skill val labels:", Counter(l for _, l in sv))
    for s, l in train[:5]:
        print(f"{l}  {s}")