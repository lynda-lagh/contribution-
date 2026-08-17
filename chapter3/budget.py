"""
★★ THE CORE OF CHAPTER 3 — a context budget that is actually enforced.

WHY THIS FILE IS THE METHOD
---------------------------
The chapter's claim is *"at a FIXED context budget, does it matter where the
context goes?"*. That claim is only testable if the budget is real. The first
Chapter 3 run failed this: L0..L4 changed how much context was supplied AND how
specifically it was targeted, so no difference could be attributed to either.

    L0  17,866,755 tokens   L3  2,977,580 tokens   -> different quantity
    L0  no routing          L3  per-quality-band   -> different specificity
                                                      => uninterpretable

Here, every policy receives the SAME budget and differs only in priority order.
Any difference in MRR is specificity, and nothing else.

⚠️ THE BUDGET MUST BE COUNTED IN TOKENS, NOT BLOCKS.
   "two blocks each" is not a fixed budget: a typed-neighbour list is ~60 tokens
   and a type tag is ~8. Counting blocks would let a policy smuggle in 7x the
   context while appearing matched.

HOW ALLOCATION WORKS
--------------------
Greedy by priority, then a partial block if it still fits. Every policy is a
function `(block, context) -> float`; higher wins. Ties break deterministically
on `(kind, target)` so two runs with the same seed produce byte-identical
prompts -- otherwise the comparison is noise.

    blocks = candidate_blocks(...)          # everything that COULD be included
    kept   = allocate(blocks, budget=120, policy=POLICIES["S4_instance"])

★ `allocate` returns the kept blocks AND a per-decision reason, because the
  reason is what `faithfulness.py` later audits. An allocation that cannot say
  why it allocated is not explicable, and explicability is half the thesis title.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

# Kinds of context that can occupy budget. Ordered by typical size, smallest
# first -- used only for deterministic tie-breaking, never for priority.
#
# ★ `demonstrations` added after reading P30 (KICGPTv2), whose Knowledge Prompt
#   supplies other triples as in-context examples; CATS uses k=3 supporting
#   triples sharing the query relation, RealKGC shows triples sharing r_q.
#   It is by far the most expensive kind, and plausibly the most valuable for
#   exactly the long-tail elements that have nothing else -- which is what makes
#   the budget decision interesting rather than obvious.
KINDS = ("type_tag", "relation_description", "entity_description",
         "exclusions", "neighbours", "demonstrations")

# ⚠️ Truncating a NEIGHBOUR LIST loses the tail of a list; truncating a
#    DESCRIPTION can leave a fragment that is worse than nothing. Kinds listed
#    here are kept whole or dropped -- never cut. Set per-kind, not globally,
#    because a global `allow_partial=False` would starve policies of the ability
#    to use leftover budget at all.
ATOMIC_KINDS = frozenset({"entity_description", "relation_description", "type_tag"})


@dataclass
class Block:
    """One candidate piece of context, with its measured token cost."""
    kind: str
    target: str                 # entity id, relation id, or "h|r" for a query
    text: str
    tokens: int
    # features the policies key on; filled by sources.py
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown block kind {self.kind!r}; expected {KINDS}")
        if self.tokens < 0:
            raise ValueError("token cost cannot be negative")


@dataclass
class Allocation:
    """What was kept, what it cost, and WHY each decision was made."""
    kept: list[Block]
    budget: int
    spent: int
    reasons: dict[str, str]                 # "kind:target" -> stated reason
    dropped: list[Block] = field(default_factory=list)

    @property
    def utilisation(self) -> float:
        return self.spent / self.budget if self.budget else 0.0

    def text_blocks(self) -> list[str]:
        return [b.text for b in self.kept]

    def summary(self) -> dict:
        by_kind: dict[str, int] = {}
        for b in self.kept:
            by_kind[b.kind] = by_kind.get(b.kind, 0) + b.tokens
        return {
            "budget": self.budget,
            "spent": self.spent,
            "utilisation": self.utilisation,
            "n_kept": len(self.kept),
            "n_dropped": len(self.dropped),
            "tokens_by_kind": by_kind,
        }


def truncate_to(text: str, tokens: int, count: Callable[[str], int]) -> tuple[str, int]:
    """
    Cut `text` down to at most `tokens`, on a word boundary.

    Binary search on words rather than characters: cutting mid-word produces a
    fragment the tokeniser splits unpredictably, so the "budget" would be
    approximate exactly where it needs to be exact.
    """
    if tokens <= 0:
        return "", 0
    if count(text) <= tokens:
        return text, count(text)
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(" ".join(words[:mid])) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    out = " ".join(words[:lo])
    return out, count(out) if out else 0


def allocate(blocks: Iterable[Block], budget: int, policy,
             count: Callable[[str], int] | None = None,
             allow_partial: bool = True) -> Allocation:
    """
    Fill `budget` tokens with the highest-priority blocks.

    policy : object with .priority(block, ctx) -> float and .reason(block, ctx) -> str
             see policies.py
    count  : token counter. Defaults to whitespace words so tests run without a
             tokeniser; production passes the model's tokeniser.

    ⚠️ A block whose priority is None or -inf is NEVER included, whatever the
       budget. That is how a policy says "this element does not need this".
    """
    count = count or (lambda s: len(s.split()))
    blocks = list(blocks)
    ctx = {"blocks": blocks, "budget": budget}

    scored = []
    for b in blocks:
        p = policy.priority(b, ctx)
        if p is None or p == float("-inf"):
            continue
        scored.append((p, b))

    # deterministic: priority desc, then a stable key. Without the tiebreak two
    # identical-priority blocks could swap between runs and the comparison
    # between policies would include ordering noise.
    scored.sort(key=lambda pb: (-pb[0], KINDS.index(pb[1].kind), pb[1].target))

    kept: list[Block] = []
    dropped: list[Block] = []
    reasons: dict[str, str] = {}
    spent = 0

    for _, b in scored:
        room = budget - spent
        if room <= 0:
            dropped.append(b)
            continue
        if b.tokens <= room:
            kept.append(b)
            spent += b.tokens
            reasons[f"{b.kind}:{b.target}"] = policy.reason(b, ctx)
        elif allow_partial and b.kind not in ATOMIC_KINDS:
            text, n = truncate_to(b.text, room, count)
            if n > 0:
                kept.append(Block(b.kind, b.target, text, n, dict(b.meta, truncated=True)))
                spent += n
                reasons[f"{b.kind}:{b.target}"] = (
                    policy.reason(b, ctx) + f" (truncated to {n} tokens)")
            else:
                dropped.append(b)
        else:
            dropped.append(b)

    # blocks the policy refused outright still count as dropped, for reporting
    refused = [b for b in blocks if b not in kept and b not in dropped]
    dropped.extend(refused)

    return Allocation(kept=kept, budget=budget, spent=spent,
                      reasons=reasons, dropped=dropped)


def token_counter(model_name: str):
    """The real counter. Imported lazily so tests need no transformers."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
