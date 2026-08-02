"""
Negative sampling.

KG-LLM's baseline (instructions_WN11.py, verbatim logic):

    rnd = random.random()
    if rnd <= 0.5:   corrupt head with random.choice(all_entities)
    else:            corrupt tail with random.choice(all_entities)

One negative per positive -> 10,000 triples become ~20,000 training instances.

Why this matters for Chapter 2
------------------------------
Uniformly random corruption produces mostly TYPE-VIOLATING negatives, which are
trivially separable. TCM uses the same scheme and the literature review flags it
as too easy; EGIT deliberately uses hard negatives instead.

So KG-LLM's own data already has a (chosen, rejected) pair structure -- DPO
conversion is nearly free -- and its negatives are the weakest possible baseline
for our hard-negative contribution to beat.
"""
from __future__ import annotations

import random
from collections import defaultdict

from .loaders import KG, Triple


# ------------------------------------------------------------------ baseline
def random_negative(t: Triple, kg: KG, rng: random.Random) -> Triple:
    """KG-LLM's scheme. 50/50 head or tail, uniform over ALL entities."""
    ents = kg.entities
    if rng.random() <= 0.5:
        h = t.head
        while h == t.head:
            h = rng.choice(ents)
        return Triple(h, t.relation, t.tail, -1)
    tl = t.tail
    while tl == t.tail:
        tl = rng.choice(ents)
    return Triple(t.head, t.relation, tl, -1)


# ------------------------------------------------------------------ hard: type
def build_relation_type_index(kg: KG) -> dict[str, tuple[set[str], set[str]]]:
    """
    Observed domain/range per relation, induced from the training graph.
    Also the basis for Chapter 4's TYPE-2 HALLUCINATION check: a predicted entity
    that is real but violates the relation's range.
    """
    dom: dict[str, set[str]] = defaultdict(set)
    rng_: dict[str, set[str]] = defaultdict(set)
    for t in kg.train:
        dom[t.relation].add(t.head)
        rng_[t.relation].add(t.tail)
    return {r: (dom[r], rng_[r]) for r in set(dom) | set(rng_)}


def type_consistent_negative(t: Triple, kg: KG, rng: random.Random,
                             type_index: dict[str, tuple[set[str], set[str]]]) -> Triple | None:
    """
    HARD negative: corrupt the tail with an entity that has been observed in this
    relation's range, so the negative is type-VALID but factually wrong.
    Exactly the "plausible but wrong" case that type-2 hallucination describes.
    """
    rng_set = type_index.get(t.relation, (set(), set()))[1] - {t.tail}
    if not rng_set:
        return None
    return Triple(t.head, t.relation, rng.choice(sorted(rng_set)), -1)


# ------------------------------------------------------------------ hard: KGE
def kge_near_miss_negative(t: Triple, candidates: dict[tuple[str, str], list[str]],
                           rng: random.Random) -> Triple | None:
    """
    HARD negative from a pretrained KGE's top-k (EGIT's recipe).
    `candidates` maps (head, relation) -> ranked entity ids, excluding the gold.
    Build it once with pykeen; see chapters/ch2_adaptation.
    """
    cand = candidates.get((t.head, t.relation))
    if not cand:
        return None
    top = [c for c in cand[:20] if c != t.tail]
    if not top:
        return None
    return Triple(t.head, t.relation, rng.choice(top), -1)


# ------------------------------------------------------------------ dispatcher
def make_negatives(triples: list[Triple], kg: KG, strategy: str = "random",
                   seed: int = 42, **kw) -> list[Triple]:
    """
    strategy: random | type_consistent | kge_near_miss
    Falls back to `random` whenever a hard strategy cannot produce a candidate,
    and reports the fallback rate -- which is itself worth stating in the thesis.
    """
    rng = random.Random(seed)
    type_index = build_relation_type_index(kg) if strategy == "type_consistent" else None
    candidates = kw.get("kge_candidates", {})

    out, fallbacks = [], 0
    for t in triples:
        neg = None
        if strategy == "type_consistent":
            neg = type_consistent_negative(t, kg, rng, type_index)
        elif strategy == "kge_near_miss":
            neg = kge_near_miss_negative(t, candidates, rng)
        if neg is None:
            if strategy != "random":
                fallbacks += 1
            neg = random_negative(t, kg, rng)
        out.append(neg)

    if fallbacks:
        print(f"[negatives] strategy={strategy}: {fallbacks}/{len(triples)} "
              f"({fallbacks/len(triples):.1%}) fell back to random")
    return out
