"""
Subsampling.

Two independent axes, and they must not be confused:

  * TRIPLE budget  -- how many training triples (compute knob + Ch2's second axis)
  * ENTITY budget  -- |E|, Chapter 2's INDEPENDENT VARIABLE

Why stratify by relation
------------------------
Relation frequency in KGs is heavily Zipfian. A uniform random sample of 10k from
FB15k-237's 272k drops rare relations entirely -- and rare relations are exactly
where the interesting behaviour lives:

  * TSP: statistical rule mining discards rules for rare relations like `sisterOf`
    "even though these rules may hold substantial value".
  * Chapter 2's frequency-stratified analysis (EIR / PopBS) needs the tail to exist.

Why subsample entities WITHIN one dataset
-----------------------------------------
Sweeping |E| across FB15k-237 (14.5k) -> WN18RR (41k) -> YAGO3-10 (123k) confounds
|E| with relation count (237/11/37), density and label quality. Subsampling inside
YAGO3-10 holds everything else constant.
"""
from __future__ import annotations

import random
from collections import defaultdict

from .loaders import KG, Triple


def sample_triples(
    triples: list[Triple],
    n: int,
    seed: int = 42,
    stratified: bool = True,
    min_per_relation: int = 10,
) -> list[Triple]:
    """Sample `n` triples, guaranteeing `min_per_relation` per relation where possible."""
    rng = random.Random(seed)
    if n >= len(triples):
        return list(triples)

    if not stratified:
        return rng.sample(triples, n)

    by_rel: dict[str, list[Triple]] = defaultdict(list)
    for t in triples:
        by_rel[t.relation].append(t)

    selected: list[Triple] = []
    remaining: list[Triple] = []

    # pass 1 -- floor per relation
    for rel, ts in by_rel.items():
        rng.shuffle(ts)
        take = min(min_per_relation, len(ts))
        selected.extend(ts[:take])
        remaining.extend(ts[take:])

    if len(selected) > n:
        # budget too small for the floor -> warn and fall back proportionally
        print(f"[sampling] WARNING: {len(by_rel)} relations x min {min_per_relation} "
              f"= {len(selected)} > budget {n}. Floor reduced.")
        rng.shuffle(selected)
        return selected[:n]

    # pass 2 -- fill proportionally at random
    rng.shuffle(remaining)
    selected.extend(remaining[: n - len(selected)])
    rng.shuffle(selected)
    return selected


def entity_subset(kg: KG, n_entities: int, seed: int = 42) -> KG:
    """
    Chapter 2's independent variable: keep `n_entities` entities and the INDUCED
    subgraph (triples whose head AND tail both survive).
    """
    rng = random.Random(seed)
    ents = sorted(kg.ent2txt)
    if n_entities >= len(ents):
        return kg

    # degree-biased keep: preserves graph connectivity better than uniform choice,
    # which would leave the induced subgraph almost empty.
    deg: dict[str, int] = defaultdict(int)
    for t in kg.train:
        deg[t.head] += 1
        deg[t.tail] += 1
    ents.sort(key=lambda e: (-deg.get(e, 0), e))
    keep_core = ents[: int(n_entities * 0.7)]              # highest-degree core
    rest = ents[int(n_entities * 0.7):]
    rng.shuffle(rest)
    keep = set(keep_core) | set(rest[: n_entities - len(keep_core)])

    def induced(ts: list[Triple]) -> list[Triple]:
        return [t for t in ts if t.head in keep and t.tail in keep]

    sub = KG(
        name=f"{kg.name}-E{n_entities}",
        ent2txt={e: kg.ent2txt[e] for e in keep},
        rel2txt=dict(kg.rel2txt),
        train=induced(kg.train),
        test=induced(kg.test),
    )
    print(f"[sampling] |E| {len(kg.ent2txt)} -> {len(sub.ent2txt)} | "
          f"train {len(kg.train)} -> {len(sub.train)} | "
          f"test {len(kg.test)} -> {len(sub.test)}")
    return sub


def relation_frequency_report(triples: list[Triple]) -> dict:
    """For EIR / PopBS (Analyzing Bias) and the frequency-stratified analysis."""
    counts: dict[str, int] = defaultdict(int)
    for t in triples:
        counts[t.relation] += 1
    vals = sorted(counts.values(), reverse=True)
    total = sum(vals) or 1
    top10 = sum(vals[: max(1, len(vals) // 10)])
    return {
        "n_relations": len(counts),
        "n_triples": total,
        "max_freq": vals[0] if vals else 0,
        "min_freq": vals[-1] if vals else 0,
        "edge_imbalance_ratio": (vals[0] / vals[-1]) if vals and vals[-1] else None,
        "top10pct_share": top10 / total,
    }
