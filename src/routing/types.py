"""
Semantic type extraction -- the L2 rung of the conditioning ladder.

THE PROBLEM YOU SPOTTED
-----------------------
KG-LLM's four files contain NO explicit type column:

    entity2text.txt      entity_id \t surface text
    relation2text.txt    relation_id \t surface text
    train.tsv            head \t relation \t tail
    test.tsv             head \t relation \t tail \t label

So a naive reading says the L2 rung is impossible on this data. It is not --
the type signal is present, just encoded differently in each dataset.

FOUR RECOVERY ROUTES
--------------------
  1. POS suffix          WN18RR entities are `stool_NN_2` -> NN/VB/JJ/RB.
                         These are EXACTLY Knit's four categories
                         ("NN denotes nouns, VB denotes verbs, RB adverbs, JJ adjectives"),
                         so WN18RR reproduces Knit's conditioning for free.
  2. Relation-path types FB15k-237 relations are `/people/person/nationality`
                         -> domain `person`, range `location`. The Freebase schema
                         is embedded in the relation NAME.
  3. Induced types       Cluster entities by the relations they participate in.
                         Dataset-agnostic; works on YAGO3-10 where names are bare
                         surface strings. Reuses build_relation_type_index().
  4. LLM-labelled        Ask an LLM for each entity's type. General but costs
                         |E| calls -- exactly the cost the router exists to avoid,
                         so it is the fallback, not the default.

Route 3 is the default because it needs nothing beyond train.tsv and therefore
works on every dataset identically -- which keeps the ladder comparable across
Chapter 2's |E| sweep.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..data.loaders import KG

# ----------------------------------------------------------------- route 1
_POS = re.compile(r"_(NN|VB|JJ|RB)_?\d*$")


def pos_type(entity_id: str, surface: str | None = None) -> str | None:
    """WordNet POS suffix: `stool_NN_2` -> 'NN'. Knit's four categories."""
    for s in (entity_id, surface or ""):
        m = _POS.search(s)
        if m:
            return m.group(1)
    return None


# ----------------------------------------------------------------- route 2
def freebase_relation_types(relation_id: str) -> tuple[str | None, str | None]:
    """
    `/people/person/nationality` -> ('person', 'nationality')
    `/film/film/genre`           -> ('film', 'genre')

    The middle segment is the DOMAIN type; the last is the property. Range type
    is not always recoverable from the name, so induced types (route 3) complete it.
    """
    if not relation_id.startswith("/"):
        return None, None
    parts = [p for p in relation_id.split("/") if p]
    if len(parts) >= 3:
        return parts[1], parts[-1]
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


# ----------------------------------------------------------------- route 3
def induced_types(kg: KG, min_support: int = 5, max_types: int = 64) -> dict[str, str]:
    """
    Type = the entity's most distinctive relation signature.

    An entity that appears as head of `/people/person/nationality` is a person;
    one that appears as tail is a location. We do not need a type ontology -- the
    relation an entity participates in, and the SIDE it appears on, is the signal.

    Dataset-agnostic: needs only train.tsv.
    """
    sig: dict[str, Counter] = defaultdict(Counter)
    for t in kg.train:
        sig[t.head][f"{t.relation}::head"] += 1
        sig[t.tail][f"{t.relation}::tail"] += 1

    # global frequency -> keep only the most common signatures as type labels
    global_counts: Counter = Counter()
    for c in sig.values():
        if c:
            global_counts[c.most_common(1)[0][0]] += 1
    vocab = {k for k, v in global_counts.most_common(max_types) if v >= min_support}

    out: dict[str, str] = {}
    for e, c in sig.items():
        for cand, _ in c.most_common():
            if cand in vocab:
                out[e] = cand
                break
        else:
            out[e] = "OTHER"
    return out


# ----------------------------------------------------------------- dispatcher
def entity_types(kg: KG, method: str = "auto") -> dict[str, str]:
    """
    method: auto | pos | freebase | induced

    'auto' picks per dataset:
        WN*  -> pos       (Knit's four POS categories, present in the entity ids)
        FB*  -> freebase  (schema embedded in relation names)
        else -> induced   (YAGO3-10 and anything else)
    """
    if method == "auto":
        n = kg.name.upper()
        method = "pos" if n.startswith("WN") else "freebase" if n.startswith("FB") else "induced"
        print(f"[types] dataset {kg.name} -> method '{method}'")

    if method == "pos":
        out = {e: (pos_type(e, kg.ent2txt.get(e)) or "OTHER") for e in kg.ent2txt}
    elif method == "freebase":
        dom: dict[str, Counter] = defaultdict(Counter)
        for t in kg.train:
            d, _ = freebase_relation_types(t.relation)
            if d:
                dom[t.head][d] += 1
        out = {e: (dom[e].most_common(1)[0][0] if dom.get(e) else "OTHER")
               for e in kg.ent2txt}
    elif method == "induced":
        out = induced_types(kg)
        out = {e: out.get(e, "OTHER") for e in kg.ent2txt}
    else:
        raise ValueError(f"unknown method: {method}")

    dist = Counter(out.values())
    print(f"[types] {len(dist)} distinct types | "
          f"OTHER = {dist.get('OTHER', 0)/max(len(out),1):.1%} | "
          f"top: {dist.most_common(5)}")
    return out


def type_coverage_report(types: dict[str, str]) -> dict:
    """
    Is the L2 rung usable on this dataset?

    If OTHER dominates, semantic-type conditioning has nothing to condition on --
    which is itself a finding, and exactly the label-opacity boundary that
    UKGEBN (opaque protein IDs), GS-KGC (`stool_NN_2`) and MKGL ("14 entities
    named 'call'") each hit from a different direction.
    """
    dist = Counter(types.values())
    n = len(types) or 1
    other = dist.get("OTHER", 0) / n
    return {
        "n_entities": n,
        "n_types": len(dist),
        "other_rate": other,
        "largest_type_share": max(dist.values()) / n if dist else 0.0,
        "distribution": dict(dist.most_common(20)),
        "usable_for_L2": other < 0.5 and len(dist) >= 3,
        "note": ("L2 conditioning viable" if other < 0.5
                 else "OTHER dominates -- L2 has little to condition on; report this"),
    }
