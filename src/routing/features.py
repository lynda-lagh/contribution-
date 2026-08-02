"""
Conditioning features -- the input to the router (Stage C, N11).

Everything here is computable from KG-LLM's four files. Nothing external is needed.

    L1  entity vs relation        element kind
    L2  semantic type             types.py (POS / Freebase schema / induced)
    L3  label quality             THIS module -- length, ambiguity, description
                                  present, degree
    L4  instance                  the raw feature vector itself

Each feature exists because a specific paper measured that it matters:

  description_present   ColKGC: rewriting descriptions that ALREADY EXIST gives
                        ~0 gain (FB15k-237 MRR 0.333 -> 0.332). "LLMs only modify
                        this description without actually introducing new information."
  label_informative     UKGEBN: opaque ids (`394.NGR_c11990`) -> "the advantages of
                        natural language processing technology disappear."
  label_ambiguity       MKGL: "there are 14 different entities named 'call'."
  degree                Analyzing Bias: "degree imbalance, popularity bias and
                        long-tail underrepresentation."
  pos/semantic type     Knit: "NN denotes nouns, VB denotes verbs, RB adverbs, JJ adjectives"
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from ..data.loaders import KG
from .types import entity_types

_WORDY = re.compile(r"[A-Za-z]{3,}")
_OPAQUE = re.compile(r"^[\W_]*(?:[A-Za-z]{0,3}[\W_]*\d|\d)")   # ids like 394.NGR_c11990, /m/0f8l9c


@dataclass
class ElementFeatures:
    element_id: str
    kind: str                  # "entity" | "relation"   -> L1
    semantic_type: str         #                            L2
    quality_band: str          # rich | moderate | poor  -> L3
    # raw features                                          L4
    label_len_chars: int
    label_len_words: int
    n_alpha_words: int
    looks_opaque: bool
    has_description: bool
    description_len: int
    ambiguity: int             # entities sharing this surface form
    degree: int
    degree_percentile: float

    def as_dict(self) -> dict:
        return asdict(self)


def _quality_band(*, looks_opaque: bool, n_alpha_words: int,
                  has_description: bool, ambiguity: int) -> str:
    """
    L3 bands. The thresholds are deliberately simple and reported, not tuned --
    the ladder measures WHERE conditioning stops paying, not how well a
    hand-tuned band performs.
    """
    if looks_opaque or n_alpha_words == 0:
        return "poor"                       # UKGEBN's PPI5k case
    if ambiguity > 3 and not has_description:
        return "poor"                       # MKGL's "14 entities named 'call'"
    if has_description and n_alpha_words >= 2 and ambiguity <= 3:
        return "rich"
    return "moderate"


def compute_features(kg: KG, type_method: str = "auto") -> dict[str, ElementFeatures]:
    """One ElementFeatures per entity and per relation."""
    types = entity_types(kg, method=type_method)

    deg: Counter = Counter()
    for t in kg.train:
        deg[t.head] += 1
        deg[t.tail] += 1
    degs = sorted(deg.values()) or [0]

    def pct(d: int) -> float:
        lo = sum(1 for x in degs if x < d)
        return lo / len(degs)

    surface_counts: Counter = Counter(v.strip().lower() for v in kg.ent2txt.values())

    out: dict[str, ElementFeatures] = {}

    for e, txt in kg.ent2txt.items():
        txt = txt or ""
        # KG-BERT files sometimes carry "name, gloss" -- a gloss counts as a description
        name = txt.split(",")[0].strip()
        has_desc = len(txt) > len(name) + 3
        words = _WORDY.findall(name)
        d = deg.get(e, 0)
        opaque = bool(_OPAQUE.search(name)) or not words
        amb = surface_counts.get(name.lower(), 1)
        out[e] = ElementFeatures(
            element_id=e, kind="entity", semantic_type=types.get(e, "OTHER"),
            quality_band=_quality_band(looks_opaque=opaque, n_alpha_words=len(words),
                                       has_description=has_desc, ambiguity=amb),
            label_len_chars=len(name), label_len_words=len(name.split()),
            n_alpha_words=len(words), looks_opaque=opaque,
            has_description=has_desc, description_len=max(0, len(txt) - len(name)),
            ambiguity=amb, degree=d, degree_percentile=pct(d),
        )

    rel_deg: Counter = Counter(t.relation for t in kg.train)
    rdegs = sorted(rel_deg.values()) or [0]
    for r, txt in kg.rel2txt.items():
        txt = txt or ""
        words = _WORDY.findall(txt)
        d = rel_deg.get(r, 0)
        lo = sum(1 for x in rdegs if x < d) / len(rdegs)
        out[r] = ElementFeatures(
            element_id=r, kind="relation", semantic_type="RELATION",
            # RelSemEnh / ColKGC: relation DESCRIPTIONS are "commonly missing",
            # which is why generating them pays while rewriting entity ones does not
            quality_band=_quality_band(looks_opaque=not words, n_alpha_words=len(words),
                                       has_description=False, ambiguity=1),
            label_len_chars=len(txt), label_len_words=len(txt.split()),
            n_alpha_words=len(words), looks_opaque=not words,
            has_description=False, description_len=0,
            ambiguity=1, degree=d, degree_percentile=lo,
        )

    return out


def feature_report(feats: dict[str, ElementFeatures]) -> dict:
    """Distributions -- run this BEFORE Chapter 3 to check the ladder has signal."""
    ents = [f for f in feats.values() if f.kind == "entity"]
    rels = [f for f in feats.values() if f.kind == "relation"]
    n = len(ents) or 1
    bands = Counter(f.quality_band for f in ents)
    types_ = Counter(f.semantic_type for f in ents)
    return {
        "n_entities": len(ents),
        "n_relations": len(rels),
        "quality_bands": {k: v / n for k, v in bands.items()},
        "n_semantic_types": len(types_),
        "type_OTHER_rate": types_.get("OTHER", 0) / n,
        "opaque_rate": sum(f.looks_opaque for f in ents) / n,
        "has_description_rate": sum(f.has_description for f in ents) / n,
        "ambiguous_rate": sum(f.ambiguity > 1 for f in ents) / n,
        "median_degree": sorted(f.degree for f in ents)[len(ents) // 2] if ents else 0,
        # the ladder needs variation at every rung, or a flat result is
        # uninterpretable rather than informative
        "L2_usable": len(types_) >= 3 and types_.get("OTHER", 0) / n < 0.5,
        "L3_usable": len(bands) >= 2 and max(bands.values()) / n < 0.9,
    }


def entropy(feats: dict[str, ElementFeatures], attr: str) -> float:
    """Diagnostic: a near-zero entropy rung cannot condition on anything."""
    c = Counter(getattr(f, attr) for f in feats.values())
    tot = sum(c.values()) or 1
    return -sum((v / tot) * math.log2(v / tot) for v in c.values() if v)
