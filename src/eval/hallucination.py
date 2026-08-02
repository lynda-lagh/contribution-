"""
HALLUCINATION RATES -- type 1 (measured once) and type 2 (never measured).

Three incompatible taxonomies exist in the corpus, none citing the others:

  EGIT       type 1 out-of-vocabulary entity | type 2 real entity, structurally
             implausible                     -> MEASURES NEITHER
  Knit       knowledge deficiency (incl. "I don't know.") | knowledge
             over-generalisation ("processioning" for "procession") -> n=2 case study
  TSP        fabricated premises (reasoning over triples that do not exist) |
             rule violation                  -> case study

One measurement between them: GS-KGC counts out-of-vocabulary generation on
WN18RR at 38.9% / 45.3% ("the LLM generated 1,220 and 1,424 entities that do not
exist in WN18RR").

★ TYPE 2 IS THE GAP, and it is the one MKGL cannot close: MKGL's output space IS
the KG vocabulary, so every prediction is a VALID entity -- nothing stops it being
the WRONG valid entity.

We already have what type 2 needs: induced domain/range from
negatives.build_relation_type_index(), built for hard-negative sampling.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..data.loaders import KG
from ..data.negatives import build_relation_type_index


def _norm(s: str) -> str:
    return " ".join(s.strip().strip(".").split()).lower()


def build_surface_index(kg: KG) -> dict[str, str]:
    """
    Normalised surface form -> entity id, for grounding generated strings.

    ⚠️ KG-BERT's entity2text.txt is "name, description" -- e.g.
           NeXT  ->  "NeXT, computer company founded by Steve Jobs"
    A model asked to complete a triple emits the NAME ("NeXT"), never the full
    entry. Indexing only the full string makes every correct answer look
    out-of-vocabulary and would inflate the type-1 rate to near 100%.

    So we index THREE keys per entity -- full text, name-before-comma, and the raw
    id -- with the name taking precedence on collisions (it is what gets generated).
    """
    idx: dict[str, str] = {}
    for eid, text in kg.ent2txt.items():
        text = text or ""
        full = _norm(text)
        name = _norm(text.split(",")[0])
        if full and full not in idx:
            idx[full] = eid
        if _norm(eid) not in idx:
            idx[_norm(eid)] = eid
        if name:
            idx[name] = eid              # highest precedence -- overwrite
    return idx


def classify_prediction(pred_text: str, relation: str, kg: KG,
                        surface_index: dict[str, str],
                        type_index: dict[str, tuple[set, set]],
                        gold: str | None = None) -> dict:
    """
    Categories
    ----------
      correct              matches the gold entity
      type1_oov            ★ the string is not a KG entity at all (GS-KGC's measure)
      type2_type_violation ★ a REAL entity that violates the relation's observed range
                             (EGIT defined this; nobody has measured it)
      plausible_wrong      a real, type-valid entity that is simply not the gold one
                             -> candidate for the closed-world penalty (GS-KGC's X/Y)
    """
    norm = _norm(pred_text)
    eid = surface_index.get(norm)

    if eid is None:
        return {"category": "type1_oov", "entity_id": None,
                "reason": "generated string is not an entity in the KG"}

    if gold is not None and eid == gold:
        return {"category": "correct", "entity_id": eid, "reason": "matches gold"}

    observed_range = type_index.get(relation, (set(), set()))[1]
    if observed_range and eid not in observed_range:
        return {
            "category": "type2_type_violation", "entity_id": eid,
            # ★ Trace 3 -- the type-check reason, ready for the review queue
            "reason": (f"'{pred_text}' is a real entity but has never been observed "
                       f"in the range of '{relation}' (|range| = {len(observed_range)})"),
        }

    return {"category": "plausible_wrong", "entity_id": eid,
            "reason": ("real, type-valid entity that is not the gold answer -- "
                       "may be a closed-world artefact rather than an error")}


def hallucination_report(predictions: list[str], relations: list[str],
                         golds: list[str | None], kg: KG,
                         out_path: str | None = None) -> dict:
    """
    The two rates, side by side.

    type1_oov_rate  comparable with GS-KGC's 38.9% / 45.3% on WN18RR
    type2_rate      ★ first published measurement of EGIT's type 2
    """
    surface = build_surface_index(kg)
    types = build_relation_type_index(kg)

    detail = [classify_prediction(p, r, kg, surface, types, g)
              for p, r, g in zip(predictions, relations, golds)]
    cats = Counter(d["category"] for d in detail)
    n = len(detail) or 1

    out = {
        "n": n,
        "counts": dict(cats),
        "rates": {k: v / n for k, v in cats.items()},
        "type1_oov_rate": cats.get("type1_oov", 0) / n,
        "type2_rate": cats.get("type2_type_violation", 0) / n,
        "plausible_wrong_rate": cats.get("plausible_wrong", 0) / n,
        "accuracy": cats.get("correct", 0) / n,
        "baselines": {
            "gs_kgc_wn18rr_oov_forward": 0.389,
            "gs_kgc_wn18rr_oov_backward": 0.453,
        },
        "notes": [
            "type1 is comparable with GS-KGC's published 38.9% / 45.3% on WN18RR",
            "type2 was defined by EGIT and has never been measured",
            "MKGL's construction eliminates type1 by design and CANNOT prevent type2",
            "plausible_wrong is the closed-world penalty candidate set (GS-KGC's X/Y)",
        ],
        "examples": {c: [d["reason"] for d in detail if d["category"] == c][:5]
                     for c in cats},
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))

    print(f"[hallucination] type1 OOV {out['type1_oov_rate']:.1%} "
          f"(GS-KGC WN18RR baseline 38.9-45.3%) | "
          f"★ type2 violation {out['type2_rate']:.1%} | "
          f"plausible-wrong {out['plausible_wrong_rate']:.1%}")
    return out


def compare_methods(reports: dict[str, dict], out_path: str | None = None) -> dict:
    """
    Across LoRA / MoRA / BOFT / probe x SFT / DPO.

    ★ The direct test of whether DPO worked: preference optimisation on hard
    negatives should reduce type2 specifically -- and it may do so even if
    aggregate accuracy is flat, which would be the better result for a
    quality-oriented thesis.
    """
    rows = {m: {"accuracy": r["accuracy"], "type1_oov": r["type1_oov_rate"],
                "type2": r["type2_rate"], "plausible_wrong": r["plausible_wrong_rate"]}
            for m, r in reports.items()}
    best_t2 = min(rows.items(), key=lambda kv: kv[1]["type2"])
    out = {"per_method": rows, "lowest_type2": best_t2[0],
           "interpretation": ("If DPO lowers type2 without lowering accuracy, "
                              "preference optimisation reduced plausible-but-wrong "
                              "predictions at no cost -- the quality result.")}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"[hallucination] lowest type2: {best_t2[0]} ({best_t2[1]['type2']:.1%})")
    return out
