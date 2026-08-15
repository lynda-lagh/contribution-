"""
CHAPTER 1 — data construction for every condition and prompt variant.

    python -m chapter1.data --condition A --dataset WN11
    python -m chapter1.data --all --dataset WN11

★ ONE builder for the whole grid. Serialization is a CONTROLLED VARIABLE: 16
distinct input representations exist across the corpus and no paper compares two
of them, so anything that is not the variable under test is held byte-identical.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from src.data.loaders import KG, Triple, anonymise, load_kg, shuffle_surface_forms
from src.data.negatives import build_relation_type_index, make_negatives
from src.data.prompts import ALPACA_NO_INPUT, NO, YES
from src.routing.types import entity_types
from src.utils.progress import phase, track

from .conditions import CONDITIONS, PROMPTS, STRUCTURAL_INSTRUCTION, Condition, PromptVariant


# =============================================================================
#  PROMPT RENDERING  — the only place a prompt string is ever produced
# =============================================================================
def render(t: Triple, kg: KG, pv: PromptVariant,
           types: dict[str, str] | None = None,
           demos: list[Triple] | None = None) -> str:
    """
    KG-LLM's question, verbatim, plus exactly the blocks `pv` switches on.

    P0   Is this true: {h} {r} {t}?
    P1   Is this true: {h} [Person] {r} {t} [Location]?
    P2   Is this true: {h} {r} {t}? Before answering, consider whether …
    P3   both
    P4   Other triples using {r}: … | Is this true: {h} {r} {t}?
    """
    h = kg.ent2txt.get(t.head, t.head)
    r = kg.rel2txt.get(t.relation, t.relation)
    tl = kg.ent2txt.get(t.tail, t.tail)

    if pv.types and types:
        th, tt = types.get(t.head), types.get(t.tail)
        h = f"{h} [{th}]" if th else h
        tl = f"{tl} [{tt}]" if tt else tl

    parts = []
    if pv.demonstrations and demos:
        # ★ RealKGC's mechanism: DEMONSTRATE the relation's type pattern rather
        # than assert it. The model compares the query against real instances.
        ex = "; ".join(f"{kg.ent2txt.get(d.head, d.head)} {r} "
                       f"{kg.ent2txt.get(d.tail, d.tail)}"
                       for d in demos[:pv.demonstrations])
        parts.append(f"Other triples using {r}: {ex}.")

    parts.append(f"Is this true: {h} {r} {tl}?")

    if pv.instruction:
        parts.append(STRUCTURAL_INSTRUCTION)

    return " ".join(parts)


# =============================================================================
#  DEMONSTRATION POOL
# =============================================================================
def build_types(kg: KG, cond, dataset: str, max_other: float = 0.5) -> dict[str, str]:
    """
    ★★ TYPES THAT ACTUALLY CARRY INFORMATION — with a hard guard.

    THE BUG THIS EXISTS TO PREVENT
    ------------------------------
    `entity_types(kg, method="auto")` picks the POS method, which reads a
    part-of-speech marker out of the identifier (`stool_NN_2`, WN18RR style).

        WN11 identifiers look like `__east_indian_1` -- NO POS marker.
        Result: 1 distinct type, OTHER = 100.0% of 38,588 entities.

    Every optional type block then renders nothing, so condition C produced a
    prompt BYTE-IDENTICAL to condition B:

        [B] Is this true: entity712 has instance entity21807?
        [C] Is this true: entity712 has instance entity21807?     <-- same

    C, D, E and G would all have silently run as B, and the ladder would have
    "shown" that types do not help. Anonymisation makes it worse (`entity712`
    carries nothing at all), but the failure is present on the RAW graph too.

    THE FIX
    -------
    Fall back to INDUCED types: an entity's type is the set of relation
    positions it occupies (`_has_instance::head`, `_type_of::tail`, ...).
    Derived from graph structure, which anonymisation PRESERVES -- relations are
    never hidden, only entities. So induced types are identical before and after
    anonymisation, which is exactly what conditions B->C need.

    ⚠️ Say so in the paper. An induced type is "participates in these relations",
    which is weaker than a semantic type like Person/Location. On WN11 it is the
    only type signal available. **FB13 has real semantic types** -- that is why
    the type conditions belong there.
    """
    for method in ("auto", "induced"):
        t = entity_types(kg, method=method)
        n = len(t) or 1
        other = sum(1 for v in t.values() if v in (None, "OTHER")) / n
        distinct = len({v for v in t.values() if v not in (None, "OTHER")})
        print(f"[types] {dataset} {cond.id}: method={method} -> "
              f"{distinct} distinct, OTHER={other:.1%}")
        if other <= max_other and distinct >= 2:
            return t

    raise SystemExit(
        f"\n★ TYPE EXTRACTION FAILED for {dataset} (condition {cond.id}).\n"
        f"  No method produced usable types: >{max_other:.0%} of entities are OTHER.\n"
        f"  Condition {cond.id} would render prompts IDENTICAL to the no-types\n"
        f"  condition, and the experiment would silently measure nothing.\n\n"
        f"  Options:\n"
        f"    1. Use FB13 — its relations (profession, nationality, place_of_birth)\n"
        f"       carry real domain/range, which is where `Person -bornIn-> Location`\n"
        f"       actually lives. WN11's ids (`__east_indian_1`) have no type marker.\n"
        f"    2. Supply types from an external source (WordNet lexnames, YAGO's\n"
        f"       type hierarchy) and pass them in.\n"
        f"  Refusing to build a condition that cannot differ from its baseline.")


def demo_pool(kg: KG, k: int = 50) -> dict[str, list[Triple]]:
    """Per relation, a few real training triples to show as examples (P4)."""
    pool: dict[str, list[Triple]] = {}
    for t in kg.train:
        pool.setdefault(t.relation, [])
        if len(pool[t.relation]) < k:
            pool[t.relation].append(t)
    return pool


# =============================================================================
#  BUILD
# =============================================================================
def build_condition(cond: Condition, dataset: str, root: str, n_triples: int,
                    seed: int, prompt_id: str = "P0", out_root: str | None = None) -> dict:
    from src.data.sampling import sample_triples
    from src.utils.config import set_all_seeds
    set_all_seeds(seed)

    kg = load_kg(dataset, root)
    if cond.anonymise:
        kg = anonymise(kg)          # ⚠️ anonymises train AND test consistently
    elif getattr(cond, "shuffle", False):
        # ★ real names kept, permuted across entities. Same vocabulary, same
        # lengths, same readability -- only the name↔entity binding is gone.
        kg = shuffle_surface_forms(kg, seed=seed)

    # ★ CONDITION-level and PROMPT-level `types` are two different switches.
    #
    #   cond.types  -- this condition TRAINS with type tags   (C, D, E, G)
    #   pv.types    -- this prompt VARIANT shows type tags    (P1, P3)
    #
    # `render` only consults `pv.types`, so condition C at prompt P0 computed the
    # types and then silently dropped them -- C came out byte-identical to B.
    # Fold the condition's switch into the effective variant so one flag governs
    # rendering and the two can never disagree again.
    pv = PROMPTS[prompt_id]
    if cond.types and not pv.types:
        pv = replace(pv, types=True)

    types = build_types(kg, cond, dataset) if pv.types else None
    demos = demo_pool(kg) if pv.demonstrations else None

    # ---- training instances -------------------------------------------------
    pos = sample_triples(kg.train, n_triples, seed=seed,
                         stratified=True, min_per_relation=10)

    records: list[dict] = []
    rng = random.Random(seed)
    n_out = len(pos) * (1 + cond.n_negatives)
    print(f"  building {len(pos):,} positives × (1 + {cond.n_negatives} negatives) "
          f"= {n_out:,} instances")

    # ★ Build the relation type index ONCE. It scans the entire training graph
    #   (1,079,040 triples on YAGO3-10). Rebuilding it inside the loop made
    #   condition D take ~1.5 h and condition E ~9 h; with it hoisted, both are
    #   minutes. Conditions A/B/C were never affected because `random` negatives
    #   do not touch the index.
    neg_index = (build_relation_type_index(kg)
                 if cond.negatives == "type_consistent" else None)
    if neg_index is not None:
        print(f"  [negatives] type index built once over {len(kg.train):,} "
              f"training triples -> {len(neg_index)} relations")

    for p in track(pos, f"[{cond.id}] train instances", total=len(pos), unit="triple"):
        d = demos.get(p.relation) if demos else None
        records.append({"instruction": render(p, kg, pv, types, d),
                        "input": "", "output": YES})
        # ★ n_negatives per positive -- the axis D/E move
        for _ in range(cond.n_negatives):
            neg = make_negatives([p], kg, strategy=cond.negatives,
                                 seed=rng.randrange(1 << 30),
                                 type_index=neg_index)[0]
            records.append({"instruction": render(neg, kg, pv, types, d),
                            "input": "", "output": NO})
    rng.shuffle(records)

    # ---- test instances -----------------------------------------------------
    test = [{**{"instruction": render(t, kg, pv, types,
                                      demos.get(t.relation) if demos else None),
                "input": "", "output": YES if t.label == 1 else NO},
             "label": t.label, "head": t.head, "relation": t.relation, "tail": t.tail}
            for t in track(kg.test, f"[{cond.id}] test instances",
                           total=len(kg.test), unit="triple")]

    # ---- ★ seen/unseen split, computed at build time ------------------------
    # We train on 10k of WN11's 112,581 triples, so roughly half the test
    # entities were never seen. That accidental inductive split is an
    # INDEPENDENT test of memorisation -- see analysis.seen_unseen().
    seen = {e for t in pos for e in (t.head, t.tail)}
    for r in test:
        r["seen_head"] = r["head"] in seen
        r["seen_tail"] = r["tail"] in seen
        r["seen_both"] = r["seen_head"] and r["seen_tail"]

    # ★★ DIFFERENTIATION GUARD.
    # A condition that renders the same prompts as its baseline measures nothing.
    # This is the failure that produced identical B and C prompts, and it is
    # invisible downstream: training succeeds, accuracy looks plausible, and the
    # ladder reports "types do not help" when types were never present.
    if cond.types:
        from .conditions import Condition
        bare = Condition(cond.id + "_bare", cond.anonymise, False,
                         cond.negatives, cond.n_negatives, "", "")
        same = sum(1 for r, p in zip(records[:200], pos[:200])
                   if r["instruction"] == render(p, kg, PROMPTS["P0"], None, None))
        typed_marker = sum(1 for r in records[:200] if "[" in r["instruction"])
        if typed_marker == 0:
            raise SystemExit(
                f"\n★ CONDITION {cond.id} HAS types=True BUT NO TYPE TAG APPEARS "
                f"IN ANY PROMPT.\n  It would be identical to the no-types condition. "
                f"See build_types() for why.\n"
                f"  Example: {records[0]['instruction'][:100]}")
        print(f"[guard] {cond.id}: {typed_marker}/200 prompts carry a type tag ✓")

    tag = f"{dataset}-{cond.id}" + (f"-{prompt_id}" if prompt_id != "P0" else "")
    out = Path(out_root or root, tag, "built")
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_instructions.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    (out / "test_instructions.json").write_text(json.dumps(test, indent=1), encoding="utf-8")

    manifest = {
        "condition": cond.id, "prompt": prompt_id, "dataset": dataset, "seed": seed,
        "anonymised": cond.anonymise, "types": cond.types,
        "negatives": cond.negatives, "n_negatives": cond.n_negatives,
        "n_triples_sampled": len(pos),
        "n_train_instances": len(records),      # ⚠️ report alongside triples
        "n_test_instances": len(test),
        "n_entities_seen_in_training": len(seen),
        "n_entities_total": len(kg.ent2txt),
        "seen_coverage": len(seen) / max(1, len(kg.ent2txt)),
        "test_seen_both_rate": sum(r["seen_both"] for r in test) / max(1, len(test)),
        "isolates": cond.isolates, "reference": cond.reference,
        "example_positive": records[0] if records else None,
        "example_test": test[0] if test else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="A", choices=list(CONDITIONS))
    ap.add_argument("--prompt", default="P0", choices=list(PROMPTS))
    ap.add_argument("--dataset", default="WN11")
    ap.add_argument("--root", default="data")
    ap.add_argument("--n_triples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true", help="build every condition at P0")
    ns = ap.parse_args()

    todo = list(CONDITIONS) if ns.all else [ns.condition]
    for cid in todo:
        m = build_condition(CONDITIONS[cid], ns.dataset, ns.root,
                            ns.n_triples, ns.seed, ns.prompt)
        print(f"\n[{cid}] {m['n_triples_sampled']:,} triples -> "
              f"{m['n_train_instances']:,} instances | "
              f"seen coverage {m['seen_coverage']:.1%} | "
              f"test seen-both {m['test_seen_both_rate']:.1%}")
        print(f"     {m['example_positive']['instruction'][:110]}")


if __name__ == "__main__":
    main()
