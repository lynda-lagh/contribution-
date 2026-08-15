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
from pathlib import Path

from src.data.loaders import KG, Triple, anonymise, load_kg
from src.data.negatives import build_relation_type_index, make_negatives
from src.data.prompts import ALPACA_NO_INPUT, NO, YES
from src.routing.types import entity_types

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

    pv = PROMPTS[prompt_id]
    types = entity_types(kg) if (cond.types or pv.types) else None
    demos = demo_pool(kg) if pv.demonstrations else None

    # ---- training instances -------------------------------------------------
    pos = sample_triples(kg.train, n_triples, seed=seed,
                         stratified=True, min_per_relation=10)

    records: list[dict] = []
    rng = random.Random(seed)
    for p in pos:
        d = demos.get(p.relation) if demos else None
        records.append({"instruction": render(p, kg, pv, types, d),
                        "input": "", "output": YES})
        # ★ n_negatives per positive -- the axis D/E move
        for _ in range(cond.n_negatives):
            neg = make_negatives([p], kg, strategy=cond.negatives, seed=rng.randrange(1 << 30))[0]
            records.append({"instruction": render(neg, kg, pv, types, d),
                            "input": "", "output": NO})
    rng.shuffle(records)

    # ---- test instances -----------------------------------------------------
    test = [{**{"instruction": render(t, kg, pv, types,
                                      demos.get(t.relation) if demos else None),
                "input": "", "output": YES if t.label == 1 else NO},
             "label": t.label, "head": t.head, "relation": t.relation, "tail": t.tail}
            for t in kg.test]

    # ---- ★ seen/unseen split, computed at build time ------------------------
    # We train on 10k of WN11's 112,581 triples, so roughly half the test
    # entities were never seen. That accidental inductive split is an
    # INDEPENDENT test of memorisation -- see analysis.seen_unseen().
    seen = {e for t in pos for e in (t.head, t.tail)}
    for r in test:
        r["seen_head"] = r["head"] in seen
        r["seen_tail"] = r["tail"] in seen
        r["seen_both"] = r["seen_head"] and r["seen_tail"]

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
