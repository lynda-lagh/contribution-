"""
★ DOES THE TYPE TAG LEAK THE LABEL?

THE WORRY
---------
An induced type is an entity's DOMINANT relation position over the whole graph,
and it is rendered on head and tail regardless of the query relation:

    Is this true: entity48634 [playsFor::head] is affiliated to
                  entity21777 [graduatedFrom::tail]?

The relation is `isAffiliatedTo`, and neither tag mentions it. That is by design
-- an entity has one type. But it creates a way for the label to leak.

    POSITIVE  (h, r, t)   t is a genuine tail of r, so t's dominant position is
                          more likely to BE `r::tail`.
    NEGATIVE  (h, r, t')  t' is corrupted. If it is drawn uniformly over all
                          entities (which is how the YAGO3-10 TEST negatives are
                          made), its tag is essentially a random type.

If that asymmetry is large, a model can answer

    "tail tag == {relation}::tail  ->  Yes"

and score well **with no relational knowledge at all** -- a shortcut that our own
prompt design handed it. Conditions C, D, E and G would be inflated, and the type
ladder would measure the artefact instead of the question.

This is the same failure mode as the two bugs already found (100% OTHER types,
and `render` reading the wrong switch): the condition still runs, still prints
plausible numbers, and silently measures nothing.

WHAT THIS REPORTS
-----------------
The accuracy of the trivial tag-only rule on the TEST set of a built condition.

    <55%  -> the tag carries almost no label information. Clean.
    >=55% -> ⚠️ MATERIAL. This is the floor for every typed condition, and the
             comparison "C vs B" is no longer valid: C gets these points free.
    >=75% -> ✋ the condition is mostly measuring the artefact. Fix before training.

⚠️ The first version of this file called anything under 65% "clean". That was too
   lenient and it mislabelled the measured YAGO3-10 value of 62.4% -- twelve
   points of free accuracy is not clean. Thresholds tightened.

Test records already carry head/relation/tail/label, so nothing is re-derived and
nothing is guessed.

    python -m chapter1.check_type_leak --dataset YAGO3-10 --condition C D E G
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.loaders import anonymise, load_kg, shuffle_surface_forms

from .conditions import CONDITIONS
from .data import build_types


def _kg_for(cond, dataset: str, root: str, seed: int):
    kg = load_kg(dataset, root)
    if cond.anonymise:
        kg = anonymise(kg)
    elif getattr(cond, "shuffle", False):
        kg = shuffle_surface_forms(kg, seed=seed)
    return kg


def audit(cond_id: str, dataset: str, root: str, seed: int) -> dict | None:
    cond = CONDITIONS[cond_id]
    if not cond.types:
        print(f"[{cond_id}] no type tags in this condition — nothing to leak.")
        return None

    tag = f"{dataset}-{cond_id}"
    path = Path(root, tag, "built", "test_instructions.json")
    if not path.exists():
        print(f"[{cond_id}] not built yet ({path})")
        return None

    kg = _kg_for(cond, dataset, root, seed)
    types = build_types(kg, cond, dataset)
    test = json.loads(path.read_text(encoding="utf-8"))

    n = correct = 0
    pos_match = neg_match = n_pos = n_neg = 0
    by_rel: dict[str, list[int]] = {}

    for r in test:
        lab = r.get("label")
        rel, tl = r.get("relation"), r.get("tail")
        if lab is None or rel is None:
            continue
        n += 1
        # the trivial rule: does the tail's induced type name this relation?
        says_yes = types.get(tl) == f"{rel}::tail"
        truth_yes = lab == 1
        correct += says_yes == truth_yes
        if truth_yes:
            n_pos += 1
            pos_match += says_yes
        else:
            n_neg += 1
            neg_match += says_yes
        by_rel.setdefault(rel, [0, 0])
        by_rel[rel][0] += says_yes == truth_yes
        by_rel[rel][1] += 1

    if not n:
        print(f"[{cond_id}] test records carry no labels — cannot audit.")
        return None

    acc = correct / n
    p_yes = pos_match / max(1, n_pos)
    p_no = neg_match / max(1, n_neg)

    verdict = ("clean" if acc < 0.55 else
               "MATERIAL LEAK" if acc < 0.75 else "MEASURING THE ARTEFACT")
    mark = {"clean": "✓", "MATERIAL LEAK": "⚠️", "MEASURING THE ARTEFACT": "✋"}[verdict]

    print(f"\n{mark} [{cond_id}] tag-only rule accuracy {acc:.1%}   ({verdict})")
    if acc >= 0.55:
        print(f"      → FLOOR for this condition is {acc:.1%}, not 50%. "
              f"Compare {cond_id} against {acc:.1%}, NOT against B.")
    print(f"      P(tail tag == '{{r}}::tail')   positives {p_yes:.1%}   "
          f"negatives {p_no:.1%}   separation {abs(p_yes - p_no):.1%}")
    print(f"      n = {n:,}  ({n_pos:,} positive / {n_neg:,} negative)")

    worst = sorted(((c / max(1, t), r, t) for r, (c, t) in by_rel.items()
                    if t >= 30), reverse=True)[:5]
    if worst:
        print("      most-leaking relations:")
        for a, r, t in worst:
            print(f"        {a:6.1%}  {r}  (n={t})")

    return {"condition": cond_id, "tag_only_accuracy": acc,
            "p_match_positive": p_yes, "p_match_negative": p_no,
            "separation": abs(p_yes - p_no), "n": n, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--condition", nargs="+", default=["C", "D", "E", "G"],
                    choices=list(CONDITIONS), metavar="ID")
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default=None, help="write the report here")
    ns = ap.parse_args()

    print("=" * 72)
    print(f"TYPE-TAG LEAK AUDIT — {ns.dataset}")
    print("  Can the label be predicted from the type tag alone? 50% = clean.")
    print("=" * 72)

    out = [a for c in ns.condition if (a := audit(c, ns.dataset, ns.root, ns.seed))]

    if out:
        worst = max(out, key=lambda d: d["tag_only_accuracy"])
        print("\n" + "-" * 72)
        print(f"worst: {worst['condition']} at {worst['tag_only_accuracy']:.1%} "
              f"({worst['verdict']})")
        if worst["tag_only_accuracy"] >= 0.55:
            print("\n★ WHAT TO DO — pick one, and say which in the paper:")
            print("  1. Report the tag-only accuracy as the FLOOR for C/D/E/G.")
            print("     Honest, costs nothing, and is itself a finding: it says the")
            print("     type-augmentation literature may be measuring the same thing.")
            print("  2. Make the TEST negatives type-consistent too, so the tag no")
            print("     longer separates the classes:")
            print("       python -m scripts.make_test_negatives --dataset "
                  f"{ns.dataset} --strategy type_consistent --seed {ns.seed}")
            print("     ⚠️ changes the test set — A and B must be re-evaluated on it.")
            print("  3. Tag only the entity NOT being corrupted. Removes the leak,")
            print("     but breaks symmetry with G and with the literature's prompts.")

    if ns.json:
        Path(ns.json).parent.mkdir(parents=True, exist_ok=True)
        Path(ns.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {ns.json}")


if __name__ == "__main__":
    main()
