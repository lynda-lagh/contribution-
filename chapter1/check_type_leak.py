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


def trivial_rules(types: dict[str, str], kg) -> list[tuple[str, object]]:
    """
    ★★ THE ONE-LINE RULES, MATCHED TO THE TAG FAMILY IN USE.

    ✋ THE BUG THIS FIXES. The rule was hardcoded as

            says_yes = types.get(tail) == f"{relation}::tail"

    which is an INDUCED-tag pattern. `build_types` is now exogenous-first and
    returns WordNet/wikicat classes (`football_team`, `administrative_district`).
    That string can never equal `playsFor::tail`, so the rule answered No to
    every row, scored exactly the negative base rate on a balanced set, and
    printed **"clean"** — a pass that proves nothing. The audit that exists to
    stop a silent artefact had itself become one.

    The exogenous analogue of "does the tag name the relation" is "does the tag
    match what this relation's tails normally ARE", learned from the training
    edges only. We run BOTH available strengths and report the harder floor:

      MODAL   tail class == the single most common tail class of r
      RANGE   tail class in the SET of classes seen as tails of r

    ✋ AND RANGE IS TAUTOLOGICAL ON THIS TEST SET — measured, not suspected.
       Reporting RANGE alone returned:

           tag-only rule accuracy 50.0%   (clean)
           P(rule fires)   positives 100.0%   negatives 100.0%   separation 0.0%

       A rule that fires on every row of both classes has not been cleared, it
       has not been asked anything. The cause is that
       `src.data.negatives.build_relation_type_index` draws a type-consistent
       negative uniformly from the ENTITIES observed as tails of r, so the
       corrupted tail is itself a training tail of r and its class is in the
       class-range by construction. The check restated the sampler.

       This is the third vacuous pass in this file's history — first 100% OTHER
       types, then an induced rule against exogenous tags, now a rule that
       mirrors the negative generator. All three printed a tick. `saturated()`
       below now refuses to call any of them clean.

       MODAL is not tautological: negatives are uniform over the range while
       positives concentrate on whatever the relation actually takes, so the
       two can separate. It is the rule that carries the floor.
    """
    modal: dict[str, str] = {}
    rng_: dict[str, set[str]] = {}
    counts: dict[str, Counter] = {}
    for t in kg.train:
        c = types.get(t.tail)
        if c is None:
            continue
        counts.setdefault(t.relation, Counter())[c] += 1
        rng_.setdefault(t.relation, set()).add(c)
    for r, c in counts.items():
        modal[r] = c.most_common(1)[0][0]

    # induced tags still exist as a fallback path in build_types, so detect
    # rather than assume which family we were handed
    sample = next((v for v in types.values() if v), "")
    if "::" in sample:
        return [("tail tag == '{r}::tail'  [induced]",
                 lambda rel, tl: types.get(tl) == f"{rel}::tail")]

    return [
        ("tail class == the MODAL tail class of r  [exogenous]",
         lambda rel, tl: types.get(tl) == modal.get(rel)),
        ("tail class in the training RANGE of r  [exogenous]",
         lambda rel, tl: types.get(tl) in rng_.get(rel, ())),
    ]


def saturated(p_yes: float, p_no: float, eps: float = 0.02) -> str | None:
    """
    ✋ A CONSTANT RULE IS NOT A CLEAN RULE.

    If a rule fires on ~everything or ~nothing in BOTH classes, its accuracy is
    the base rate and it has measured nothing about the tags. On a balanced set
    that prints as exactly 50.0% — indistinguishable from a genuine null, and
    that is how a vacuous check passes review.
    """
    if p_yes > 1 - eps and p_no > 1 - eps:
        return "fires on EVERY row of both classes"
    if p_yes < eps and p_no < eps:
        return "fires on NO row of either class"
    return None


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
    rows = [r for r in test
            if r.get("label") is not None and r.get("relation") is not None]
    if not rows:
        print(f"[{cond_id}] test records carry no labels — cannot audit.")
        return None

    print(f"\n[{cond_id}] n = {len(rows):,}  "
          f"({sum(r['label'] == 1 for r in rows):,} positive)")

    scored = []
    for rule_name, rule in trivial_rules(types, kg):
        n = correct = 0
        pos_match = neg_match = n_pos = n_neg = 0
        by_rel: dict[str, list[int]] = {}
        for r in rows:
            rel, tl = r["relation"], r.get("tail")
            n += 1
            says_yes = bool(rule(rel, tl))
            truth_yes = r["label"] == 1
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

        acc = correct / n
        p_yes = pos_match / max(1, n_pos)
        p_no = neg_match / max(1, n_neg)
        vac = saturated(p_yes, p_no)
        verdict = ("VACUOUS" if vac else
                   "clean" if acc < 0.55 else
                   "MATERIAL LEAK" if acc < 0.75 else "MEASURING THE ARTEFACT")
        mark = {"VACUOUS": "✋", "clean": "✓", "MATERIAL LEAK": "⚠️",
                "MEASURING THE ARTEFACT": "✋"}[verdict]

        print(f"  {mark} {acc:6.1%}  {rule_name}")
        print(f"        fires on   positives {p_yes:5.1%}   negatives {p_no:5.1%}"
              f"   separation {abs(p_yes - p_no):5.1%}")
        if vac:
            print(f"        ✋ {vac} — this rule measured NOTHING. Its {acc:.1%} "
                  f"is the base rate, not a cleared floor.")
        worst = sorted(((c / max(1, t), r, t) for r, (c, t) in by_rel.items()
                        if t >= 30), reverse=True)[:3]
        if worst and not vac:
            print("        worst relations: "
                  + " · ".join(f"{r} {a:.1%} (n={t})" for a, r, t in worst))

        scored.append({"rule": rule_name, "tag_only_accuracy": acc,
                       "p_match_positive": p_yes, "p_match_negative": p_no,
                       "separation": abs(p_yes - p_no), "verdict": verdict,
                       "vacuous": bool(vac)})

    # ★ THE FLOOR IS THE HARDEST NON-VACUOUS RULE. A vacuous rule cannot lower
    #   it -- "the check returned 50%" is not evidence when the check was
    #   constant -- and if EVERY rule is vacuous there is no measured floor and
    #   we say so rather than emit a number.
    usable = [s for s in scored if not s["vacuous"]]
    if not usable:
        print(f"\n  ✋ [{cond_id}] every trivial rule was vacuous on this test "
              f"set — there is NO measured floor. Do not record one.")
        return {"condition": cond_id, "n": len(rows), "rules": scored,
                "tag_only_accuracy": None, "verdict": "NO USABLE RULE"}

    best = max(usable, key=lambda s: s["tag_only_accuracy"])
    acc = best["tag_only_accuracy"]
    print(f"\n  → floor for {cond_id}: {acc:.3f}  ({best['verdict']}, "
          f"via {best['rule'].split('  [')[0]})")
    if acc >= 0.55:
        print(f"    Compare {cond_id} against {acc:.1%}, NOT against 0.5 or B.")

    return {"condition": cond_id, "n": len(rows), "rules": scored,
            "tag_only_accuracy": acc, "rule": best["rule"],
            "p_match_positive": best["p_match_positive"],
            "p_match_negative": best["p_match_negative"],
            "separation": best["separation"], "verdict": best["verdict"]}


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

    measured = [d for d in out if d.get("tag_only_accuracy") is not None]
    if out and not measured:
        print("\n" + "-" * 72)
        print("✋ NO CONDITION PRODUCED A MEASURED FLOOR — every trivial rule "
              "was constant.\n   Leave TYPE_TAG_FLOOR unset: an unmeasured "
              "floor and a cleared one\n   are not the same thing, and "
              "preflight is right to refuse.")
    if measured:
        worst = max(measured, key=lambda d: d["tag_only_accuracy"])
        print("\n" + "-" * 72)
        print(f"worst: {worst['condition']} at {worst['tag_only_accuracy']:.1%} "
              f"({worst['verdict']})")
        print(f"  TYPE_TAG_FLOOR[{ns.dataset!r}] = "
              f"{worst['tag_only_accuracy']:.3f}")
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
