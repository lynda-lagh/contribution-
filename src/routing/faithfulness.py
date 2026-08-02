"""
FAITHFULNESS TEST -- the Stage E contribution.

    Explanation says feature X caused the decision
                    |
        remove / change X   (counterfactual)
                    |
        does the decision change?
                    |
    YES -> faithful      NO -> the stated reason was not operative

Why this is possible here and nowhere else
------------------------------------------
Our explanations are FEATURE ATTRIBUTIONS produced by a deterministic router, so
they have a counterfactual. Generated prose does not: you cannot ablate "the model
said it was thinking step by step".

The gap it fills
----------------
  * 106 of 188 papers claim explainability; ~1 evaluates explanations at all.
  * That one -- EZ-Check -- uses BLEU, ROUGE and BERTScore against its OWN
    retrieved evidence, and lists faithfulness under Future Work:
    "we plan to conduct human-in-the-loop evaluations to assess the clarity,
     faithfulness, and trustworthiness of generated explanations".
  * EZ-Check's own §5.3.1 shows why grounding is not enough: "some explanations
    closely rephrase the underlying triplets ... factually aligned but offer
    limited additional insight". Grounded, yet explaining nothing.

An explanation can be perfectly grounded and still be decoration. A decision-flip
test separates the two mechanically.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from .features import ElementFeatures
from .router import Decision, Router

# How to counterfactually negate each feature the router may cite.
# Each must produce a DIFFERENT value while leaving everything else intact.
COUNTERFACTUALS = {
    "looks_opaque":      lambda f: replace(f, looks_opaque=not f.looks_opaque,
                                           n_alpha_words=2 if f.looks_opaque else 0),
    "has_description":   lambda f: replace(f, has_description=not f.has_description,
                                           description_len=40 if not f.has_description else 0),
    "ambiguity":         lambda f: replace(f, ambiguity=1 if f.ambiguity > 3 else 9),
    "quality_band":      lambda f: replace(f, quality_band=(
                             "poor" if f.quality_band == "rich" else "rich")),
    "semantic_type":     lambda f: replace(f, semantic_type=(
                             "PERSON" if f.semantic_type == "OTHER" else "OTHER")),
    "degree_percentile": lambda f: replace(f, degree_percentile=(
                             0.95 if f.degree_percentile < 0.5 else 0.05)),
    "kind":              lambda f: replace(f, kind=(
                             "relation" if f.kind == "entity" else "entity")),
}


def test_one(f: ElementFeatures, router: Router) -> dict:
    """
    Ablate EACH feature the explanation names; record whether the action changed.

    faithful           at least one cited feature, when changed, changes the action
    partially_faithful some cited features matter, others do not
    unfaithful         no cited feature changes anything -> the reason is decoration
    """
    original: Decision = router.route(f)
    cited = [c for c in original.reason_features if c in COUNTERFACTUALS]

    per_feature = {}
    for feat in cited:
        d_cf = router.route(COUNTERFACTUALS[feat](f))
        per_feature[feat] = {
            "action_before": original.action,
            "action_after": d_cf.action,
            "flipped": d_cf.action != original.action,
            "reason_after": d_cf.reason,
        }

    n_flip = sum(v["flipped"] for v in per_feature.values())
    verdict = ("no_features_cited" if not cited
               else "faithful" if n_flip == len(cited)
               else "partially_faithful" if n_flip > 0
               else "unfaithful")

    return {
        "element_id": f.element_id,
        "action": original.action,
        "reason": original.reason,
        "cited_features": cited,
        "n_cited": len(cited),
        "n_flipped": n_flip,
        "verdict": verdict,
        "per_feature": per_feature,
    }


def evaluate(feats: dict[str, ElementFeatures], level: str = "L3",
             out_path: str | None = None) -> dict:
    """
    DECISION-FLIP RATE per explanation type -- Chapter 3's headline table.

    Reported per REASON TEMPLATE, not just globally: a router can be faithful
    about opacity and unfaithful about ambiguity, and that distinction is the
    actionable part.
    """
    router = Router(level)
    tests = [test_one(f, router) for f in feats.values()]

    by_reason: dict[str, list] = defaultdict(list)
    for t in tests:
        by_reason[t["reason"][:60]].append(t)

    per_reason = {}
    for reason, ts in by_reason.items():
        n = len(ts)
        flips = sum(t["n_flipped"] > 0 for t in ts)
        per_reason[reason] = {
            "n": n,
            "decision_flip_rate": flips / n,
            "verdicts": dict(Counter(t["verdict"] for t in ts)),
            "action": ts[0]["action"],
        }

    verdicts = Counter(t["verdict"] for t in tests)
    n = len(tests) or 1
    testable = [t for t in tests if t["n_cited"] > 0]

    out = {
        "level": level,
        "n_decisions": n,
        "n_testable": len(testable),
        "verdicts": {k: v / n for k, v in verdicts.items()},
        "overall_decision_flip_rate": (
            sum(t["n_flipped"] > 0 for t in testable) / len(testable) if testable else 0.0),
        "per_reason": per_reason,
        "interpretation": (
            "Decision-flip rate is the fraction of stated reasons that are OPERATIVE: "
            "changing the named feature changes the decision. A low rate means the "
            "explanations are decoration. 106/188 papers claim explainability; none "
            "measures this."
        ),
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))

    print(f"[faithfulness] {level}: flip rate "
          f"{out['overall_decision_flip_rate']:.1%} over {len(testable)} testable decisions")
    for v, r in sorted(out["verdicts"].items(), key=lambda kv: -kv[1]):
        print(f"    {v:20s} {r:6.1%}")
    return out


def probe(feats: dict[str, ElementFeatures], element_id: str, level: str = "L3") -> dict:
    """
    Single-element probe -- what the web app's [Test this reason] button calls.

        "skipped because the label is opaque"
        -> make the label readable, re-run
        -> decision flipped => the reason was real
    """
    router = Router(level)
    f = feats[element_id]
    r = test_one(f, router)
    print(f"\n  element  {element_id}")
    print(f"  action   {r['action']}")
    print(f"  reason   {r['reason']}")
    for feat, res in r["per_feature"].items():
        mark = "FLIPPED" if res["flipped"] else "unchanged"
        print(f"    change '{feat}' -> {res['action_after']:22s} [{mark}]")
    print(f"  verdict  {r['verdict']}")
    return r
