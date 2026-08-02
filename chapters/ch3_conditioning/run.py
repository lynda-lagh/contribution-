"""
CHAPTER 3 -- At what granularity does conditioning stop paying?

    L0  none (KG-LLM baseline)   L1  entity/relation   L2  semantic type
    L3  label quality            L4  instance

Outcomes, all publishable
-------------------------
    gains grow to L4     per-instance conditioning matters
    gains saturate at L2 useful negative result -- saves the next person the work
    gains flat           the SPECIFICITY analogue of "more context is not better"
                         (P02, P08, P11, P12, P19, P20, P21)
    gains NEGATIVE       ★ most interesting: over-conditioning hurts, mirroring
                         GS-KGC's finding that neighbours ALONE score below
                         no-context on 3/3 datasets

★ Why this chapter cannot fail: the ∅ branch. If conditioning is worse at every
level, the router learns to always skip, and we report "the optimal policy is not
to condition" -- with evidence, across four granularities.

    python -m chapters.ch3_conditioning.run --analyse            # routing + faithfulness, no training
    python -m chapters.ch3_conditioning.run --level L3 --train   # one training run
    python -m chapters.ch3_conditioning.run --plan               # the 6-run plan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data.build_instructions import build
from src.data.loaders import load_kg
from src.data.prompts import PromptConfig, build_enrichment_extras
from src.routing.faithfulness import evaluate as eval_faithfulness
from src.routing.features import compute_features, feature_report
from src.routing.router import Router, ladder
from src.train.sft import train_sft

LEVELS = ("L0", "L1", "L2", "L3", "L4")


def prompt_cfg_for(level: str, feats, plan, kg) -> PromptConfig:
    """
    Translate routing decisions into prompt content.

    CONTENT routing, not adapter routing: one adapter per LEVEL (6 runs), not one
    per bucket (24+). The ∅ branch still removes tokens, so the cost argument
    against MKGL's finding holds -- see router.py's design note.

    ⚠️ Flags alone are inert. `build_enrichment_extras` must supply the CONTENT for
    every flag we switch on, otherwise all five levels emit IDENTICAL prompts and
    the ladder measures nothing. PromptConfig.__post_init__ now enforces this.
    """
    per_element = {eid: d.action for eid, d in plan.decisions.items()}

    # L0 is the KG-LLM BASELINE: entity description only, nothing else. If L0 also
    # carried relation descriptions there would be no un-conditioned control and the
    # whole ladder would be measured against the wrong floor.
    rel_desc = level in ("L1", "L2", "L3", "L4")
    types    = level in ("L2", "L3", "L4")
    excl     = level in ("L3", "L4")
    nbrs     = level in ("L2", "L3", "L4")

    extras = build_enrichment_extras(
        kg, relation_descriptions=rel_desc, entity_types=types,
        exclusions=excl, neighbours=nbrs, n_neighbours=5)

    # ★ the ∅ DO-NOT-ENRICH branch: elements the router chose to skip get their
    # enrichment REMOVED, which is where the token savings come from.
    skipped = {eid for eid, act in per_element.items() if act in ("skip", "none", "∅")}
    if skipped:
        for key in ("entity_types", "neighbours"):
            if key in extras:
                extras[key] = {k: v for k, v in extras[key].items() if k not in skipped}
        if "relation_descriptions" in extras:
            extras["relation_descriptions"] = {
                k: v for k, v in extras["relation_descriptions"].items() if k not in skipped}
        if "exclusions" in extras:
            extras["exclusions"] = {k: v for k, v in extras["exclusions"].items()
                                    if k[0] not in skipped}

    extras |= {"routing": per_element, "level": level}

    return PromptConfig(
        include_entity_description=True,
        regenerate_entity_description=False,       # ColKGC: ~0 gain
        include_relation_description=rel_desc,
        include_exclusion_list=excl,
        include_type_tag=types,
        include_neighbours=nbrs,
        n_neighbours=5,
        typed_neighbours=True,                     # KG-LLM/APE verbalise UNTYPED; we do not
        extras=extras,
    )


def analyse(cfg: dict, dataset: str) -> dict:
    """No training. Routing distributions, token savings, and faithfulness."""
    kg = load_kg(dataset, cfg["data"]["root"])
    feats = compute_features(kg)

    rep = feature_report(feats)
    print("\n[ch3] feature report")
    for k, v in rep.items():
        print(f"    {k:24s} {v if not isinstance(v, float) else round(v, 4)}")
    if not rep["L2_usable"]:
        print("    ⚠️ L2 has little to condition on -- report this as the "
              "label-opacity boundary (UKGEBN / GS-KGC / MKGL), not as a failure")
    if not rep["L3_usable"]:
        print("    ⚠️ L3 bands are degenerate -- check the quality thresholds")

    # ★ PRE-FLIGHT: the ∅ branch fires only for the "rich" band. If no element
    # reaches "rich", the router can never skip, and Chapter 3's cost argument
    # (the answer to MKGL's 10.6x finding) has nothing to stand on.
    rich = rep["quality_bands"].get("rich", 0.0)
    if rich < 0.02:
        print(f"\n    ⚠️⚠️ 'rich' band = {rich:.1%} -- the ∅ DO-NOT-ENRICH branch will "
              "almost never fire.")
        print("        Check: are entity2text.txt entries actually 'name, gloss', or "
              "bare names?")
        print("        If descriptions really are absent, that is a FINDING (enrichment "
              "always pays here),")
        print("        but say so explicitly rather than reporting a 0% skip rate as a "
              "routing result.")
    else:
        print(f"\n    ✅ 'rich' band = {rich:.1%} -- the ∅ branch has elements to skip")

    print("\n[ch3] routing ladder")
    plans = ladder(feats, LEVELS)

    print("\n[ch3] faithfulness")
    faith = {lv: eval_faithfulness(feats, lv) for lv in ("L1", "L2", "L3", "L4")}

    out = {
        "dataset": dataset,
        "feature_report": rep,
        "routing": {lv: p.summary() for lv, p in plans.items()},
        "faithfulness": {lv: {k: v for k, v in f.items() if k != "per_reason"}
                         for lv, f in faith.items()},
        "faithfulness_per_reason": {lv: f["per_reason"] for lv, f in faith.items()},
    }
    res = Path(cfg["output"]["results_dir"]); res.mkdir(parents=True, exist_ok=True)
    (res / f"ch3_analysis_{dataset}.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 66)
    print("CHAPTER 3 -- routing (no training)")
    print("=" * 66)
    print(f"{'level':6s} {'skip':>7s} {'tokens':>10s} {'vs L0':>8s} {'flip rate':>10s}")
    for lv in LEVELS:
        s = plans[lv].summary()
        fr = faith.get(lv, {}).get("overall_decision_flip_rate")
        print(f"{lv:6s} {s['skip_rate']:6.1%} {s['routed_tokens']:10,d} "
              f"{s['savings_rate']:+7.1%} {('-' if fr is None else f'{fr:9.1%}')}")
    print(f"\nsaved -> {res / f'ch3_analysis_{dataset}.json'}")
    return out


def train_level(cfg: dict, dataset: str, level: str) -> dict:
    kg = load_kg(dataset, cfg["data"]["root"])
    feats = compute_features(kg)
    plan = Router(level).route_all(feats)

    name = f"{dataset}-{level}"
    data_dir = Path(cfg["data"]["root"], name, "built")
    if not (data_dir / "train_instructions.json").exists():
        build(dataset=dataset, n_triples=cfg["data"]["train_triples"], seed=cfg["seed"],
              root=cfg["data"]["root"], negatives=cfg["data"]["negatives"],
              out_dir=str(data_dir), prompt_cfg=prompt_cfg_for(level, feats, plan, kg))

    rid = f"ch3-{dataset}-{level}-{cfg['peft']['method']}"
    summary = train_sft(cfg, str(data_dir), str(Path(cfg["output"]["adapter_dir"], rid)),
                        run_name=rid)
    summary |= {"level": level, "routing": plan.summary()}
    res = Path(cfg["output"]["results_dir"]); res.mkdir(parents=True, exist_ok=True)
    (res / f"{rid}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--level", default="L3", choices=LEVELS)
    ap.add_argument("--analyse", action="store_true", help="routing + faithfulness, no training")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ns = ap.parse_args()

    if ns.plan:
        print("CHAPTER 3 PLAN -- 6 runs")
        print("-" * 46)
        for i, lv in enumerate(LEVELS, 1):
            print(f"{i}. ch3-{ns.dataset}-{lv}   (content routing, 1 adapter per level)")
        print("6. enrich-everything control (= L0)")
        print("-" * 46)
        print("Routing analysis + faithfulness cost NO training.")
        print("⚠️ Cut L4 first if compute runs short -- L1..L3 already answer the question.")
        return

    from src.utils.config import load_config
    cfg = load_config(ns.config)              # ★ seeds python/numpy/torch/HF -- raw
                                              # yaml.safe_load here silently skipped it
    if ns.analyse or not ns.train:
        analyse(cfg, ns.dataset)
    if ns.train:
        train_level(cfg, ns.dataset, ns.level)


if __name__ == "__main__":
    main()
