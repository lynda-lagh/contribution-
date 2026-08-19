"""
★ CHAPTER 3 — THE ONE TABLE THE CHAPTER IS ABOUT.

The question in the chapter title is *at what granularity does conditioning stop
paying?* That cannot be answered by any single artefact currently on disk:

    ch3_analysis_*.json   routing + faithfulness, no accuracy, estimated tokens
    ch3-*-lora.json       train_loss + runtime, no accuracy
    ch3_eval_*.json       accuracy                          (chapters.ch3_conditioning.evaluate)
    ch3_tokens_*.json     measured tokens                   (chapters.ch3_conditioning.measure)

This joins all four into one row per level:

    level · balanced accuracy · measured tokens · faithfulness · runtime

and then applies the only test that matters: **is this level dominated?** A level
that costs more tokens, scores no better, and explains itself worse than a
simpler level is dominated, and a dominated level is a negative result worth
reporting rather than a failure worth hiding.

    python -m chapters.ch3_conditioning.report --dataset YAGO3-10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LEVELS = ("L0", "L1", "L2", "L3", "L4")
DESC = {"L0": "none (KG-LLM baseline)", "L1": "entity / relation",
        "L2": "semantic type", "L3": "label quality", "L4": "instance"}


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--results", default=None)
    ns = ap.parse_args()

    from src.utils.config import load_config
    cfg = load_config(ns.config)
    res = Path(ns.results or cfg["output"]["results_dir"])
    method = cfg["peft"]["method"]

    analysis = load(res / f"ch3_analysis_{ns.dataset}.json")
    evals = load(res / f"ch3_eval_{ns.dataset}.json")
    tokens = load(res / f"ch3_tokens_{ns.dataset}_train.json").get("levels", {})
    runs = {lv: load(res / f"ch3-{ns.dataset}-{lv}-{method}.json") for lv in LEVELS}

    have = {"routing": bool(analysis), "accuracy": bool(evals),
            "measured tokens": bool(tokens), "runtime": any(runs.values())}
    print("inputs:", "  ".join(f"{'✅' if v else '❌'} {k}" for k, v in have.items()))
    for k, v in have.items():
        if not v:
            hint = {"accuracy": "python -m chapters.ch3_conditioning.evaluate",
                    "measured tokens": "python -m chapters.ch3_conditioning.measure",
                    "routing": "python -m chapters.ch3_conditioning.run --analyse",
                    "runtime": "train the levels"}[k]
            print(f"   ↳ missing {k}: {hint} --dataset {ns.dataset}")

    rows = []
    for lv in LEVELS:
        r = {"level": lv, "conditioning": DESC[lv]}
        e = evals.get(lv, {})
        r["accuracy"] = e.get("accuracy")
        r["balanced_accuracy"] = e.get("balanced_accuracy")
        r["degenerate"] = e.get("degenerate")
        t = tokens.get(lv, {})
        r["tokens_total"] = t.get("total_tokens")
        r["tokens_mean"] = t.get("mean_tokens")
        f = (analysis.get("faithfulness") or {}).get(lv, {})
        v = f.get("verdicts") or {}
        r["faithful_rate"] = v.get("faithful")
        r["skip_rate"] = (analysis.get("routing") or {}).get(lv, {}).get("skip_rate")
        r["train_s"] = (runs.get(lv) or {}).get("train_runtime_s")
        rows.append(r)

    print(f"\n{'='*94}")
    print(f"CHAPTER 3 — COST AND BENEFIT BY CONDITIONING LEVEL ({ns.dataset})")
    print("=" * 94)
    print(f"{'lvl':4s} {'conditioning':22s} {'bal acc':>8s} {'tokens':>12s} "
          f"{'tok/prompt':>10s} {'faithful':>9s} {'skip':>6s} {'train s':>8s}")
    fmt = lambda v, s: (s.format(v) if v is not None else "—")
    for r in rows:
        print(f"{r['level']:4s} {r['conditioning']:22s} "
              f"{fmt(r['balanced_accuracy'], '{:>8.4f}'):>8s} "
              f"{fmt(r['tokens_total'], '{:>12,d}'):>12s} "
              f"{fmt(r['tokens_mean'], '{:>10.1f}'):>10s} "
              f"{fmt(r['faithful_rate'], '{:>9.1%}'):>9s} "
              f"{fmt(r['skip_rate'], '{:>6.1%}'):>6s} "
              f"{fmt(r['train_s'], '{:>8.0f}'):>8s}"
              + ("  ⚠️ degenerate" if r.get("degenerate") else ""))

    # ---- domination -------------------------------------------------------
    # A level is dominated if a SIMPLER level is at least as accurate, at least
    # as cheap, and at least as faithful. Nothing subtle; that is the point.
    print(f"\n{'-'*94}")
    print("DOMINANCE — is a simpler level at least as good on every axis?")
    print("-" * 94)
    idx = {r["level"]: r for r in rows}
    any_dom = False
    for i, lv in enumerate(LEVELS):
        r = idx[lv]
        if r["balanced_accuracy"] is None or r["tokens_total"] is None:
            continue
        for prev in LEVELS[:i]:
            s = idx[prev]
            if s["balanced_accuracy"] is None or s["tokens_total"] is None:
                continue
            acc_ok = s["balanced_accuracy"] >= r["balanced_accuracy"] - 0.005
            tok_ok = s["tokens_total"] <= r["tokens_total"]
            fa_r, fa_s = r.get("faithful_rate"), s.get("faithful_rate")
            fai_ok = (fa_s is None or fa_r is None or fa_s >= fa_r)
            if acc_ok and tok_ok and fai_ok:
                any_dom = True
                print(f"  ✋ {lv} is DOMINATED by {prev}: "
                      f"bal acc {s['balanced_accuracy']:.4f} ≥ {r['balanced_accuracy']:.4f}, "
                      f"tokens {s['tokens_total']:,} ≤ {r['tokens_total']:,}"
                      + (f", faithful {fa_s:.0%} ≥ {fa_r:.0%}" if fa_s and fa_r else ""))
                break
    if not any_dom:
        print("  no level is dominated — each buys something the simpler ones do not")

    # ---- what granularity actually bought ---------------------------------
    if idx["L1"]["tokens_total"] and idx["L0"]["tokens_total"]:
        b, l1 = idx["L0"]["tokens_total"], idx["L1"]["tokens_total"]
        print(f"\n{'-'*94}")
        print("WHERE THE SAVINGS COME FROM")
        print("-" * 94)
        print(f"  L0 -> L1 (a FIXED policy, no routing):  {(b-l1)/b:+.2%} of baseline")
        for lv in ("L2", "L3", "L4"):
            t = idx[lv]["tokens_total"]
            if t:
                print(f"  L1 -> {lv} (routing granularity):        {(l1-t)/b:+.2%}")
        print("\n  ★ If the first line dominates the rest, the headline saving is not")
        print("    a routing result. Attribute it correctly — it is still a real")
        print("    saving, just not evidence that granularity pays.")

    out = res / f"ch3_report_{ns.dataset}.json"
    out.write_text(json.dumps({"dataset": ns.dataset, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
