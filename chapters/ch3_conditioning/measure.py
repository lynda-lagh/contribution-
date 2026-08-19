"""
★ CHAPTER 3 — MEASURE WHAT WAS ACTUALLY BUILT, not what the router assumed.

THE PROBLEM THIS EXISTS TO FIX
------------------------------
`router.ACTION_TOKEN_COST` carries this comment:

    # Rough prompt-token cost of each action -- used for the savings report.
    # Calibrate against your tokeniser once; the ordering is what matters.

The calibration never happened. So the reported

    baseline_tokens 17,866,755   routed_tokens 3,080,290   saved 82.76%

is `123,219 elements x 145 tokens` against `x ~25`, i.e. an assumption restated,
not a measurement. Worse, the constants describe the router's *hypothetical*
"full" action, while `prompt_cfg_for("L0", ...)` disables relation descriptions,
types, exclusions and neighbours — so the prompts L0 was actually TRAINED on are
not the prompts the 17.9M figure describes.

WHAT THIS DOES INSTEAD
----------------------
Tokenises the prompts on disk — the exact strings each level trained on — with
the exact tokeniser the model uses, and reports both numbers side by side so the
discrepancy is visible rather than hidden.

    python -m chapters.ch3_conditioning.measure --dataset YAGO3-10

⚠️ The savings you can defend are the MEASURED ones. If they disagree with the
   router's estimate, that disagreement is itself worth a sentence: it is the
   difference between a policy's intended cost and its realised cost.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

LEVELS = ("L0", "L1", "L2", "L3", "L4")


def token_stats(prompts: list[str], tok) -> dict:
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    lens.sort()
    return {
        "n_prompts": len(lens),
        "total_tokens": sum(lens),
        "mean_tokens": statistics.mean(lens) if lens else 0.0,
        "median_tokens": statistics.median(lens) if lens else 0,
        "p95_tokens": lens[int(0.95 * (len(lens) - 1))] if lens else 0,
        "max_tokens": lens[-1] if lens else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 = every prompt")
    ap.add_argument("--json", default=None)
    ns = ap.parse_args()

    from transformers import AutoTokenizer

    from src.utils.config import load_config
    cfg = load_config(ns.config)
    root = cfg["data"]["root"]
    base = cfg["model"]["name"]

    print(f"tokeniser: {base}")
    tok = AutoTokenizer.from_pretrained(base)

    fname = f"{ns.split}_instructions.json"
    rows, missing = {}, []
    for lv in LEVELS:
        p = Path(root, f"{ns.dataset}-{lv}", "built", fname)
        if not p.exists():
            missing.append(lv)
            continue
        recs = json.loads(p.read_text(encoding="utf-8"))
        if ns.limit:
            recs = recs[: ns.limit]
        prompts = [r["instruction"] for r in recs]
        rows[lv] = token_stats(prompts, tok)
        rows[lv]["example"] = prompts[0][:300] if prompts else ""

    if missing:
        print(f"⚠️ not built: {', '.join(missing)} — "
              f"run `python -m chapters.ch3_conditioning.run --level {missing[0]} --train`")
    if not rows:
        raise SystemExit("nothing to measure")

    # the router's own estimate, for the side-by-side
    est = {}
    an = Path(cfg["output"]["results_dir"], f"ch3_analysis_{ns.dataset}.json")
    if an.exists():
        est = {lv: v for lv, v in json.loads(an.read_text(encoding="utf-8"))
               .get("routing", {}).items()}

    base_lv = "L0" if "L0" in rows else next(iter(rows))
    b_total = rows[base_lv]["total_tokens"]

    print(f"\n{'='*78}")
    print(f"MEASURED PROMPT TOKENS — {ns.dataset} / {ns.split} split")
    print(f"  tokenised on disk, baseline = {base_lv}")
    print("=" * 78)
    print(f"{'level':6s} {'prompts':>8s} {'total':>13s} {'mean':>7s} {'p95':>6s} "
          f"{'vs '+base_lv:>9s} {'router est.':>12s}")
    for lv in LEVELS:
        if lv not in rows:
            continue
        r = rows[lv]
        delta = (b_total - r["total_tokens"]) / b_total if b_total else 0.0
        e = est.get(lv, {}).get("savings_rate")
        r["measured_savings_vs_" + base_lv] = delta
        r["router_estimated_savings"] = e
        print(f"{lv:6s} {r['n_prompts']:>8,d} {r['total_tokens']:>13,d} "
              f"{r['mean_tokens']:>7.1f} {r['p95_tokens']:>6d} {delta:>+8.2%} "
              f"{('—' if e is None else f'{e:+11.2%}')}")

    # ★ the comparison that matters: does the estimate match reality?
    disagree = [(lv, rows[lv]['measured_savings_vs_' + base_lv],
                 rows[lv]['router_estimated_savings'])
                for lv in rows
                if rows[lv].get('router_estimated_savings') is not None
                and abs(rows[lv]['measured_savings_vs_' + base_lv]
                        - rows[lv]['router_estimated_savings']) > 0.05]
    print()
    if disagree:
        print("⚠️ MEASURED AND ESTIMATED SAVINGS DISAGREE BY >5 POINTS:")
        for lv, m, e in disagree:
            print(f"     {lv}: measured {m:+.2%}  ·  router estimated {e:+.2%}")
        print("   Report the MEASURED figure. ACTION_TOKEN_COST was never calibrated")
        print("   against a tokeniser, and it prices the router's hypothetical 'full'")
        print("   action rather than the prompts prompt_cfg_for() actually emits.")
    else:
        print("✅ measured and estimated savings agree within 5 points")

    # ★ the ladder's real question: what does GRANULARITY buy over the simplest policy?
    if "L1" in rows:
        l1 = rows["L1"]["total_tokens"]
        print(f"\n{'-'*78}")
        print("WHAT GRANULARITY BUYS OVER L1 (the fixed 'description only' policy)")
        print("-" * 78)
        for lv in ("L2", "L3", "L4"):
            if lv not in rows:
                continue
            gain = (l1 - rows[lv]["total_tokens"]) / b_total if b_total else 0.0
            verdict = ("worse than L1" if gain < 0 else
                       "negligible" if gain < 0.01 else "real")
            print(f"  {lv}: {gain:+.2%} of baseline   ({verdict})")
        print("\n  ★ If these are all near zero, the ladder's savings come from")
        print("    L0 -> L1 — a FIXED policy — and not from routing granularity.")
        print("    That is a finding, and the chapter should say it plainly.")

    out = {"dataset": ns.dataset, "split": ns.split, "tokenizer": base,
           "baseline_level": base_lv, "levels": rows}
    dest = Path(ns.json or Path(cfg["output"]["results_dir"],
                                f"ch3_tokens_{ns.dataset}_{ns.split}.json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
