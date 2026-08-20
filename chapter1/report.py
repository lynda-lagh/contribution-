"""
CHAPTER 1 — the full evaluation report.

    python -m chapter1.report --dataset WN11

Accuracy alone hides everything that matters here. A model that always answers
"Yes" scores 50% on a balanced set and looks like a coin; a model that memorises
scores 93% and looks like it understands. This module produces the numbers that
tell those apart.

  1. classification   accuracy · precision/recall/F1 per class · confusion matrix
  2. degenerate check is it just answering the same thing every time?
  3. per-relation     which relations carry the score
  4. seen / unseen    the familiarity gap
  5. calibration      ECE · Brier · reliability
  6. abstention       risk-coverage
  7. significance     McNemar vs a named baseline
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# =============================================================================
def classification(pred: np.ndarray, label: np.ndarray) -> dict:
    """Per-class precision/recall/F1 + the confusion matrix, on ±1 labels."""
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == -1)).sum())
    tn = int(((pred == -1) & (label == -1)).sum())
    fn = int(((pred == -1) & (label == 1)).sum())
    n = tp + fp + tn + fn

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if (t + f_p) else 0.0
        r = t / (t + f_n) if (t + f_n) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": p, "recall": r, "f1": f}

    pos, neg = prf(tp, fp, fn), prf(tn, fn, fp)
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "positive_class": pos, "negative_class": neg,
        "macro_f1": (pos["f1"] + neg["f1"]) / 2,
        # ⚠️ balanced accuracy is unaffected by class skew; if it differs a lot
        # from plain accuracy, the test set is not balanced and accuracy misleads
        "balanced_accuracy": (pos["recall"] + neg["recall"]) / 2,
    }


def degenerate_check(pred: np.ndarray, label: np.ndarray) -> dict:
    """
    ★ THE CHECK THAT CATCHES A MEANINGLESS NUMBER.

    A constant answer is invisible in accuracy but obvious here. It is exactly
    what a one-class test set produces -- the failure mode suspected for
    YAGO3-10, whose test.tsv has no ±1 labels so every gold answer became "No".
    """
    pr = float((pred == 1).mean())
    lr = float((label == 1).mean())
    always = bool(pr < 0.02 or pr > 0.98)
    majority = max(lr, 1 - lr)
    acc = float((pred == label).mean())
    return {
        "predicted_yes_rate": pr,
        "actual_yes_rate": lr,
        "always_same_answer": always,
        "majority_class_baseline": majority,
        "beats_majority_by": acc - majority,
        "verdict": (
            f"⚠️ DEGENERATE — the model answered {'Yes' if pr > .5 else 'No'} on "
            f"{max(pr, 1-pr):.1%} of items. Accuracy here measures the class balance, "
            f"not the model."
            if always else
            f"⚠️ does not beat the majority-class baseline ({majority:.4f})"
            if acc <= majority + 0.01 else
            f"ok — predicts Yes {pr:.1%}, beats majority baseline by {acc - majority:+.4f}"
        ),
    }


def per_relation(pred: np.ndarray, label: np.ndarray, rels: list[str],
                 min_n: int = 20) -> dict:
    """
    Which relations carry the score?

    Relevant because our sample is imbalanced: EIR 28.96 on WN11 — one relation is
    3,157 of 10,000 sampled triples, the rarest is 109. A headline number can be
    one frequent relation solved and ten rare ones failed.
    """
    by = defaultdict(lambda: {"n": 0, "correct": 0})
    for p, l, r in zip(pred, label, rels):
        by[r]["n"] += 1
        by[r]["correct"] += int(p == l)
    out = {r: {"n": v["n"], "accuracy": v["correct"] / v["n"]}
           for r, v in by.items() if v["n"] >= min_n}
    if not out:
        return {"relations": {}, "note": f"no relation has >= {min_n} test items"}
    accs = [v["accuracy"] for v in out.values()]
    worst = min(out.items(), key=lambda kv: kv[1]["accuracy"])
    best = max(out.items(), key=lambda kv: kv[1]["accuracy"])
    return {
        "relations": dict(sorted(out.items(), key=lambda kv: kv[1]["accuracy"])),
        "spread": max(accs) - min(accs),
        "worst": {"relation": worst[0], **worst[1]},
        "best": {"relation": best[0], **best[1]},
        "note": ("a large spread means the headline number is carried by a few "
                 "relations — report the per-relation table, not just the mean"),
    }


# =============================================================================
def full_report(conf: list[float], label: list[int], records: list[dict],
                baseline_correct: list[bool] | None = None) -> dict:
    """
    `conf` = P(Yes) per item (the logit parser's output).
    `label` = +1 / -1.
    """
    p = np.clip(np.asarray(conf, float), 0, 1)
    y = np.asarray(label, int)
    pred = np.where(p > 0.5, 1, -1)
    correct = (pred == y)

    rep: dict = {"n": int(len(y))}
    rep["classification"] = classification(pred, y)
    rep["degenerate"] = degenerate_check(pred, y)

    rels = [r.get("relation") for r in records] if records else []
    if rels and all(rels):
        rep["per_relation"] = per_relation(pred, y, rels)

    if records and ("seen_head" in records[0] or "seen_both" in records[0]):
        from .analysis import calibration_by_familiarity, seen_unseen
        rep["seen_unseen"] = seen_unseen(records, correct.tolist())
        rep["calibration"] = calibration_by_familiarity(
            [max(c, 1 - c) for c in p], correct.tolist(), records)

    # ---- calibration + abstention on the model's own confidence -------------
    try:
        from src.eval.abstention import risk_coverage_curve
        from src.eval.calibration import (brier_score, expected_calibration_error,
                                          reliability_table)
        margin = np.abs(p - 0.5) * 2          # 0 = coin flip, 1 = certain
        rep["confidence"] = {
            "ECE": float(expected_calibration_error(np.maximum(p, 1 - p), correct)),
            "Brier": float(brier_score(np.maximum(p, 1 - p), correct)),
            "reliability": reliability_table(np.maximum(p, 1 - p), correct),
            "mean_margin": float(margin.mean()),
        }
        rc = risk_coverage_curve(margin, correct)
        rep["abstention"] = {k: v for k, v in rc.items() if k != "curve"}
        rep["abstention"]["curve_head"] = rc.get("curve", [])[:10]
    except Exception as e:      # never let a reporting extra kill the run
        rep["confidence_error"] = f"{type(e).__name__}: {e}"

    # ---- significance vs a named baseline -----------------------------------
    if baseline_correct is not None and len(baseline_correct) == len(correct):
        from src.eval.significance import mcnemar
        rep["vs_baseline"] = mcnemar(correct, np.asarray(baseline_correct, bool))

    return rep


# =============================================================================
def print_report(name: str, rep: dict) -> None:
    c, d = rep["classification"], rep["degenerate"]
    cm = c["confusion"]
    print(f"\n{'=' * 68}\n{name}   (n={rep['n']})\n{'=' * 68}")

    print(f"  accuracy          {c['accuracy']:.4f}")
    print(f"  balanced accuracy {c['balanced_accuracy']:.4f}")
    print(f"  macro F1          {c['macro_f1']:.4f}")
    print(f"\n  confusion      predicted Yes   predicted No")
    print(f"    actual Yes   {cm['TP']:>13d} {cm['FN']:>14d}")
    print(f"    actual No    {cm['FP']:>13d} {cm['TN']:>14d}")
    print(f"\n  {'':10s} {'precision':>10s} {'recall':>8s} {'F1':>8s}")
    for k in ("positive_class", "negative_class"):
        v = c[k]
        print(f"  {k[:8]:10s} {v['precision']:10.4f} {v['recall']:8.4f} {v['f1']:8.4f}")

    print(f"\n  {d['verdict']}")

    if "per_relation" in rep and rep["per_relation"].get("relations"):
        pr = rep["per_relation"]
        print(f"\n  per relation (spread {pr['spread']:.3f})")
        for r, v in list(pr["relations"].items())[:5]:
            print(f"    {r[:34]:34s} n={v['n']:5d}  {v['accuracy']:.4f}")
        if len(pr["relations"]) > 5:
            print(f"    … {len(pr['relations']) - 5} more")

    if "seen_unseen" in rep:
        su = rep["seen_unseen"]
        print("\n  seen / unseen entities")
        for k in ("both_seen", "one_seen", "neither"):
            b = su[k]
            a = f"{b['accuracy']:.4f}" if b["accuracy"] is not None else "   —"
            print(f"    {k:11s} n={b['n']:5d} ({b['share']:5.1%})  acc={a}")
        if su.get("familiarity_gap") is not None:
            print(f"    familiarity gap {su['familiarity_gap']:+.4f}")
        print(f"    {su['verdict']}")

    if "confidence" in rep:
        cf = rep["confidence"]
        print(f"\n  ECE {cf['ECE']:.4f} · Brier {cf['Brier']:.4f} · "
              f"mean margin {cf['mean_margin']:.4f}")
    if "abstention" in rep and rep["abstention"].get("coverage_at_90"):
        print(f"  coverage at 90% precision: {rep['abstention']['coverage_at_90']}")
    if "calibration" in rep and rep["calibration"].get("verdict"):
        print(f"  {rep['calibration']['verdict']}")
    if "vs_baseline" in rep:
        v = rep["vs_baseline"]
        print(f"\n  McNemar vs baseline: p={v.get('p_value')} "
              f"({'significant' if v.get('p_value', 1) < 0.05 else 'not significant'})")


def _blocks(res: Path, dataset: str, cond: str | None) -> dict[str, dict]:
    """{condition: eval json}, optionally filtered to one condition."""
    out = {}
    for f in sorted(res.glob(f"ch1-{dataset}-*-eval.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        cid = d.get("condition", f.stem)
        if cond is None or cid == cond:
            out[cid] = d
    return out


def main() -> None:
    """
    ★ THIS FUNCTION USED TO BE A STUB.

      It printed a four-column cond/real/anon/gap table and never called
      `full_report`, so per-relation accuracy, the confusion matrix, the
      degeneracy check, calibration and the risk-coverage curve — all defined
      above, all depended on by the paper's Threats and per-relation
      discussion — were dead code. It also took no --condition argument, so
      the notebook cell that loops over conditions failed five times with
      `unrecognized arguments: --condition A` and the loop looked like it had
      simply produced no output.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--dataset", default="WN11")
    ap.add_argument("--condition", nargs="+", default=None, metavar="ID",
                    help="one or more condition ids; default = all found")
    ap.add_argument("--side", default="real", choices=("real", "anon"),
                    help="which test set to report on")
    ap.add_argument("--baseline", default="A",
                    help="condition to McNemar against (skip with '')")
    ap.add_argument("--summary-only", action="store_true")
    ns = ap.parse_args()

    res = Path(ns.results)
    found = _blocks(res, ns.dataset, None)
    if not found:
        raise SystemExit(
            f"no ch1-{ns.dataset}-*-eval.json in {res}/ — "
            f"run chapter1.run --evaluate first")

    print("=" * 68)
    print(f"CHAPTER 1 — SUMMARY  ({ns.dataset})")
    print("=" * 68)
    print(f"{'cond':6s} {'real':>9s} {'anon':>9s} {'gap':>9s} {'chance':>8s}")
    for cid, d in sorted(found.items()):
        f_ = lambda v: f"{v:9.4f}" if isinstance(v, (int, float)) else "        —"
        ch = d.get("chance_level")
        print(f"{cid:6s} {f_(d.get('acc_real'))} {f_(d.get('acc_anon'))} "
              f"{f_(d.get('gap'))} {ch if ch is None else f'{ch:8.3f}'}")
    print("\n★ the GAP column is the contribution — a single accuracy cannot "
          "express it")
    if ns.summary_only:
        return

    want = ns.condition or sorted(found)

    # baseline correctness vector, for McNemar. Must be the SAME rows.
    base = found.get(ns.baseline) if ns.baseline else None
    base_correct = (base or {}).get(f"_{ns.side}", {}).get("correct")

    for cid in want:
        d = found.get(cid)
        if d is None:
            print(f"\n[{cid}] no eval file — skipped")
            continue
        blk = d.get(f"_{ns.side}") or {}
        p, y = blk.get("p_yes"), blk.get("labels")
        if not p or not y:
            print(f"\n[{cid}] this eval file predates the full-array change — "
                  f"re-run `chapter1.run --evaluate --condition {cid}` "
                  f"(inference only) to get the detailed report")
            continue
        bc = base_correct if (cid != ns.baseline and base_correct
                              and len(base_correct) == len(y)) else None
        rep = full_report(p, y, blk.get("records") or [], baseline_correct=bc)
        label = f"{ns.dataset} · condition {cid} · {ns.side} test set"
        if bc:
            label += f"   [McNemar vs {ns.baseline}]"
        print_report(label, rep)


if __name__ == "__main__":
    main()
