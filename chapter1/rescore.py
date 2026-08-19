"""
★★ RESCORE A FINISHED RUN — no GPU, no retraining.

    python -m chapter1.rescore --dataset YAGO3-10 --condition E
    python -m chapter1.rescore --dataset YAGO3-10 --all

WHY THIS EXISTS — THE CONDITION E PROBLEM
-----------------------------------------
Condition E trains with **n_negatives = 6**, so the training set is 1 positive to
6 negatives: **14% positive**. The test set is 50/50. The model learned the
TRAINING prior and answered "No" to essentially everything:

    acc 0.5010 · positive_rate 0.0010 · TPR 0.002 · TNR 1.000
    every p_yes in [0.08, 0.37] — never once above 0.5

Reported as accuracy, that is 0.501 and looks like "learned nothing at chance".
**That conclusion may be wrong**, and the stored probabilities can settle it.

    argmax at 0.5 is the WRONG OPERATING POINT for a model trained at 1:6.

A model can rank positives above negatives perfectly while putting every
probability below 0.5 — accuracy would be at chance and AUC would be 1.0.
Accuracy conflates *discrimination* with *calibration*; AUC measures only the
first, and is invariant to any monotonic reweighting of the prior.

So this module reports three things per run:

    AUC                     can it SEPARATE positives from negatives at all?
    best-threshold bal.acc  what it scores at its OWN best operating point
    argmax accuracy         what was originally reported

★ THE THREE OUTCOMES, all interpretable, written before looking:

    AUC ≈ 0.5              genuinely learned nothing. The collapse is the result.
    AUC >> 0.5, argmax ≈ 0.5   ★ it DID learn; the 1:6 prior miscalibrated it.
                               Report AUC and say the recipe collapses at this
                               ratio — a limitation of the recipe, not a finding
                               about negative count.
    AUC > 0.5 modestly     partial. Report both numbers and let them disagree.

⚠️ THE HONEST CAVEAT. The best threshold is chosen ON the test set, so
   best-threshold balanced accuracy is an OPTIMISTIC ceiling, not a deployable
   number. It answers "is the signal there?", never "here is our accuracy".
   AUC is the number to quote; the threshold figure is a diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# ------------------------------------------------------------------ metrics
def auc(scores: list[float], labels: list[int]) -> float:
    """
    Mann–Whitney AUC with proper tie handling, computed by rank.

    Ties matter here: a collapsed model produces many near-identical scores, and
    counting a tie as a win would inflate AUC exactly where we are most
    suspicious. Tied scores each contribute 0.5.
    """
    pairs = sorted(zip(scores, labels, strict=True))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                      # average rank, 1-based
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = [r for r, (_, l) in zip(ranks, pairs, strict=True) if l == 1]
    n_pos, n_neg = len(pos), n - len(pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (sum(pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_threshold(scores: list[float], labels: list[int]) -> dict:
    """Threshold maximising BALANCED accuracy (chance 0.5 whatever the skew)."""
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return {}
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    best = {"balanced_accuracy": 0.0, "threshold": 1.0, "TPR": 0.0, "TNR": 1.0}
    for idx, i in enumerate(order):
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        if idx + 1 < len(order) and scores[order[idx + 1]] == scores[i]:
            continue                                    # never split a tie
        tpr, tnr = tp / n_pos, 1 - fp / n_neg
        bal = (tpr + tnr) / 2
        if bal > best["balanced_accuracy"]:
            best = {"balanced_accuracy": bal, "threshold": scores[i],
                    "TPR": tpr, "TNR": tnr}
    return best


def degeneracy(pred_pos_rate: float, tpr: float, tnr: float) -> str | None:
    if min(tpr, tnr) < 0.10:
        side = "Yes" if pred_pos_rate > 0.5 else "No"
        return f"✋ COLLAPSED — answers '{side}' to {max(pred_pos_rate, 1-pred_pos_rate):.1%} of items"
    if min(tpr, tnr) < 0.25:
        return "⚠️ strongly skewed toward one class"
    return None


# --------------------------------------------------------------------- main
def rescore_one(res: dict, side: str, labels: list[int]) -> dict:
    r = res.get("_" + side)
    if not isinstance(r, dict) or "p_yes" not in r:
        return {}
    p = r["p_yes"]
    n = min(len(p), len(labels))
    p, lab = p[:n], labels[:n]

    # reproduce the ORIGINAL decision rule: p_yes >= p_no  <=>  p_yes >= 0.5
    pred = [1 if x >= 0.5 else -1 for x in p]
    tp = sum(1 for a, b in zip(pred, lab, strict=True) if a == 1 and b == 1)
    fn = sum(1 for a, b in zip(pred, lab, strict=True) if a == -1 and b == 1)
    tn = sum(1 for a, b in zip(pred, lab, strict=True) if a == -1 and b == -1)
    fp = sum(1 for a, b in zip(pred, lab, strict=True) if a == 1 and b == -1)
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    argmax_acc = (tp + tn) / n
    bal = (tpr + tnr) / 2

    a = auc(p, lab)
    bt = best_threshold(p, lab)
    return {
        "n": n,
        "argmax_accuracy": argmax_acc,
        "balanced_accuracy_at_0.5": bal,
        "TPR": tpr, "TNR": tnr,
        "predicted_positive_rate": sum(1 for x in pred if x == 1) / n,
        "AUC": a,
        "best_threshold": bt,
        "degenerate": degeneracy(sum(1 for x in pred if x == 1) / n, tpr, tnr),
        "score_range": [min(p), max(p)],
    }


def verdict(m: dict) -> str:
    a = m.get("AUC")
    if a != a:                                          # nan
        return "no labels of one class — AUC undefined"
    bal = m["balanced_accuracy_at_0.5"]
    if a < 0.55:
        return ("★ genuinely learned nothing — AUC at chance, so the collapse IS "
                "the result, not an artefact of the operating point")
    if bal < 0.55 and a >= 0.60:
        return (f"★★ IT DID LEARN. AUC {a:.3f} but argmax balanced accuracy "
                f"{bal:.3f} — the model separates the classes and the DECISION "
                f"THRESHOLD is wrong. Report AUC; describe the accuracy as a "
                f"calibration failure of the recipe, not an inability to learn.")
    if a >= 0.60:
        return f"discriminates (AUC {a:.3f}) and the threshold is roughly right"
    return f"weak but non-zero discrimination (AUC {a:.3f})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--root", default="data")
    ap.add_argument("--condition", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--results", default="results")
    ns = ap.parse_args()

    from src.data.loaders import load_kg
    kg = load_kg(ns.dataset, ns.root)
    labelled = any(t.label is not None for t in kg.test)
    if not labelled:
        raise SystemExit(
            f"{ns.dataset}'s test set carries no ±1 labels, so triple "
            f"classification was never defined on it. Nothing to rescore.")
    labels = [t.label for t in kg.test]

    res_dir = Path(ns.results)
    conds = ns.condition or (["A", "B", "C", "D", "E", "G", "S"] if ns.all else ["E"])

    print(f"{'='*86}")
    print(f"RESCORING {ns.dataset} — AUC separates DISCRIMINATION from CALIBRATION")
    print(f"{'='*86}")
    print(f"{'cond':5s} {'side':5s} {'argmax':>8s} {'bal@.5':>8s} {'AUC':>7s} "
          f"{'best bal':>9s} {'thresh':>8s}   note")

    out = {}
    for c in conds:
        f = res_dir / f"ch1-{ns.dataset}-{c}-eval.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        out[c] = {}
        for side in ("real", "anon"):
            m = rescore_one(d, side, labels)
            if not m:
                continue
            out[c][side] = m
            bt = m["best_threshold"]
            print(f"{c:5s} {side:5s} {m['argmax_accuracy']:>8.4f} "
                  f"{m['balanced_accuracy_at_0.5']:>8.4f} {m['AUC']:>7.4f} "
                  f"{bt.get('balanced_accuracy', 0):>9.4f} "
                  f"{bt.get('threshold', 0):>8.4f}   {m['degenerate'] or ''}")
        print()

    print(f"{'='*86}\nVERDICTS\n{'='*86}")
    for c, sides in out.items():
        for side, m in sides.items():
            if m.get("degenerate") or m["AUC"] < 0.6:
                print(f"  {c} ({side}): {verdict(m)}")
                print(f"      scores span [{m['score_range'][0]:.3f}, "
                      f"{m['score_range'][1]:.3f}] — a collapsed model still has a "
                      f"usable ORDER even when every value sits on one side of 0.5")

    print(f"\n  ⚠️ Best-threshold balanced accuracy is chosen ON the test set. It is")
    print(f"     an optimistic CEILING and a diagnostic, never a reportable accuracy.")
    print(f"     Quote AUC — it is threshold-free and prior-invariant.")

    dest = res_dir / f"ch1_rescore_{ns.dataset}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
