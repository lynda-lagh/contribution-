"""
★★ OUR NUMBERS BESIDE THE PUBLISHED ONES — and the rules for when that is fair.

    python -m chapter1.compare --dataset WN11
    python -m chapter1.compare --dataset WN11 --extra      # F1, AUC, McNemar

KG-LLM's Table II lists 21 methods on WN11 and FB13. Putting our row in it is
the single clearest way to show the reproduction is faithful — but only for the
cells where the comparison is actually valid.

★ WHAT IS COMPARABLE
    WN11 triple classification.  Same dataset, same SHIPPED ±1 labels, same
    task, same metric.  Our A row belongs in their table.

✋ WHAT IS NOT, AND WHY
    FB13            we never ran it.
    YAGO3-10 class. our negatives are GENERATED, theirs are not. Different
                    test set -> different number. Never put it in their table.
    link prediction they rank against ALL 123,182 entities; we rank against
                    50. Their Hits@1 0.0782 and our 0.714 are not the same
                    quantity and must never share a column.
    model size      ours is 1.5B, theirs 7B/13B. Any gap is expected and is
                    the honest explanation, not a defect to hide.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# =============================================================================
#  KG-LLM, Table II — triple classification accuracy (%), verbatim.
#  Yao, Peng, Mao, Luo. "Exploring Large Language Models for Knowledge Graph
#  Completion." ICASSP 2025 / arXiv:2308.13916.
#  "The baseline results with citations are obtained from corresponding papers."
# =============================================================================
KGLLM_TABLE2: list[tuple[str, float | None, float | None]] = [
    ("NTN [7]",              86.2, 90.0),
    ("TransE [38]",          75.9, 81.5),
    ("TransH [38]",          78.8, 83.3),
    ("TransR [39]",          85.9, 82.5),
    ("TransD [40]",          86.4, 89.1),
    ("TEKE [10]",            86.1, 84.2),
    ("TransG [41]",          87.4, 87.3),
    ("TranSparse-S [42]",    86.4, 88.2),
    ("DistMult [43]",        87.1, 86.2),
    ("DistMult-HRS [43]",    88.9, 89.0),
    ("AATE [12]",            88.0, 87.2),
    ("ConvKB [27]",          87.6, 88.8),
    ("DOLORES [44]",         87.5, 89.3),
    ("DKRL (BERT)",          87.3, 79.8),
    ("KG-BERT(a) [13]",      93.5, 90.4),
    ("KGT5",                 72.8, 66.3),
    ("LLaMA-7B",             21.1,  9.1),
    ("LLaMA-13B",            28.1, 17.6),
    ("KG-LLaMA-7B",          95.5, 89.2),
    ("KG-LLaMA-13B",         95.6, 90.2),
    ("KG-LLaMA2-13B",        96.6, 90.7),
]

# ✋ Never place these in the same column as anything above.
NOT_COMPARABLE = {
    "YAGO3-10": "our YAGO3-10 negatives are GENERATED; KG-LLM's WN11/FB13 "
                "negatives ship with the benchmark. Different test set.",
    "FB13": "we never ran FB13.",
}


def our_rows(results: Path, dataset: str) -> list[tuple[str, float]]:
    """Our classification accuracies for `dataset`, one row per condition."""
    out = []
    for f in sorted(results.glob(f"ch1-{dataset}-*-eval.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("acc_real") is not None:
            out.append((d.get("condition", f.stem), 100 * d["acc_real"]))
    return out


# =============================================================================
#  EXTRA METRICS — things the current evaluation does not report
# =============================================================================
def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC. Ties contribute 0.5, as they should."""
    pairs = sorted(zip(scores, labels, strict=True))
    pos = sum(1 for _, l in pairs if l == 1)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float("nan")
    rank, i, total = 0.0, 0, 0.0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 1) / 2.0                       # 1-based average rank
        total += sum(avg for k in range(i, j) if pairs[k][1] == 1)
        i = j
    return (total - pos * (pos + 1) / 2) / (pos * neg)


def prf(pred: list[int], labels: list[int]) -> dict:
    """Precision / recall / F1 per class, plus macro-F1."""
    out = {}
    for cls, name in ((1, "positive"), (-1, "negative")):
        tp = sum(1 for p, l in zip(pred, labels, strict=True) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(pred, labels, strict=True) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(pred, labels, strict=True) if p != cls and l == cls)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[name] = {"precision": pr, "recall": rc,
                     "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0}
    out["macro_f1"] = (out["positive"]["f1"] + out["negative"]["f1"]) / 2
    return out


def mcnemar(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """
    ★ THE RIGHT TEST for two models on the SAME test set.

    Accuracy differences get compared by eye all the time; McNemar asks whether
    the DISAGREEMENTS are lopsided, which is the actual question, and it is
    exact so it needs no normal approximation and no repeated seeds.
    """
    from math import comb
    b = sum(1 for a, c in zip(correct_a, correct_b, strict=True) if a and not c)
    c = sum(1 for a, c in zip(correct_a, correct_b, strict=True) if c and not a)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0,
                "reading": "the two models are correct on exactly the same rows"}
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"b": b, "c": c, "p": p,
            "reading": (f"A right / B wrong on {b} rows, B right / A wrong on "
                        f"{c}; exact two-sided p = {p:.2g}"
                        + ("  → a real difference" if p < 0.05 else
                           "  → NOT distinguishable at this sample size"))}


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WN11")
    ap.add_argument("--results", default="results")
    ap.add_argument("--extra", action="store_true",
                    help="F1, AUC and McNemar from the saved probabilities")
    ap.add_argument("--out", default=None)
    ns = ap.parse_args()
    res = Path(ns.results)

    print("=" * 74)
    print(f"TRIPLE CLASSIFICATION — ours beside KG-LLM's Table II")
    print("=" * 74)

    if ns.dataset in NOT_COMPARABLE:
        print(f"\n✋ {ns.dataset} is NOT comparable to KG-LLM's table.")
        print(f"   {NOT_COMPARABLE[ns.dataset]}")
        print(f"   Printing our rows alone. Do NOT paste them into their table.\n")

    ours = our_rows(res, ns.dataset)
    if not ours:
        raise SystemExit(f"no ch1-{ns.dataset}-*-eval.json in {res}")

    lines = ["| Method | WN11 | FB13 |", "|---|---|---|"]
    if ns.dataset == "WN11":
        print(f"{'Method':<24}{'WN11':>8}{'FB13':>8}")
        print("-" * 74)
        for name, wn, fb in KGLLM_TABLE2:
            print(f"{name:<24}{wn:>8.1f}{fb:>8.1f}")
            lines.append(f"| {name} | {wn:.1f} | {fb:.1f} |")
        print("-" * 74)
        lines.append("| **ours (Qwen2.5-1.5B + LoRA)** | | |")

    for cond, acc in ours:
        tag = f"ours · condition {cond}"
        print(f"{tag:<24}{acc:>8.2f}{'—':>8}")
        lines.append(f"| {tag} | {acc:.2f} | — |")

    print("\nHOW TO READ THIS")
    print("  · Our condition A is KG-LLM's recipe at 1.5B instead of 7B/13B.")
    print("    A gap of a few points is the model size, and saying so is")
    print("    stronger than hiding it.")
    print("  · Every row below KG-BERT is a LANGUAGE MODEL row. LLaMA-7B")
    print("    scores 21.1 untuned; the whole table is about what tuning adds.")
    print("  · ✋ Our conditions B/S/C/D/E/G do NOT belong in this table. They")
    print("    are diagnostics on a modified input, not competing systems.")
    print("  · ✋ NEVER put a link-prediction number in this table: theirs is")
    print("    full-ranking over 123,182 entities, ours is 50-way.")

    if ns.extra:
        print("\n" + "=" * 74)
        print("EXTRA METRICS — from the saved probabilities, no GPU")
        print("=" * 74)
        blocks = {}
        for f in sorted(res.glob(f"ch1-{ns.dataset}-*-eval.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            s = d.get("samples_real")
            if not s:
                continue
            cond = d.get("condition", f.stem)
            p = [r["p_yes"] for r in s]
            y = [r["label"] for r in s]
            pred = [r["predicted"] for r in s]
            m = prf(pred, y)
            blocks[cond] = [r["correct"] for r in s]
            print(f"\n{cond}  (n={len(s)} sampled rows)")
            print(f"   AUC            {auc(p, y):.4f}   "
                  f"← separates positives from negatives at ANY threshold")
            print(f"   macro-F1       {m['macro_f1']:.4f}")
            print(f"   positive  P {m['positive']['precision']:.3f}  "
                  f"R {m['positive']['recall']:.3f}  F1 {m['positive']['f1']:.3f}")
            print(f"   negative  P {m['negative']['precision']:.3f}  "
                  f"R {m['negative']['recall']:.3f}  F1 {m['negative']['f1']:.3f}")
        if len(blocks) >= 2:
            print("\nMcNEMAR — is the difference real, on the same rows?")
            conds = sorted(blocks)
            base = conds[0]
            for c in conds[1:]:
                r = mcnemar(blocks[base], blocks[c])
                print(f"   {base} vs {c}: {r['reading']}")
        if not blocks:
            print("\n  no samples_* blocks yet — re-run the evaluation with the "
                  "current chapter1/evaluate.py to capture them (no retraining).")

    out = Path(ns.out or res / f"ch1_vs_kgllm_{ns.dataset}.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
