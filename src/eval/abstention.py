"""
ABSTENTION + RISK-COVERAGE -- verified 0 of 188 papers.

`abstain` / `abstention` / `selective prediction` / `risk-coverage`: zero
occurrences across the corpus. Meanwhile three papers independently show the field
PUNISHES abstention:

  * KG-LLM Table VI -- GPT-4 answers "I cannot verify specific personal information
    about individuals who are not public figures" on a TRUE-labelled triple and is
    scored WRONG, because the parser sees "not".
  * Knit Fig. 1 -- lists "I don't know." as a HALLUCINATION (knowledge deficiency).
  * TSP -- GPT-3.5 emits 3,403 triples to get 96 right; nothing lets it stop.

So the mechanism and the metric have to arrive together: an abstaining system
cannot be evaluated fairly by existing protocols.

The deliverable
---------------
    A RISK-COVERAGE CURVE for KGC:
        "at 80% coverage, the admitted triples are 95% precise"

No paper in the corpus reports this. The nearest design precedent is Trust-Aware's
Table VII (accuracy vs a manual review queue of 23 links, swept over a threshold)
-- on software artifacts, not KGC.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray,
                        n_points: int = 50) -> dict:
    """
    Sort by confidence, sweep the abstention threshold, and report at every
    coverage level:

        coverage  fraction answered
        risk      error rate among ANSWERED items
        precision 1 - risk

    AURC (area under the risk-coverage curve) summarises it: lower is better.
    A FLAT curve means the model does not know what it does not know -- which is
    itself a strong negative finding about selective prediction in LLM-based KGC.
    """
    conf = np.asarray(confidence, float)
    y = np.asarray(correct, bool)
    n = len(conf)
    order = np.argsort(-conf)                    # most confident first
    y_sorted = y[order]
    conf_sorted = conf[order]

    cum_correct = np.cumsum(y_sorted)
    ks = np.unique(np.linspace(1, n, min(n_points, n)).astype(int))

    pts = []
    for k in ks:
        pts.append({
            "k": int(k),
            "coverage": float(k / n),
            "threshold": float(conf_sorted[k - 1]),
            "precision": float(cum_correct[k - 1] / k),
            "risk": float(1 - cum_correct[k - 1] / k),
            "n_abstained": int(n - k),
        })

    cov = np.array([p["coverage"] for p in pts])
    risk = np.array([p["risk"] for p in pts])
    aurc = float(np.trapz(risk, cov) / (cov[-1] - cov[0])) if len(cov) > 1 else float(risk[0])

    full_risk = float(1 - y.mean())
    # a random-confidence baseline is flat at full_risk; AURC below it means signal
    return {
        "n": n,
        "points": pts,
        "aurc": aurc,
        "full_coverage_risk": full_risk,
        "aurc_vs_random": aurc - full_risk,
        "has_signal": aurc < full_risk - 0.01,
        "note": ("confidence carries signal" if aurc < full_risk - 0.01 else
                 "FLAT curve -- the model does not know what it does not know; "
                 "selective prediction is not available with this confidence source"),
    }


def coverage_at_precision(curve: dict, target_precision: float = 0.95,
                          min_coverage: float = 0.05) -> dict | None:
    """
    ★ The practitioner's number: how much can I answer while staying this precise?
    Returns the HIGHEST coverage meeting the target.

    `min_coverage` guards against a degenerate answer. With a weak model the only
    point reaching 95% precision may be k=1 ("answer one triple, be perfect"),
    which is true and useless. Below the floor we return None with a reason, so
    the report says "no operating point reaches this precision" instead of
    printing a meaningless 0.1% coverage.
    """
    ok = [p for p in curve["points"] if p["precision"] >= target_precision]
    if not ok:
        return None
    best = max(ok, key=lambda p: p["coverage"])
    if best["coverage"] < min_coverage:
        return {"target_precision": target_precision, "degenerate": True,
                "max_coverage_at_target": best["coverage"],
                "reason": (f"only {best['k']} of {curve['n']} items reach "
                           f"{target_precision:.0%} precision -- no usable operating "
                           f"point at this target"),
                **best}
    return {"target_precision": target_precision, "degenerate": False, **best}


def precision_at_coverage(curve: dict, target_coverage: float = 0.8) -> dict:
    return min(curve["points"], key=lambda p: abs(p["coverage"] - target_coverage))


def abstention_policy(confidence: np.ndarray, threshold: float) -> np.ndarray:
    """True = answer, False = abstain (route to the human queue)."""
    return np.asarray(confidence, float) >= threshold


def evaluate_abstention(confidence: np.ndarray, correct: np.ndarray,
                        threshold: float, explanations: list[str] | None = None) -> dict:
    """
    Score a concrete operating point, three-way instead of two-way.

    ⚠️ This is the protocol fix: `correct / incorrect / abstained` rather than
    forcing everything into correct/incorrect. Under the standard protocol an
    abstention is silently converted into a wrong answer -- see KG-LLM Table VI.
    """
    conf = np.asarray(confidence, float)
    y = np.asarray(correct, bool)
    answer = abstention_policy(conf, threshold)

    n = len(y)
    n_ans = int(answer.sum())
    n_abs = n - n_ans

    forced_acc = float(y.mean())                      # what the field reports
    sel_acc = float(y[answer].mean()) if n_ans else 0.0
    would_have_been_wrong = int((~y[~answer]).sum()) if n_abs else 0

    out = {
        "threshold": float(threshold),
        "n": n,
        "answered": n_ans, "abstained": n_abs,
        "coverage": n_ans / n,
        "abstention_rate": n_abs / n,
        "selective_accuracy": sel_acc,
        "forced_accuracy": forced_acc,
        "accuracy_gain": sel_acc - forced_acc,
        # of the items we declined, how many WOULD have been errors?
        "errors_avoided": would_have_been_wrong,
        "abstention_precision": (would_have_been_wrong / n_abs) if n_abs else None,
        # ★ the honesty penalty: what forced answering costs a truthful model
        "honesty_penalty_if_scored_as_wrong": forced_acc - (float(y[answer].sum()) / n),
    }
    if explanations is not None:
        out["queue_sample"] = [
            {"index": int(i), "confidence": float(conf[i]), "reason": explanations[i]}
            for i in np.where(~answer)[0][:20]
        ]
    return out


def full_report(confidence: np.ndarray, correct: np.ndarray,
                explanations: list[str] | None = None,
                targets: tuple[float, ...] = (0.90, 0.95, 0.99),
                out_path: str | None = None) -> dict:
    curve = risk_coverage_curve(confidence, correct)

    ops, unreachable = {}, {}
    for t in targets:
        pt = coverage_at_precision(curve, t)
        if pt is None:
            unreachable[f"precision_{t}"] = "no point reaches this precision at any coverage"
        elif pt.get("degenerate"):
            unreachable[f"precision_{t}"] = pt["reason"]
        else:
            ops[f"precision_{t}"] = {
                **evaluate_abstention(confidence, correct, pt["threshold"], explanations),
                "coverage_at_target": pt["coverage"],
            }

    out = {
        "risk_coverage": {k: v for k, v in curve.items() if k != "points"},
        "curve_points": curve["points"],
        "operating_points": ops,
        "unreachable_targets": unreachable,
        "precision_at_80_coverage": precision_at_coverage(curve, 0.8),
        "headline": None,
    }
    p95 = coverage_at_precision(curve, 0.95)
    if p95 and not p95.get("degenerate"):
        out["headline"] = (f"at {p95['coverage']:.0%} coverage, admitted triples are "
                           f"{p95['precision']:.1%} precise "
                           f"({p95['n_abstained']} routed to human review)")
    else:
        p80 = precision_at_coverage(curve, 0.8)
        out["headline"] = (f"95% precision unreachable; at 80% coverage precision is "
                           f"{p80['precision']:.1%} "
                           f"({p80['n_abstained']} routed to human review)")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))

    print(f"[abstention] AURC {curve['aurc']:.4f} vs random {curve['full_coverage_risk']:.4f} "
          f"| {curve['note']}")
    if out["headline"]:
        print(f"[abstention] {out['headline']}")
    return out
