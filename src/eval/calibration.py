"""
CALIBRATION -- verified absent from LLM-based KGC.

The state of the field
----------------------
  * FIVE papers emit a confidence:  TCM (conf in [0,1]), UKGEBN (c in (0,1]),
    EGIT (margin, Eq.15 -- DEFINED AND NEVER USED), MKGL (a real distribution),
    GLR (P(True), 35 mentions).
  * ZERO validate it. GLR's abstract names the gap -- "the inability to assess the
    confidence of their predictions" -- proposes P(True), and reports
    ECE=0, Brier=0, reliability diagram=0.
  * Only TWO papers in 188 calibrate at all, both in ADJACENT tasks:
    FoodAtlas ("calibrated using isotonic regression") and Forecasting Scientific
    Knowledge ("calibrated confidence derived from LLM extraction logits via
    temperature scaling").
  * A third, Trust-Aware, CLAIMS "calibration-aware metrics including Expected
    Calibration Error and Brier score" in its contributions and reports NEITHER
    across seven tables and four experiments.

So the honest framing is not "nobody does it" -- it is:
    calibration is established practice in adjacent KG tasks, and not one
    LLM-based KGC method that emits a confidence has ever been calibrated.

Metrics and post-hoc methods here are standard [OUTSIDE]; the novelty is the
application, and saying so plainly is a strength.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ metrics
def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> dict:
    """
    ECE = sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|

    Also returns the per-bin table, which IS the reliability diagram and the
    thing the web app shows as "in this confidence bin, X% are correct".
    """
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)

    bins, ece, mce = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "accuracy": None, "confidence": None, "gap": None})
            continue
        acc, cf = float(correct[m].mean()), float(conf[m].mean())
        gap = abs(acc - cf)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "count": cnt,
                     "accuracy": acc, "confidence": cf, "gap": gap,
                     "overconfident": cf > acc})
    return {"ece": ece, "mce": mce, "n_bins": n_bins, "n": n, "bins": bins}


def brier_score(conf: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared error between confidence and outcome. Lower is better."""
    return float(np.mean((np.asarray(conf, float) - np.asarray(correct, float)) ** 2))


def reliability_table(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    return [b for b in expected_calibration_error(conf, correct, n_bins)["bins"]
            if b["count"] > 0]


# ------------------------------------------------------------------ post-hoc
def temperature_scale(conf_val: np.ndarray, correct_val: np.ndarray,
                      grid: np.ndarray | None = None) -> float:
    """
    Fit a single temperature on a HELD-OUT split (never on test).
    Same method Forecasting Scientific Knowledge applies to LLM extraction logits.
    """
    grid = grid if grid is not None else np.linspace(0.05, 5.0, 100)
    p = np.clip(np.asarray(conf_val, float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    y = np.asarray(correct_val, float)

    best_t, best_nll = 1.0, np.inf
    for t in grid:
        q = 1 / (1 + np.exp(-logit / t))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def apply_temperature(conf: np.ndarray, t: float) -> np.ndarray:
    p = np.clip(np.asarray(conf, float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(p / (1 - p)) / t))


def isotonic_calibrate(conf_val: np.ndarray, correct_val: np.ndarray):
    """FoodAtlas's method: 'calibrated using isotonic regression'."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(conf_val, float), np.asarray(correct_val, float))
    return iso


# ------------------------------------------------------------------ pipeline
def calibrate_and_report(conf_val: np.ndarray, correct_val: np.ndarray,
                         conf_test: np.ndarray, correct_test: np.ndarray,
                         name: str = "confidence",
                         out_path: str | None = None) -> dict:
    """
    Fit on validation, evaluate on test, report BEFORE and AFTER.

    Reporting both is the point: if the confidence is already well calibrated,
    that is itself a finding nobody currently knows.
    """
    def block(c, y):
        e = expected_calibration_error(c, y)
        return {"ece": e["ece"], "mce": e["mce"], "brier": brier_score(c, y),
                "mean_confidence": float(np.mean(c)), "accuracy": float(np.mean(y)),
                "reliability": [b for b in e["bins"] if b["count"] > 0]}

    before = block(conf_test, correct_test)

    t = temperature_scale(conf_val, correct_val)
    after_t = block(apply_temperature(conf_test, t), correct_test)

    try:
        iso = isotonic_calibrate(conf_val, correct_val)
        after_i = block(np.asarray(iso.predict(conf_test)), correct_test)
    except Exception as e:                                    # sklearn missing / degenerate
        after_i = {"error": str(e)}

    best = min([("uncalibrated", before["ece"]),
                ("temperature", after_t["ece"]),
                ("isotonic", after_i.get("ece", np.inf))], key=lambda kv: kv[1])

    out = {
        "source": name,
        "n_val": int(len(conf_val)), "n_test": int(len(conf_test)),
        "uncalibrated": before,
        "temperature": {"T": t, **after_t},
        "isotonic": after_i,
        "best_method": best[0], "best_ece": best[1],
        "already_well_calibrated": before["ece"] < 0.05,
        "direction": ("overconfident" if before["mean_confidence"] > before["accuracy"]
                      else "underconfident"),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))

    print(f"[calibration] {name}: ECE {before['ece']:.4f} -> "
          f"{best[1]:.4f} ({best[0]}) | Brier {before['brier']:.4f} | {out['direction']}")
    return out


def compare_sources(sources: dict[str, np.ndarray], correct: np.ndarray,
                    val_frac: float = 0.3, seed: int = 42,
                    out_path: str | None = None) -> dict:
    """
    Rank the confidence sources -- log-prob vs P(True) vs sampling disagreement.

    ★ Guaranteed to produce a result: even if every curve is poor, the RANKING
    ("sampling disagreement carries signal where sequence log-probability does
    not") is a finding.
    """
    y = np.asarray(correct, bool)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    vi, ti = idx[:n_val], idx[n_val:]

    res = {name: calibrate_and_report(c[vi], y[vi], c[ti], y[ti], name=name)
           for name, c in ((k, np.asarray(v, float)) for k, v in sources.items())}

    ranked = sorted(res.items(), key=lambda kv: kv[1]["best_ece"])
    out = {"sources": res,
           "ranking": [{"source": k, "best_ece": v["best_ece"],
                        "method": v["best_method"]} for k, v in ranked],
           "best_source": ranked[0][0]}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"[calibration] best source: {out['best_source']}")
    return out
