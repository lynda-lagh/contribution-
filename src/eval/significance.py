"""
Significance testing -- P04's protocol.

Only 16 of 188 papers report statistical significance, and they are overwhelmingly
the biomedical ones, not the KGC methods. This matters more than it sounds:

    ColKGC's entire contribution is +0.022 MRR.
    MKGL beats KICGPT by +0.003 MRR.

Neither could distinguish that from noise. With a fixed test subset evaluated by
every condition, we can -- and PAIRED tests give far more power per unit of compute
than unpaired ones, which is why the test subset is fixed and identical everywhere.

P04's protocol: two tests (parametric + non-parametric) with multiple-comparison
correction.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray, exact: bool = True) -> dict:
    """
    Paired test for BINARY outcomes on the same items -- the right test for
    "did A and B classify the same test triples differently?".

    Uses only the discordant pairs (where the two systems disagree), which is
    exactly what makes it more sensitive than comparing two accuracies.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    assert a.shape == b.shape, "paired test requires the SAME test items"

    n01 = int(np.sum(~a & b))       # A wrong, B right
    n10 = int(np.sum(a & ~b))       # A right, B wrong
    n = n01 + n10

    if n == 0:
        return {"test": "mcnemar", "n_discordant": 0, "p_value": 1.0,
                "statistic": 0.0, "note": "identical predictions"}

    if exact or n < 25:
        p = float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
        stat = float(min(n01, n10))
        name = "mcnemar_exact"
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / n          # continuity-corrected chi2
        p = float(stats.chi2.sf(stat, df=1))
        name = "mcnemar_chi2"

    return {"test": name, "n01": n01, "n10": n10, "n_discordant": n,
            "statistic": float(stat), "p_value": p,
            "favours": "B" if n01 > n10 else ("A" if n10 > n01 else "tie")}


def paired_scores(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """
    P04's pair for CONTINUOUS per-item scores: paired t-test (parametric) AND
    Wilcoxon signed-rank (non-parametric). Reporting both is the point -- if they
    disagree, the effect is fragile.
    """
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    assert a.shape == b.shape
    d = b - a
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:                                  # all differences zero
        w_stat, w_p = 0.0, 1.0
    sd = d.std(ddof=1)
    return {
        "mean_difference": float(d.mean()),
        "ttest_rel": {"statistic": float(t_stat), "p_value": float(t_p)},
        "wilcoxon": {"statistic": float(w_stat), "p_value": float(w_p)},
        "cohens_dz": float(d.mean() / sd) if sd > 0 else 0.0,
        "agree": (t_p < 0.05) == (w_p < 0.05),
    }


def bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Multiple-comparison correction -- P04 uses Bonferroni."""
    m = len(p_values)
    return {k: {"p_raw": p, "p_corrected": min(1.0, p * m),
                "significant": p * m < alpha}
            for k, p in p_values.items()} | {"_n_comparisons": m, "_alpha": alpha}


def compare_conditions(results: dict[str, np.ndarray], kind: str = "binary",
                       alpha: float = 0.05, out_path: str | None = None) -> dict:
    """
    results : {"lora": per_item_correct_or_score, "mora": ..., "boft": ...}
              Every array must be aligned to the SAME fixed test subset.

    Runs all pairwise comparisons, then corrects.
    """
    names = list(results)
    pairwise, raw_p = {}, {}
    for a, b in combinations(names, 2):
        key = f"{a}_vs_{b}"
        if kind == "binary":
            r = mcnemar(results[a], results[b])
            raw_p[key] = r["p_value"]
        else:
            r = paired_scores(results[a], results[b])
            raw_p[key] = max(r["ttest_rel"]["p_value"], r["wilcoxon"]["p_value"])
        pairwise[key] = r

    out = {
        "kind": kind,
        "conditions": names,
        "n_items": int(len(next(iter(results.values())))),
        "pairwise": pairwise,
        "corrected": bonferroni(raw_p, alpha),
        "protocol": "P04: two tests + Bonferroni; paired on a fixed test subset",
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2, default=float))
    return out


def seed_variance(accuracies: list[float]) -> dict:
    """
    Spread across seeds. Report this next to every headline number -- it is what
    tells a reader whether +0.022 (ColKGC) or +0.003 (MKGL) means anything.
    """
    a = np.asarray(accuracies, float)
    n = len(a)
    sem = a.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    ci = 1.96 * sem
    return {"n_seeds": n, "mean": float(a.mean()), "std": float(a.std(ddof=1)) if n > 1 else 0.0,
            "sem": float(sem), "ci95_halfwidth": float(ci),
            "min": float(a.min()), "max": float(a.max()),
            "note": f"a difference below {2*ci:.4f} is not distinguishable from seed noise"}
