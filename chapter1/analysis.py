"""
CHAPTER 1 — analysis. Three instruments, one claim.

    python -m chapter1.analysis                    # everything found in results/
    python -m chapter1.analysis --seen-unseen      # the free, no-GPU one

  1. THE GAP          accuracy on real test - accuracy on anonymised test
                      -> how much is carried by entity surface forms
  2. SEEN / UNSEEN    accuracy split by whether the entity was in the training
                      sample -> the SAME claim, INDEPENDENTLY, without touching
                      the data
  3. CALIBRATION      is a memoriser calibrated about what it memorised?
                      -> feeds Chapter 4, costs nothing extra

★ Instruments 1 and 2 are independent. Anonymisation destroys names; the
seen/unseen split leaves them intact and asks whether FAMILIARITY is doing the
work. Two instruments agreeing is much harder to dismiss than one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# =============================================================================
#  1. THE GAP
# =============================================================================
def gap_table(rows: list[dict]) -> dict:
    """
    rows: [{condition, acc_real, acc_anon}, ...]

    A large gap means surface forms carry the accuracy. The TRADE-OFF the thesis
    must show is that a model can give up a little acc_real for a much smaller
    gap -- and if no condition does, that is the finding.
    """
    out = []
    for r in rows:
        ar, aa = r.get("acc_real"), r.get("acc_anon")
        if ar is None or aa is None:
            continue
        out.append({**r, "gap": ar - aa,
                    "memorisation_share": (ar - aa) / (ar - 0.5) if ar > 0.5 else None})
    out.sort(key=lambda x: x["gap"])
    return {
        "rows": out,
        "best_generalisation": out[0]["condition"] if out else None,
        "note": ("memorisation_share = gap / (acc_real - chance). It answers "
                 "'what fraction of the above-chance performance is surface form?'"),
    }


def recovery(acc: float, acc_A: float, acc_B: float) -> float:
    """
    % of the memorisation gap recovered. Ceiling = acc_A - acc_B.
    Report THIS, not raw accuracy: 'recovers 12% of the gap' is informative,
    'accuracy rose to 0.588' is not.
    """
    denom = acc_A - acc_B
    return float((acc - acc_B) / denom) if denom > 1e-9 else float("nan")


# =============================================================================
#  2. SEEN / UNSEEN  — free, no GPU
# =============================================================================
def seen_unseen(test_records: list[dict], correct: list[bool]) -> dict:
    """
    ★ THE INDEPENDENT TEST.

    We trained on 10,000 of WN11's 112,581 triples, so only part of the entity
    set was ever seen. `data.build_condition` stamps each test row with
    seen_head / seen_tail / seen_both.

    If accuracy is high on seen-both and collapses on neither-seen, memorisation
    is confirmed WITHOUT anonymising anything.

    This is also the enrichment argument: enrichment means new entities, and the
    'neither' column is the only one that speaks to it.
    """
    c = np.asarray(correct, dtype=bool)

    # Derive `seen_both` rather than trusting the stamp: older builds do not carry
    # it, and a silently-absent key would make every bucket empty and the whole
    # instrument return None instead of failing.
    def sb(r):
        if "seen_both" in r:
            return bool(r["seen_both"])
        return bool(r.get("seen_head", False)) and bool(r.get("seen_tail", False))

    h = np.array([bool(r.get("seen_head", False)) for r in test_records])
    t = np.array([bool(r.get("seen_tail", False)) for r in test_records])
    buckets = {
        "both_seen": np.array([sb(r) for r in test_records]),
        "one_seen":  h ^ t,
        "neither":   (~h) & (~t),
    }
    out = {}
    for name, mask in buckets.items():
        n = int(mask.sum())
        out[name] = {"n": n, "share": n / max(1, len(c)),
                     "accuracy": float(c[mask].mean()) if n else None}

    b, ne = out["both_seen"]["accuracy"], out["neither"]["accuracy"]
    out["familiarity_gap"] = (b - ne) if (b is not None and ne is not None) else None
    out["verdict"] = (
        "MEMORISATION CONFIRMED by a second, independent instrument — accuracy "
        "tracks whether the entity was seen in training, with names left intact"
        if (out["familiarity_gap"] or 0) > 0.10 else
        "accuracy does NOT depend on having seen the entity — the anonymisation "
        "result must then come from something other than familiarity, which is "
        "itself worth reporting"
    )
    return out


# =============================================================================
#  3. CALIBRATION — is a memoriser calibrated?
# =============================================================================
def calibration_by_familiarity(conf: list[float], correct: list[bool],
                               test_records: list[dict], n_bins: int = 10) -> dict:
    """
    ★ The Chapter 4 question that only exists because of Chapter 1:
      if confidence tracks seen-vs-unseen, ABSTENTION IS A MEMORISATION DETECTOR.
    Nobody in 188 papers has asked this (abstention 0/188, calibration 2/188).
    """
    from src.eval.calibration import brier_score, expected_calibration_error

    c = np.asarray(correct, dtype=bool)
    p = np.clip(np.asarray(conf, dtype=float), 0, 1)
    seen = np.array([r.get("seen_both", False) for r in test_records])

    def block(mask):
        if mask.sum() < 20:
            return None
        return {"n": int(mask.sum()),
                "accuracy": float(c[mask].mean()),
                "mean_confidence": float(p[mask].mean()),
                "ECE": float(expected_calibration_error(p[mask], c[mask], n_bins)),
                "Brier": float(brier_score(p[mask], c[mask]))}

    overall, s, u = block(np.ones_like(seen, bool)), block(seen), block(~seen)
    out = {"overall": overall, "seen": s, "unseen": u}
    if s and u:
        out["confidence_drop_on_unseen"] = s["mean_confidence"] - u["mean_confidence"]
        out["accuracy_drop_on_unseen"] = s["accuracy"] - u["accuracy"]
        # honest calibration: confidence should fall as much as accuracy does
        out["tracks_familiarity"] = abs(out["confidence_drop_on_unseen"] -
                                        out["accuracy_drop_on_unseen"]) < 0.05
        out["verdict"] = (
            "confidence falls in step with accuracy on unseen entities -> the model "
            "KNOWS when it is guessing, and abstention will work"
            if out["tracks_familiarity"] else
            "confidence does NOT fall as much as accuracy -> the model is "
            "OVERCONFIDENT exactly where it has no memorised answer. That is the "
            "worst case for deployment and the strongest argument for abstention.")
    return out


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--dataset", default="WN11")
    ap.add_argument("--ch1-json", default=None,
                    help="an existing ch1_<DATASET>.json to analyse (no GPU)")
    ns = ap.parse_args()

    res = Path(ns.results)
    print("=" * 74)
    print("CHAPTER 1 ANALYSIS")
    print("=" * 74)

    # ---- the gap table ------------------------------------------------------
    rows = []
    for f in sorted(res.glob("ch1-*-eval.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"condition": d.get("condition", f.stem),
                     "acc_real": d.get("acc_real"), "acc_anon": d.get("acc_anon")})
    if rows:
        g = gap_table(rows)
        print(f"\n{'cond':6s} {'real':>8s} {'anon':>8s} {'gap':>8s} {'memo share':>11s}")
        print("-" * 74)
        for r in g["rows"]:
            ms = f"{r['memorisation_share']:.1%}" if r["memorisation_share"] else "  —"
            print(f"{r['condition']:6s} {r['acc_real']:8.4f} {r['acc_anon']:8.4f} "
                  f"{r['gap']:8.4f} {ms:>11s}")
        print(f"\n  smallest gap: {g['best_generalisation']}")
    else:
        print("\n  no ch1-*-eval.json yet — run chapter1.run first")

    # ---- seen/unseen from an existing ch1 json ------------------------------
    src = Path(ns.ch1_json) if ns.ch1_json else res / f"ch1_{ns.dataset}.json"
    built = Path(ns.data, ns.dataset, "built", "test_instructions.json")
    if src.exists() and built.exists():
        d = json.loads(src.read_text(encoding="utf-8"))
        recs = json.loads(built.read_text(encoding="utf-8"))
        for cond in ("untuned", "tuned", "tuned_anon"):
            blk = d.get(cond)
            if not blk or "confidences" not in blk:
                continue
            conf = blk["confidences"]
            n = min(len(conf), len(recs))
            if not recs[0].get("seen_both") is not None:
                print("\n  ⚠️ test_instructions.json has no seen/unseen stamps. "
                      "Rebuild with chapter1.data to add them.")
                break
            labels = [r["label"] for r in recs[:n]]
            correct = [(c > 0.5) == (l == 1) for c, l in zip(conf[:n], labels)]
            su = seen_unseen(recs[:n], correct)
            print(f"\n[{cond}] seen / unseen")
            for k in ("both_seen", "one_seen", "neither"):
                b = su[k]
                acc = f"{b['accuracy']:.4f}" if b["accuracy"] is not None else "   —"
                print(f"    {k:11s} n={b['n']:5d} ({b['share']:5.1%})  acc={acc}")
            if su["familiarity_gap"] is not None:
                print(f"    familiarity gap {su['familiarity_gap']:+.4f}")
            print(f"    {su['verdict']}")

            cal = calibration_by_familiarity(
                [max(c, 1 - c) for c in conf[:n]], correct, recs[:n])
            if cal.get("verdict"):
                print(f"    calibration: {cal['verdict']}")
    else:
        print(f"\n  seen/unseen skipped — need {src} and {built}")


if __name__ == "__main__":
    main()
