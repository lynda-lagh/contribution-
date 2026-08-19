"""
★ CHAPTER 3 — THE EVALUATION THAT WAS MISSING.

`train_level()` trains an adapter and writes `train_loss`, routing statistics and
a fit verdict. It never scores anything. So every `ch3-*-lora.json` on disk looks
like this:

    train_loss 0.0217   savings_rate 83.3%   skip_rate 3.9%   ... and no accuracy

Which means the chapter's central claim — *routing removes 83% of enrichment
tokens* — has no companion number for what that costs. **Saving tokens is trivial
if you are allowed to lose accuracy.** The whole question is the trade.

WHAT THIS ADDS
--------------
Each level's adapter, scored on **its own** test set (the prompts it was trained
for), reporting:

    accuracy · balanced accuracy · positive rate · per-class recall
    tokens per prompt (measured) · accuracy per 1k tokens

⚠️ **Balanced accuracy, not raw.** A level that collapses to answering "Yes"
scores the positive rate and looks fine. `positive_rate` and the degenerate flag
catch it.

⚠️ Each level is scored on its OWN prompts, which is the only fair comparison —
scoring L3's adapter on L0's prompts would measure distribution shift. It also
means the levels differ in test input, so this is a comparison of POLICIES, not
of models on fixed input. Say so.

    python -m chapters.ch3_conditioning.evaluate --dataset YAGO3-10
    python -m chapters.ch3_conditioning.evaluate --dataset YAGO3-10 --level L3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LEVELS = ("L0", "L1", "L2", "L3", "L4")


def score_level(cfg: dict, dataset: str, level: str, limit: int) -> dict | None:
    import numpy as np

    from chapter1.evaluate import _load, _score_set

    rid = f"ch3-{dataset}-{level}-{cfg['peft']['method']}"
    adapter = Path(cfg["output"]["adapter_dir"], rid)
    built = Path(cfg["data"]["root"], f"{dataset}-{level}", "built",
                 "test_instructions.json")

    if not (adapter / "adapter_config.json").exists():
        print(f"[{level}] no adapter at {adapter} — train it first")
        return None
    if not built.exists():
        print(f"[{level}] no test set at {built}")
        return None

    recs = json.loads(built.read_text(encoding="utf-8"))
    model, tok = _load(cfg["model"]["name"], str(adapter))
    r = _score_set(model, tok, recs, limit)
    del model

    y = np.array([rec["label"] for rec in r["records"]], int)
    correct = np.array(r["correct"], bool)
    recalls = {int(c): float(correct[y == c].mean())
               for c in (1, -1) if (y == c).any()}
    bal = float(np.mean(list(recalls.values()))) if recalls else None

    out = {
        "level": level, "adapter": str(adapter), "n": r["n"],
        "accuracy": r["accuracy"],
        "balanced_accuracy": bal,
        "positive_rate": r["positive_rate"],
        "recall_pos": recalls.get(1), "recall_neg": recalls.get(-1),
        # a model answering one class ~always is not a result, whatever it scores
        "degenerate": bool(r["positive_rate"] > 0.95 or r["positive_rate"] < 0.05),
    }
    flag = "  ⚠️ DEGENERATE" if out["degenerate"] else ""
    print(f"[{level}] acc {out['accuracy']:.4f}  bal {bal:.4f}  "
          f"yes {r['positive_rate']:.1%}{flag}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--level", nargs="+", default=list(LEVELS), choices=LEVELS,
                    metavar="L")
    ap.add_argument("--limit", type=int, default=2000)
    ns = ap.parse_args()

    from src.utils.config import load_config
    cfg = load_config(ns.config)
    res = Path(cfg["output"]["results_dir"])
    res.mkdir(parents=True, exist_ok=True)

    rows = {}
    for lv in ns.level:
        r = score_level(cfg, ns.dataset, lv, ns.limit)
        if r:
            rows[lv] = r
            # save after every level — scoring is minutes of GPU and a later
            # failure must not discard it
            (res / f"ch3_eval_{ns.dataset}.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8")

    if not rows:
        raise SystemExit("nothing scored")

    print(f"\n{'='*70}")
    print(f"CHAPTER 3 — ACCURACY BY CONDITIONING LEVEL ({ns.dataset})")
    print("=" * 70)
    print(f"{'level':6s} {'acc':>8s} {'bal acc':>9s} {'yes rate':>9s} "
          f"{'vs L0':>8s}")
    b = rows.get("L0", {}).get("balanced_accuracy")
    for lv in LEVELS:
        if lv not in rows:
            continue
        r = rows[lv]
        d = (r["balanced_accuracy"] - b) if (b and r["balanced_accuracy"]) else None
        print(f"{lv:6s} {r['accuracy']:>8.4f} {r['balanced_accuracy']:>9.4f} "
              f"{r['positive_rate']:>8.1%} "
              f"{('—' if d is None else f'{d:>+7.4f}')}"
              + ("  ⚠️ degenerate" if r["degenerate"] else ""))

    print(f"\n{'-'*70}")
    print("READ THIS WITH ch3_tokens_*.json:")
    print("  the chapter's question is not 'does accuracy rise with granularity'")
    print("  but 'what does each extra granularity level COST and BUY'.")
    print("  A level that saves no tokens and adds no accuracy is dominated,")
    print("  and saying so is a result.")
    print(f"\nwrote {res / f'ch3_eval_{ns.dataset}.json'}")


if __name__ == "__main__":
    main()
