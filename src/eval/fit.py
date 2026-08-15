"""
Over- and underfitting, decided from the learning curve rather than by eye.

Deliberately free of torch/transformers so the test suite can import it on any
machine in milliseconds. `src/train/sft.py` re-exports it.

★ WHY THIS MATTERS MORE HERE THAN USUAL
This thesis CLAIMS the model memorises. If our training setup itself overfits --
20,000 instances, 2 epochs, lr 3e-4, a fixed two-word target -- then "the model
memorises" becomes a statement about OUR hyper-parameters rather than about KGC.
The claim is only safe if the curve is clean, so the curve must be reported.
"""
from __future__ import annotations


def fit_diagnosis(log_history: list[dict]) -> dict:
    """
    Read HF Trainer's `state.log_history` and classify the run.

      UNDERFIT   eval_loss still falling at the last eval
      OVERFIT    eval_loss rose after its minimum
      TRAIN/EVAL GAP  fits train far better than held-out data
      GOOD FIT   plateaued near the minimum
    """
    tr = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    ev = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

    if not ev:
        return {"verdict": "no eval logged — cannot diagnose fit",
                "final_train_loss": round(tr[-1][1], 5) if tr else None,
                "best_eval_loss": None, "best_step": None,
                "total_steps": tr[-1][0] if tr else 0,
                "train_curve": [(s, round(l, 5)) for s, l in tr], "eval_curve": []}

    best_step, best = min(ev, key=lambda x: x[1])
    last_step, last = ev[-1]
    final_train = tr[-1][1] if tr else None

    rise = (last - best) / max(best, 1e-9)
    still_falling = len(ev) >= 2 and (ev[-2][1] - last) / max(ev[-2][1], 1e-9) > 0.02
    gap = (final_train is not None) and (best - final_train) > 0.5 * max(final_train, 1e-9)

    if rise > 0.05:
        verdict = (f"⚠️ OVERFITTING — eval_loss rose {rise:.1%} after step {best_step}. "
                   f"load_best_model_at_end kept the step-{best_step} checkpoint, so the "
                   f"adapter on disk is the good one — but REPORT it: fewer epochs or a "
                   f"lower learning rate is the honest fix.")
    elif still_falling:
        verdict = ("⚠️ UNDERFITTING — eval_loss was still falling at the last eval. "
                   "More epochs or a higher learning rate would still help; this number "
                   "is a floor, not a ceiling.")
    elif gap:
        verdict = (f"⚠️ TRAIN/EVAL GAP — train {final_train:.4f} vs eval {best:.4f}. "
                   f"The model fits the training split far better than held-out data, "
                   f"which is exactly the memorisation this thesis measures. Expected — "
                   f"but say so explicitly rather than letting a reader find it.")
    else:
        verdict = "✅ GOOD FIT — eval_loss plateaued near its minimum"

    return {
        "verdict": verdict,
        "final_train_loss": round(final_train, 5) if final_train is not None else None,
        "best_eval_loss": round(best, 5),
        "last_eval_loss": round(last, 5),
        "best_step": best_step,
        "total_steps": last_step,
        "relative_rise_after_best": round(rise, 5),
        "train_curve": [(s, round(l, 5)) for s, l in tr],
        "eval_curve": [(s, round(l, 5)) for s, l in ev],
    }


def ascii_curve(diag: dict, width: int = 54, height: int = 9) -> str:
    """
    A learning curve you can read in a Kaggle log, with no matplotlib.
    `t` = train, `e` = eval, `*` = both.
    """
    tr, ev = diag.get("train_curve", []), diag.get("eval_curve", [])
    pts = tr + ev
    if not pts:
        return "(no curve)"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0 or y1 == y0:
        return "(curve too short to plot)"

    grid = [[" "] * width for _ in range(height)]
    def place(series, ch):
        for s, l in series:
            c = int((s - x0) / (x1 - x0) * (width - 1))
            r = int((1 - (l - y0) / (y1 - y0)) * (height - 1))
            grid[r][c] = "*" if grid[r][c] not in (" ", ch) else ch
    place(tr, "t")
    place(ev, "e")

    out = [f"  loss {y1:.4f} ┤" + "".join(grid[0])]
    out += ["            │" + "".join(r) for r in grid[1:-1]]
    out.append(f"       {y0:.4f} ┤" + "".join(grid[-1]))
    out.append("            └" + "─" * width)
    out.append(f"             step {x0}{' ' * max(0, width - 18)}{x1}")
    out.append("             t = train   e = eval   * = both")
    return "\n".join(out)
