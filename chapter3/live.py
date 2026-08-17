"""
★ LIVE FEEDBACK for the Chapter 3 notebook — bars, deltas, timings, state.

    from chapter3.live import step, bar, leaderboard, panel, run

WHY THIS EXISTS
---------------
A cell that runs for forty minutes and prints nothing is indistinguishable from
a cell that has hung, and a number printed with no reference point is
indistinguishable from noise. Every helper here answers one of two questions:

    "is it still moving?"        -> step(), run()   — elapsed time, live output
    "is it better than what?"    -> bar(), leaderboard()  — always vs a baseline
                                                            AND vs chance

⚠️ Every MRR bar is drawn from CHANCE, not from zero. A bar drawn from zero makes
   0.0945 look like "some performance"; drawn from the 0.0900 floor it correctly
   looks like nothing. That single choice is the difference between a figure that
   informs and one that flatters.
"""
from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# 50-way random ranking floor: (1/N) * sum_{k=1..N} 1/k
def chance_mrr(n_way: int = 50) -> float:
    return sum(1.0 / k for k in range(1, n_way + 1)) / n_way


CHANCE = chance_mrr(50)

_FILL, _EMPTY = "█", "·"          # █  ·


# ----------------------------------------------------------------- bars
def bar(value: float, lo: float = 0.0, hi: float = 1.0, width: int = 26) -> str:
    """A bar from `lo` to `hi`. Values at or below `lo` render empty, by design."""
    if hi <= lo:
        return _EMPTY * width
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    n = int(round(frac * width))
    return _FILL * n + _EMPTY * (width - n)


def delta(new: float, ref: float, digits: int = 4) -> str:
    d = new - ref
    mark = "▲" if d > 0 else ("▼" if d < 0 else "=")
    return f"{d:+.{digits}f} {mark}"


def pct_of_headroom(value: float, base: float, ceiling: float) -> str:
    """How much of the achievable gain was recovered? The honest efficiency view."""
    head = ceiling - base
    if head <= 1e-9:
        return "  n/a"
    return f"{(value - base) / head:>5.0%}"


# ------------------------------------------------------------- timing
class step:
    """
    Context manager that makes a long cell legible.

        with step("train shared model", est="35 min"):
            ...

    ⚠️ `failed` exists because a subprocess returning a non-zero code is NOT an
       exception. Without it a step whose command died still printed a tick, so
       the banner said "✔ done" directly under "✘ exit code 1". A progress
       display that reports success for a failed command is worse than none.
    """

    def __init__(self, name: str, est: str | None = None, width: int = 74):
        self.name, self.est, self.width = name, est, width
        self.failed = False

    def __enter__(self):
        self.t0 = time.time()
        print("─" * self.width)
        print(f"▶ {self.name}" + (f"   (expect ~{self.est})" if self.est else ""))
        print("─" * self.width, flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        m = (time.time() - self.t0) / 60
        if exc_type is None and not self.failed:
            print(f"✔ {self.name} — done in {m:.1f} min\n", flush=True)
        else:
            print(f"✘ {self.name} — FAILED after {m:.1f} min\n", flush=True)
        return False


def run(cmd: str | list, name: str | None = None, est: str | None = None,
        check: bool = False) -> int:
    """
    Run a command with its output streaming live, and time it.

    check=True raises SystemExit on failure. Use it wherever a later cell would
    otherwise run on data the failed command never produced.
    """
    shell = isinstance(cmd, str)
    label = name or (cmd if shell else " ".join(cmd))
    with step(label[:70], est) as s:
        print("$", cmd if shell else " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, shell=shell)
        if rc != 0:
            s.failed = True
            print(f"  ✘ exit code {rc}", flush=True)
    if rc != 0 and check:
        raise SystemExit(f"✋ '{label}' failed (exit {rc}) — stopping, because "
                         f"everything below depends on it")
    return rc


# --------------------------------------------------------- reading results
def load_cells(results: str | Path, dataset: str, direction: str = "tail",
               tag: str = "tuned") -> dict:
    """(policy, budget) -> result dict."""
    res = Path(results)
    out = {}
    rx = re.compile(rf"ch3_{re.escape(dataset)}_(.+)_B(\d+)_{direction}_{tag}\.json$")
    for f in sorted(glob.glob(str(res / f"ch3_{dataset}_*_B*_{direction}_{tag}.json"))):
        m = rx.search(Path(f).name)
        if m:
            try:
                out[(m.group(1), int(m.group(2)))] = json.loads(
                    Path(f).read_text(encoding="utf-8"))
            except Exception:
                pass
    return out


def leaderboard(results: str | Path, dataset: str, direction: str = "tail",
                tag: str = "tuned", budget: int | None = None) -> dict:
    """
    ★ The table to print after every evaluation.

    Sorted by MRR, bars drawn FROM CHANCE, and every row carries its delta
    against the S0 baseline so "did it improve?" needs no arithmetic.
    """
    R = load_cells(results, dataset, direction, tag)
    if not R:
        print("  (no results yet)")
        return {}

    buds = sorted({b for _, b in R}) if budget is None else [budget]
    for b in buds:
        rows = {p: d for (p, bb), d in R.items() if bb == b}
        if not rows:
            continue
        base = rows.get("S0_uniform", {}).get("ranking", {}).get("MRR")
        top = max(d["ranking"]["MRR"] for d in rows.values())
        hi = max(top * 1.05, CHANCE + 1e-6)

        print(f"\n  B = {b} tokens        chance MRR = {CHANCE:.4f}"
              f"        n = {len(rows)} policies")
        print(f"  {'policy':15s} {'MRR':>7s}  {'':26s}  {'vs S0':>12s}  {'tok':>6s}")
        for p, d in sorted(rows.items(), key=lambda kv: -kv[1]["ranking"]["MRR"]):
            m = d["ranking"]["MRR"]
            tok = d.get("cost", {}).get("mean_context_tokens", 0)
            dv = delta(m, base) if base is not None and p != "S0_uniform" else ""
            flag = ""
            if m <= CHANCE + 0.005:
                flag = "  ← AT CHANCE"
            elif p == "ORACLE":
                flag = "  ← ceiling"
            print(f"  {p:15s} {m:>7.4f}  {bar(m, CHANCE, hi)}  {dv:>12s}  "
                  f"{tok:>6.1f}{flag}")
    return R


def improvement_panel(results: str | Path, dataset: str, direction: str = "tail",
                      tag: str = "tuned") -> None:
    """★ The one question, answered per budget: did allocation beat S0 and R?"""
    R = load_cells(results, dataset, direction, tag)
    if not R:
        print("  (no results yet)")
        return
    print(f"\n{'='*76}\n  DID ALLOCATION HELP?   (vs uniform S0, and vs the random "
          f"control R)\n{'='*76}")
    print(f"  {'B':>5s}  {'S0':>8s} {'R':>8s} {'best':>8s} {'policy':14s} "
          f"{'vs S0':>10s} {'vs R':>10s}")
    for b in sorted({b for _, b in R}):
        rows = {p: d["ranking"]["MRR"] for (p, bb), d in R.items() if bb == b}
        s0, rr = rows.get("S0_uniform"), rows.get("R_random")
        others = {p: v for p, v in rows.items()
                  if p not in ("S0_uniform", "R_random", "ORACLE")}
        if not others:
            continue
        bp = max(others, key=lambda p: others[p])
        bm = others[bp]
        d0 = f"{bm-s0:+.4f}" if s0 is not None else "—"
        dr = f"{bm-rr:+.4f}" if rr is not None else "—"
        print(f"  {b:>5d}  {s0 if s0 else 0:>8.4f} {rr if rr else 0:>8.4f} "
              f"{bm:>8.4f} {bp:14s} {d0:>10s} {dr:>10s}")
    print("\n  ⚠️  Point estimates only. Run chapter3.report for the PAIRED")
    print("      bootstrap — a difference under ~0.03 MRR at n=300 is not")
    print("      distinguishable from noise without it.")


# --------------------------------------------------------------- state
def panel(dataset: str, results: str = "results", root: str = "data") -> None:
    """A compact 'where am I'板 that every phase ends with."""
    res, dat = Path(results), Path(root, dataset)

    def row(label, ok, detail=""):
        print(f"  {'✅' if ok else '⬜'}  {label:42s} {detail}")

    print(f"\n{'='*76}\n  STATE — {dataset}\n{'='*76}")
    row("splits fetched", (dat / "train.tsv").exists())
    row("valid split (proper filtering)", (dat / "valid.tsv").exists(),
        "" if (dat / "valid.tsv").exists() else "filtered = train ∪ test only")

    sp = res / f"ch3_specificity_{dataset}.json"
    d = json.loads(sp.read_text()) if sp.exists() else None
    row("S5 gate profiled", bool(d),
        f"{d['checks_passed']}/3 passed" if d else "S5 cannot be reported yet")

    g = dat / "relation_descriptions_gate.json"
    gd = json.loads(g.read_text()) if g.exists() else None
    row("relation descriptions gated", bool(gd),
        f"{gd['n_passed']}/{gd['n']} pass, {gd['n_rejected']} rejected" if gd else "")

    cand = list(dat.glob("candidates_*way_s*.json"))
    row("candidates FROZEN", bool(cand), f"{len(cand)} file(s)")

    built = sorted((dat / "built").glob("*_B*")) if (dat / "built").exists() else []
    row("prompts built", bool(built), f"{len(built)} cells")

    n_t = len(glob.glob(str(res / f"ch3_{dataset}_*_B*_tail_tuned.json")))
    n_u = len(glob.glob(str(res / f"ch3_{dataset}_*_B*_*_untuned.json")))
    n_h = len(glob.glob(str(res / f"ch3_{dataset}_*_B*_head_tuned.json")))
    n_r = len(glob.glob(str(res / f"ch3_{dataset}_*_B*_*_rel.json")))
    row("tail cells evaluated", n_t > 0, f"{n_t}")
    row("untuned rows", n_u > 0, f"{n_u}")
    row("head direction", n_h > 0, f"{n_h}")
    row("relation prediction (confusion matrix)", n_r > 0, f"{n_r}")
    print()


def _demo() -> None:
    print("bar() drawn FROM CHANCE, which is the whole point:\n")
    for name, v in [("A real names", 0.8169), ("S permuted", 0.2974),
                    ("B anonymised", 0.1336), ("C anon+types", 0.0945),
                    ("chance", CHANCE)]:
        flag = "  ← AT CHANCE" if v <= CHANCE + 0.005 else ""
        print(f"  {name:15s} {v:>7.4f}  {bar(v, CHANCE, 0.86)}"
              f"  {delta(v, 0.8169):>14s}{flag}")
    print("\n  the same numbers with a bar drawn from ZERO (misleading):\n")
    for name, v in [("C anon+types", 0.0945), ("chance", CHANCE)]:
        print(f"  {name:15s} {v:>7.4f}  {bar(v, 0.0, 0.86)}")
    print("\n  ← C looks like 'a little performance' instead of 'nothing'.")


if __name__ == "__main__":
    _demo()
