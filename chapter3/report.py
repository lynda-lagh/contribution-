"""
★ The Chapter 3 tables. Reads results, computes nothing on a GPU.

    python -m chapter3.report --dataset WN18RR-ind
    python -m chapter3.report --dataset WN18RR-ind --direction head
    python -m chapter3.report --dataset WN18RR-ind --compare-untuned

SEVEN VIEWS
-----------
    1  the grid            policy x budget, MRR with CONFIDENCE INTERVALS
    2  the three anchors   ORACLE ceiling · S0 baseline · R control
                           ★ every verdict now backed by a PAIRED BOOTTRAP
    3  efficiency          MRR per 1,000 context tokens
    4  beyond ranking      calibration, and real F1 when --task relation was run
    5  per relation        because one relation can be a third of the test set
    6  ★ tuned vs untuned  does allocation pay WITHOUT fine-tuning?
    7  ★ both directions   tail and head side by side

THE ONE QUESTION
----------------
    at MATCHED cost, does the allocation rule change ranking quality?

Answered by exactly three rows: **S0** (uniform), **R** (random), and the best
policy. Everything else is detail.

★★ WHAT CHANGED, AND WHY IT MATTERS MORE THAN ANY OTHER FIX HERE
   The old version declared a result when two policies differed by 0.005 MRR.
   At n=300 the standard error is around 0.02, so **every one of those verdicts
   could flip on the seed**. Verdicts are now driven by the paired bootstrap in
   `stats.py`: a difference is a result when its 95% interval excludes zero, and
   otherwise it is reported as UNMEASURABLE — which is a different sentence
   from "a small gain", and the only honest one.

⚠️ If `S_best ≈ R`, the DECISIONS added nothing and only the budget mattered.
   That is a clean negative result, not a failure — and it is the specificity
   analogue of the field's "more context is not better".
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from .stats import (align, bootstrap_ci, fmt_ci, fmt_diff, min_detectable_effect,
                    paired_bootstrap, verdict)


def load_all(results: Path, dataset: str, direction: str, tag: str,
             task: str = "link") -> dict:
    """(policy, budget) -> result dict, for one (direction, tag, task) slice."""
    out = {}
    suffix = "" if task == "link" else "_rel"
    pat = str(results / f"ch3_{dataset}_*_B*_{direction}_{tag}{suffix}.json")
    rx = re.compile(rf"ch3_{re.escape(dataset)}_(.+)_B(\d+)_{direction}_{tag}"
                    rf"{suffix}\.json$")
    for f in sorted(glob.glob(pat)):
        m = rx.search(Path(f).name)
        if not m:
            continue
        out[(m.group(1), int(m.group(2)))] = json.loads(
            Path(f).read_text(encoding="utf-8"))
    # tolerate the pre-direction filenames from the first run
    if not out and direction == "tail":
        for f in sorted(glob.glob(str(results / f"ch3_{dataset}_*_B*.json"))):
            m = re.search(rf"ch3_{re.escape(dataset)}_(.+)_B(\d+)\.json$", Path(f).name)
            if m:
                out[(m.group(1), int(m.group(2)))] = json.loads(
                    Path(f).read_text(encoding="utf-8"))
    return out


def rows_of(d: dict) -> list:
    return d.get("rows", [])


def mrr_of(d: dict) -> float:
    return d["ranking"]["MRR"]


def ci_of(d: dict, seed: int = 0) -> dict:
    """Prefer the CI computed at eval time; recompute from rows if absent."""
    if d.get("MRR_ci"):
        return d["MRR_ci"]
    r = rows_of(d)
    return bootstrap_ci([1.0 / x["rank"] for x in r], seed=seed) if r else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="WN18RR-ind")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--results", default=None)
    ap.add_argument("--direction", default="tail", choices=("tail", "head"))
    ap.add_argument("--tag", default="tuned")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--compare-untuned", action="store_true",
                    help="★ view 6: does allocation pay without fine-tuning?")
    ap.add_argument("--both-directions", action="store_true", help="★ view 7")
    ns = ap.parse_args()

    from src.utils.config import load_config
    cfg = load_config(ns.config)
    res = Path(ns.results or cfg["output"]["results_dir"])

    R = load_all(res, ns.dataset, ns.direction, ns.tag)
    if not R:
        raise SystemExit(
            f"no results under {res} for {ns.dataset} ({ns.direction}/{ns.tag}).\n"
            f"  python -m chapter3.evaluate --dataset {ns.dataset} "
            f"--policy S0_uniform --budget 120 --direction {ns.direction}")

    pols = sorted({p for p, _ in R})
    buds = sorted({b for _, b in R})
    have_rows = all(rows_of(d) for d in R.values())
    print(f"loaded {len(R)} cells · {len(pols)} policies × {len(buds)} budgets "
          f"· {ns.direction} · {ns.tag}")
    if not have_rows:
        print("  ⚠️ some cells lack per-query rows — the PAIRED bootstrap needs them.")
        print("     Re-run those cells; without pairing nothing under ~0.05 MRR")
        print("     is measurable at these sample sizes.")

    fps = {d.get("candidate_fingerprint") for d in R.values()
           if d.get("candidate_fingerprint")}
    if len(fps) > 1:
        print(f"  ✗ CANDIDATE FINGERPRINTS DIFFER: {fps}")
        print("    These cells did NOT rank against the same negatives. The")
        print("    comparison is not matched — rebuild with chapter3.candidates.")
    elif fps:
        print(f"  ✓ all cells share candidate fingerprint {fps.pop()}")

    if not any(d.get("has_valid_split") for d in R.values()):
        print("  ⚠️ no valid split on this dataset — 'filtered' means train ∪ test.")

    # ---- 1 · the grid ------------------------------------------------------
    print(f"\n{'='*100}\n1 · THE GRID   ({ns.dataset}, {ns.direction}, "
          f"{R[next(iter(R))].get('protocol', '50-way filtered')})\n{'='*100}")
    print(f"{'policy':16s} " + "".join(f"{'B='+str(b):>21s}" for b in buds))
    for p in pols:
        line = f"{p:16s} "
        for b in buds:
            d = R.get((p, b))
            if d:
                c = ci_of(d)
                line += f"{fmt_ci(c) if c else f'{mrr_of(d):.4f}':>21s}"
            else:
                line += f"{'—':>21s}"
        print(line)
    print("\n  MRR [95% bootstrap CI]. Overlapping intervals do NOT prove")
    print("  equivalence — see the paired tests in view 2, which are stricter.")

    print(f"\n{'policy':16s} " + "".join(f"{'tok B='+str(b):>13s}" for b in buds))
    for p in pols:
        line = f"{p:16s} "
        for b in buds:
            d = R.get((p, b))
            line += (f"{d['cost']['mean_context_tokens']:>13.1f}" if d else f"{'—':>13s}")
        print(line)

    # ---- 2 · the anchors, with real statistics -----------------------------
    print(f"\n{'='*100}\n2 · ★★ THE THREE ANCHORS — the whole claim, with PAIRED "
          f"BOOTSTRAP\n{'='*100}")
    for b in buds:
        s0 = R.get(("S0_uniform", b))
        rr_ = R.get(("R_random", b))
        others = {p: R[(p, b)] for p in pols
                  if (p, b) in R and p not in ("S0_uniform", "R_random", "ORACLE")}
        if not others or not s0:
            continue
        bp = max(others, key=lambda p: mrr_of(others[p]))
        bd = others[bp]

        print(f"\n  ── B = {b} tokens " + "─" * 66)
        print(f"     S0 uniform   {fmt_ci(ci_of(s0))}")
        if rr_:
            print(f"     R  random    {fmt_ci(ci_of(rr_))}")
        print(f"     {bp:12s} {fmt_ci(ci_of(bd))}   ← best policy")

        if not (rows_of(bd) and rows_of(s0)):
            print("     ⚠️ per-query rows missing; cannot pair. Re-run these cells.")
            continue
        try:
            a, c, n = align(rows_of(bd), rows_of(s0))
            d0 = paired_bootstrap(a, c, n_boot=ns.n_boot)
            print(f"\n     {bp} − S0   {fmt_diff(d0)}   (n={n} paired)")
            print(f"       {verdict(d0, bp, 'S0')}")
        except ValueError as e:
            print(f"     ✗ cannot pair against S0: {e}")
            d0 = None

        dr = None
        if rr_ and rows_of(rr_):
            try:
                a, c, n = align(rows_of(bd), rows_of(rr_))
                dr = paired_bootstrap(a, c, n_boot=ns.n_boot)
                print(f"     {bp} − R    {fmt_diff(dr)}   (n={n} paired)")
                print(f"       {verdict(dr, bp, 'R')}")
            except ValueError as e:
                print(f"     ✗ cannot pair against R: {e}")

        # ★ the sentence that goes in the paper
        print(f"\n     ➤ ", end="")
        if d0 is None or dr is None:
            print("incomplete — need S0, R and the best policy, all with rows.")
        elif not d0["significant"] and not dr["significant"]:
            print("NO EFFECT at this budget. Allocation is indistinguishable "
                  "from both\n       uniform and random. Report as a null result "
                  "and give the CI — it\n       bounds how large an effect the "
                  "experiment could have ruled out.")
        elif d0["significant"] and dr["significant"] and d0["diff"] > 0 and dr["diff"] > 0:
            print("★★ ALLOCATION PAYS at matched cost — beats uniform AND random.\n"
                  "       This is the headline. Quote both intervals.")
        elif d0["significant"] and d0["diff"] > 0 and not dr["significant"]:
            print("⚠️ Beats uniform but NOT random. The gain is the action MIX,\n"
                  "       not the targeting. Report the mix as the finding and "
                  "drop the\n       specificity claim — a reviewer will run this "
                  "comparison themselves.")
        else:
            print("allocation does not beat uniform at this budget.")

        # power, when the answer is null
        if d0 is not None and not d0["significant"]:
            mde = 2.8 * d0["sem_diff"]
            print(f"       power: this design could detect ≈{mde:.4f} MRR. "
                  f"A true effect\n       smaller than that would not have shown "
                  f"up — say so rather than\n       claiming equivalence.")

    # ---- ORACLE ------------------------------------------------------------
    orc = {b: R[("ORACLE", b)] for b in buds if ("ORACLE", b) in R}
    if orc:
        print(f"\n{'='*100}\n★ ORACLE CEILING (uses gold; not a method)\n{'='*100}")
        for b, d in sorted(orc.items()):
            s0 = R.get(("S0_uniform", b))
            if not s0:
                continue
            gap = mrr_of(d) - mrr_of(s0)
            sig = ""
            if rows_of(d) and rows_of(s0):
                try:
                    a, c, _ = align(rows_of(d), rows_of(s0))
                    t = paired_bootstrap(a, c, n_boot=ns.n_boot)
                    sig = f"   {fmt_diff(t)}"
                    if not t["significant"]:
                        sig += "\n      ✋ ORACLE is NOT distinguishable from S0 — " \
                               "NO allocation can win at\n         this budget. " \
                               "Do not run the ladder here."
                except ValueError:
                    pass
            print(f"  B={b:<4d} MRR {mrr_of(d):.4f}   headroom over S0 {gap:+.4f}{sig}")

        print(f"\n  ★ % of the ORACLE's headroom recovered:")
        for b in sorted(buds):
            if b not in orc or ("S0_uniform", b) not in R:
                continue
            s0m = mrr_of(R[("S0_uniform", b)])
            head = mrr_of(orc[b]) - s0m
            if head <= 0:
                continue
            for p in pols:
                if p in ("ORACLE", "S0_uniform") or (p, b) not in R:
                    continue
                got = mrr_of(R[(p, b)]) - s0m
                print(f"    B={b:<4d} {p:14s} {got/head:>7.1%}")

    # ---- 3 · efficiency ----------------------------------------------------
    print(f"\n{'='*100}\n3 · EFFICIENCY — MRR per 1,000 context tokens\n{'='*100}")
    print(f"{'policy':16s} " + "".join(f"{'B='+str(b):>13s}" for b in buds))
    for p in pols:
        line = f"{p:16s} "
        for b in buds:
            d = R.get((p, b))
            v = d["cost"]["MRR_per_1k_tokens"] if d else None
            line += f"{v:>13.4f}" if v else f"{'—':>13s}"
        print(line)
    print("\n  ★ This column is what makes the chapter about ALLOCATION rather")
    print("    than another ranking table: a policy only means something")
    print("    relative to what it spent.")

    # ---- 4 · beyond ranking ------------------------------------------------
    print(f"\n{'='*100}\n4 · BEYOND RANKING\n{'='*100}")
    print(f"{'policy':16s} {'B':>5s} {'H@1':>8s} {'macroH@1':>10s} "
          f"{'ECE':>8s} {'Brier':>8s}")
    for (p, b), d in sorted(R.items()):
        cal = d.get("calibration", {})
        h1 = d.get("per_relation_hits1", {})
        flag = "  ⚠️ degenerate" if d.get("degenerate") else ""
        print(f"{p:16s} {b:>5d} {d['ranking']['hits@1']:>8.4f} "
              f"{h1.get('macro_hits@1', 0):>10.4f} "
              f"{cal.get('ECE', 0):>8.4f} {cal.get('Brier', 0):>8.4f}{flag}")

    # ★ the real F1 — only exists if --task relation was run
    REL = load_all(res, ns.dataset, ns.direction, ns.tag, task="relation")
    if REL:
        print(f"\n{'-'*100}\n★★ RELATION PREDICTION — the genuine classification "
              f"view\n{'-'*100}")
        print(f"{'policy':16s} {'B':>5s} {'acc':>8s} {'macroF1':>9s} "
              f"{'microF1':>9s}  most-confused")
        for (p, b), d in sorted(REL.items()):
            c = d.get("confusion", {})
            tc = c.get("top_confusion") or {}
            conf = (f"{tc.get('gold','')[:18]} → {tc.get('predicted','')[:18]} "
                    f"({tc.get('n','')}x)" if tc else "—")
            print(f"{p:16s} {b:>5d} {c.get('accuracy',0):>8.4f} "
                  f"{c.get('macro_f1',0):>9.4f} {c.get('micro_f1',0):>9.4f}  {conf}")

        # print one full matrix — the smallest budget's best policy
        key = max(REL, key=lambda k: REL[k].get("confusion", {}).get("macro_f1", 0))
        c = REL[key]["confusion"]
        labs = c["labels"]
        if len(labs) <= 15:
            print(f"\n  CONFUSION MATRIX — {key[0]} at B={key[1]}  (rows gold, "
                  f"cols predicted)")
            w = min(16, max(len(l) for l in labs) + 1)
            print("  " + " " * w + "".join(f"{l[:6]:>7s}" for l in labs))
            for i, l in enumerate(labs):
                print(f"  {l[:w-1]:{w}s}" +
                      "".join(f"{c['matrix'][i][j]:>7d}" for j in range(len(labs))))
            print("\n  ★ Off-diagonal mass is the interesting part: inverse-relation")
            print("    confusions are directional errors that MORE tokens will not")
            print("    fix but the RIGHT tokens might — which is the argument that")
            print("    allocation is doing something semantic.")
        else:
            print(f"\n  ({len(labs)} classes — full matrix in the json)")
    else:
        print(f"\n  ⚠️ No relation-prediction results, so no real F1 and no")
        print(f"     confusion matrix. On the link task, per-relation precision,")
        print(f"     recall and F1 are all algebraically equal to Hits@1.")
        print(f"     python -m chapter3.evaluate --dataset {ns.dataset} "
              f"--policy S0_uniform --budget 120 --task relation")

    # ---- 5 · per relation --------------------------------------------------
    print(f"\n{'='*100}\n5 · PER RELATION — one relation can be a third of the test "
          f"set\n{'='*100}")
    best_b = buds[len(buds) // 2]
    shown = [(p, b) for (p, b) in sorted(R) if b == best_b][:3]
    if shown:
        rels = sorted({r for p, b in shown for r in R[(p, b)]["per_relation"]},
                      key=lambda r: -max(R[(p, b)]["per_relation"].get(r, {}).get("n", 0)
                                         for p, b in shown))[:10]
        print(f"  at B={best_b}")
        print(f"  {'relation':30s} {'n':>5s} " + "".join(f"{p[:11]:>12s}" for p, _ in shown))
        for r in rels:
            n = max(R[(p, b)]["per_relation"].get(r, {}).get("n", 0) for p, b in shown)
            line = f"  {r[:30]:30s} {n:>5d} "
            for p, b in shown:
                mm = R[(p, b)]["per_relation"].get(r)
                line += f"{mm['MRR']:>12.4f}" if mm else f"{'—':>12s}"
            print(line)

    # ---- 6 · ★ tuned vs untuned -------------------------------------------
    if ns.compare_untuned:
        U = load_all(res, ns.dataset, ns.direction, "untuned")
        print(f"\n{'='*100}\n6 · ★★ DOES ALLOCATION PAY WITHOUT FINE-TUNING?"
              f"\n{'='*100}")
        if not U:
            print(f"  no untuned results. Same command, drop --adapter:")
            print(f"    python -m chapter3.evaluate --dataset {ns.dataset} "
                  f"--policy S0_uniform --budget 120 --direction {ns.direction}")
        else:
            print("  If allocation pays UNTUNED as well, the claim is about the")
            print("  CONTEXT, not about our training recipe — a much stronger and")
            print("  much cheaper-to-replicate result.\n")
            print(f"  {'policy':16s} {'B':>5s} {'untuned':>10s} {'tuned':>10s} "
                  f"{'Δ tuning':>10s}")
            for (p, b) in sorted(set(U) & set(R)):
                print(f"  {p:16s} {b:>5d} {mrr_of(U[(p,b)]):>10.4f} "
                      f"{mrr_of(R[(p,b)]):>10.4f} "
                      f"{mrr_of(R[(p,b)])-mrr_of(U[(p,b)]):>+10.4f}")
            for b in sorted({b for _, b in U}):
                s0, best = U.get(("S0_uniform", b)), None
                cands = {p: U[(p, b)] for (p, bb) in U if bb == b
                         for p in [p] if p not in ("S0_uniform", "ORACLE", "R_random")}
                if s0 and cands:
                    bp = max(cands, key=lambda p: mrr_of(cands[p]))
                    if rows_of(cands[bp]) and rows_of(s0):
                        try:
                            a, c, n = align(rows_of(cands[bp]), rows_of(s0))
                            t = paired_bootstrap(a, c, n_boot=ns.n_boot)
                            print(f"\n  UNTUNED B={b}: {bp} − S0 {fmt_diff(t)}")
                            print(f"    {verdict(t, bp + ' (untuned)', 'S0 (untuned)')}")
                        except ValueError:
                            pass

    # ---- 7 · ★ both directions --------------------------------------------
    if ns.both_directions:
        other = "head" if ns.direction == "tail" else "tail"
        O = load_all(res, ns.dataset, other, ns.tag)
        print(f"\n{'='*100}\n7 · ★ BOTH DIRECTIONS\n{'='*100}")
        if not O:
            print(f"  no {other}-direction results.")
            print(f"    python -m chapter3.data --dataset {ns.dataset} --all "
                  f"--budget 120 --direction both")
            print(f"    python -m chapter3.candidates --dataset {ns.dataset} "
                  f"--direction both")
        else:
            print(f"  {'policy':16s} {'B':>5s} {ns.direction:>10s} {other:>10s} "
                  f"{'mean':>10s}")
            for (p, b) in sorted(set(O) & set(R)):
                x, y = mrr_of(R[(p, b)]), mrr_of(O[(p, b)])
                print(f"  {p:16s} {b:>5d} {x:>10.4f} {y:>10.4f} {(x+y)/2:>10.4f}")
            print("\n  ★ Report the mean of both directions as the headline number,")
            print("    as CATS and RealKGC do. A single direction invites the")
            print("    assumption that the easier one was chosen.")

    dest = res / f"ch3_report_{ns.dataset}_{ns.direction}_{ns.tag}.json"
    dest.write_text(json.dumps(
        {"dataset": ns.dataset, "direction": ns.direction, "tag": ns.tag,
         "cells": {f"{p}_B{b}": {k: v for k, v in d.items() if k != "rows"}
                   for (p, b), d in R.items()}}, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
