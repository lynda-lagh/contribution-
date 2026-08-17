"""
★★ THE HALF-PAGE THAT PERSUADES — the same query, the same budget, two policies.

    python -m chapter3.qualitative --dataset WN18RR-ind --budget 120 \
        --a S0_uniform --b S4_instance
    python -m chapter3.qualitative --dataset WN18RR-ind --budget 120 \
        --a S0_uniform --b S4_instance --only-disagreements --n 5
    python -m chapter3.qualitative ... --latex     # ★ paste straight into the paper

WHY THIS EXISTS
---------------
The grid in `report.py` proves the effect. It does not SHOW it. A reader who
sees `S4 = 0.412, S0 = 0.391` has to take on trust that something meaningful
happened; a reader who sees

    query   (whale, _hypernym, ?)          budget 120 tokens, both policies

    S0 spent   type_tag 4 · relation_description 22 · neighbours 94
    S4 spent   neighbours 118
                                            rank 7  ->  rank 1

understands the mechanism in one glance. Reviewers of the closest papers (CATS,
RealKGC, KICGPTv2) all include a case study for exactly this reason.

★ AND IT IS ALSO A CORRECTNESS CHECK. If the two allocations look identical, or
  the "reasons" are boilerplate, the ladder is not doing what the code claims —
  and you find out here, on three examples, rather than after a 4-hour sweep.

⚠️ Cases are selected by a stated RULE (largest rank improvement, or largest
   regression with `--worst`), never hand-picked. Say which rule in the caption.
   A hand-picked example is an anecdote; a rule-selected one is evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_cell(root: str, dataset: str, policy: str, budget: int,
              direction: str) -> tuple[list, list]:
    """(allocations, queries) for one cell, tolerating pre-direction layouts."""
    base = Path(root, dataset, "built")
    d = base / f"{policy}_B{budget}_{direction}"
    if not d.exists() and direction == "tail":
        d = base / f"{policy}_B{budget}"
    if not d.exists():
        raise SystemExit(
            f"✋ {d} not built.\n   python -m chapter3.data --dataset {dataset} "
            f"--policy {policy} --budget {budget} --direction {direction}")
    allocs = json.loads((d / "allocations.json").read_text(encoding="utf-8"))
    queries = json.loads((d / "queries.json").read_text(encoding="utf-8"))
    return allocs, queries


def load_ranks(results: Path, dataset: str, policy: str, budget: int,
               direction: str, tag: str) -> dict:
    """qid -> rank, when the cell has been evaluated. Empty if it has not."""
    p = results / f"ch3_{dataset}_{policy}_B{budget}_{direction}_{tag}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {r["qid"]: r["rank"] for r in d.get("rows", [])}


def spend_line(a: dict) -> str:
    parts = [f"{k} {v}" for k, v in sorted(a["tokens_by_kind"].items(),
                                           key=lambda kv: -kv[1])]
    return " · ".join(parts) if parts else "(nothing — budget unspent)"


def show(kg, a: dict, b: dict, pa: str, pb: str, ra, rb, budget: int,
         full_text: bool) -> None:
    q = a["query"].split("|")
    h = kg.ent2txt.get(q[0], q[0]) if kg else q[0]
    r = kg.rel2txt.get(q[1], q[1]) if kg else q[1]
    t = kg.ent2txt.get(q[2], q[2]) if (kg and len(q) > 2) else (q[2] if len(q) > 2 else "?")

    print(f"\n{'─'*78}")
    if a["direction"] == "tail":
        print(f"  ({h}, {r}, ?)        gold: {t}")
    else:
        print(f"  (?, {r}, {t})        gold: {h}")
    print(f"  budget {budget} tokens · both policies · context describes "
          f"{a.get('anchor', '?')}")
    if ra is not None and rb is not None:
        arrow = "→" if rb < ra else ("←" if rb > ra else "=")
        verdict = ("★ improved" if rb < ra else
                   "⚠️ regressed" if rb > ra else "unchanged")
        print(f"  rank  {pa} {ra}  {arrow}  {pb} {rb}   {verdict}")
    print(f"{'─'*78}")

    for pid, al in ((pa, a), (pb, b)):
        print(f"\n  {pid}   spent {al['spent']}/{al['budget']} "
              f"({al['utilisation']:.0%})")
        print(f"    {spend_line(al)}")
        for blk in al["kept"]:
            txt = blk.get("text", "")
            if not full_text and len(txt) > 88:
                txt = txt[:85] + "…"
            print(f"      · {blk['kind']:22s} {blk['tokens']:>4d} tok  {txt}")
        # ★ the reasons are the explicability half of the thesis title
        for key, why in list(al["reasons"].items())[:4]:
            print(f"        ↳ {key}: {why}")
        dropped = al.get("dropped", [])
        if dropped:
            names = ", ".join(f"{d['kind']}({d['tokens']})" for d in dropped[:4])
            print(f"      ✗ dropped: {names}")


def latex_table(kg, cases: list, pa: str, pb: str, budget: int) -> str:
    """A booktabs table, ready to paste. Escapes the underscores in relation ids."""
    def esc(s):
        return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Where the same " + str(budget) +
           r"-token budget goes under two policies. Cases are the "
           r"largest rank improvements, selected by rule, not by hand.}",
           r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
           r"\textbf{Query} & \textbf{Policy} & \textbf{Allocation} & "
           r"\textbf{Rank} \\", r"\midrule"]
    for a, b, ra, rb in cases:
        q = a["query"].split("|")
        h = kg.ent2txt.get(q[0], q[0]) if kg else q[0]
        r = kg.rel2txt.get(q[1], q[1]) if kg else q[1]
        out.append(f"\\multirow{{2}}{{*}}{{({esc(h)}, {esc(r)}, ?)}} "
                   f"& {esc(pa)} & {esc(spend_line(a))} & {ra} \\\\")
        out.append(f" & {esc(pb)} & {esc(spend_line(b))} & \\textbf{{{rb}}} \\\\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="WN18RR-ind")
    ap.add_argument("--root", default="data")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--a", default="S0_uniform", help="the baseline policy")
    ap.add_argument("--b", default="S4_instance", help="the policy under test")
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--direction", default="tail", choices=("tail", "head"))
    ap.add_argument("--tag", default="tuned")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--only-disagreements", action="store_true",
                    help="only queries where the two policies allocated differently")
    ap.add_argument("--worst", action="store_true",
                    help="★ show the largest REGRESSIONS instead — read these too")
    ap.add_argument("--full-text", action="store_true")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--results", default=None)
    ns = ap.parse_args()

    try:
        from src.data.loaders import load_kg
        kg = load_kg(ns.dataset, ns.root)
    except Exception:
        kg = None                          # ids instead of surface forms; still works

    A, _ = load_cell(ns.root, ns.dataset, ns.a, ns.budget, ns.direction)
    B, _ = load_cell(ns.root, ns.dataset, ns.b, ns.budget, ns.direction)

    res = Path(ns.results) if ns.results else None
    if res is None:
        try:
            from src.utils.config import load_config
            res = Path(load_config(ns.config)["output"]["results_dir"])
        except Exception:
            res = Path("results")
    RA = load_ranks(res, ns.dataset, ns.a, ns.budget, ns.direction, ns.tag)
    RB = load_ranks(res, ns.dataset, ns.b, ns.budget, ns.direction, ns.tag)

    bykey = {b["query"]: b for b in B}
    pairs = [(a, bykey[a["query"]]) for a in A if a["query"] in bykey]
    print(f"[qual] {ns.dataset} · B={ns.budget} · {ns.direction} · "
          f"{ns.a} vs {ns.b} · {len(pairs):,} shared queries")

    if ns.only_disagreements:
        before = len(pairs)
        pairs = [(a, b) for a, b in pairs
                 if a["tokens_by_kind"] != b["tokens_by_kind"]]
        print(f"[qual] {len(pairs):,}/{before:,} queries allocated DIFFERENTLY "
              f"({len(pairs)/max(1,before):.0%})")
        if not pairs:
            print("\n  ✋ The two policies allocated IDENTICALLY on every query.")
            print("     The cell measures nothing. Either the budget is large")
            print("     enough that everything fits, or these policies do not")
            print("     differ on this graph. Both are findings — report one.")
            return

    def qkey(a):
        p = a["query"].split("|")
        return f"{a['direction']}|{p[0]}|{p[1]}|{p[2] if len(p) > 2 else ''}"

    if RA and RB:
        scored = []
        for a, b in pairs:
            k = qkey(a)
            if k in RA and k in RB:
                scored.append((a, b, RA[k], RB[k]))
        if scored:
            scored.sort(key=lambda x: (x[3] - x[2]) if ns.worst else (x[2] - x[3]),
                        reverse=True)
            rule = ("largest REGRESSION (b worse than a)" if ns.worst
                    else "largest IMPROVEMENT (b better than a)")
            print(f"[qual] ranks available — cases selected by rule: {rule}")
            cases = scored[: ns.n]
        else:
            print("[qual] ⚠️ ranks not aligned; showing the first cases instead")
            cases = [(a, b, None, None) for a, b in pairs[: ns.n]]
    else:
        print(f"[qual] ⚠️ no evaluated ranks under {res} — showing allocations only.")
        print(f"       Run chapter3.evaluate for both policies to get the "
              f"rank column, which is what makes the table persuasive.")
        cases = [(a, b, None, None) for a, b in pairs[: ns.n]]

    for a, b, ra, rb in cases:
        show(kg, a, b, ns.a, ns.b, ra, rb, ns.budget, ns.full_text)

    if ns.latex and cases and cases[0][2] is not None:
        print(f"\n{'='*78}\nLATEX — paste into the paper\n{'='*78}")
        print(latex_table(kg, cases, ns.a, ns.b, ns.budget))

    print(f"\n{'='*78}")
    print("  ⚠️ Caption must state the SELECTION RULE. A hand-picked example is")
    print("     an anecdote; a rule-selected one is evidence. And show a")
    print("     regression too (--worst): a case study with only wins reads as")
    print("     advocacy, and reviewers of CATS-adjacent papers look for it.")


if __name__ == "__main__":
    main()
