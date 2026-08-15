"""
CHAPTER 1 — orchestrator.

    python -m chapter1.run --plan                     # what would run, and why
    python -m chapter1.run --condition A --train
    python -m chapter1.run --condition A --evaluate   # BOTH test sets -> the gap
    python -m chapter1.run --condition A --rank       # 50-way link prediction

★ Every model is evaluated on BOTH the real and the anonymised test set.
  The GAP between them is the chapter's contribution, so a single accuracy
  number is never enough and the runner will not produce one alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conditions import CEILING_NOTE, CONDITIONS, INTERPRETATION, ONLY_INCREASES, PROMPTS


def plan() -> None:
    print("=" * 78)
    print("CHAPTER 1 — PLAN")
    print("=" * 78)
    print("\nPHASE 0 · free, no GPU")
    print("  python -m chapter1.test_chapter1")
    print("  python -m chapter1.analysis --seen-unseen      # from existing results")
    print("\nPHASE 1 · training, 1 run per condition (~36 min each on a T4)")
    tot = 0
    for c in CONDITIONS.values():
        mins = 36 * (c.n_instances / 20_000)
        tot += mins
        print(f"  [{c.id}] {c.name:22s} {c.n_instances:>7,d} inst  ~{mins:4.0f} min   {c.isolates}")
    print(f"       {'':22s} {'':>7s}        ~{tot/60:4.1f} h total")
    print("\nPHASE 2 · evaluation, inference only")
    print("  each condition on BOTH test sets   -> the gap")
    print("  50-way ranking                     -> Hits@K + MRR (real KGC)")
    print("  prompt sweep P0–P4 on A and B      -> untuned AND tuned")
    print("\nORDER: C and G first — they carry the claim. Then D, E.")
    print(CEILING_NOTE)
    print("PRE-REGISTERED INTERPRETATIONS")
    for k, v in INTERPRETATION.items():
        print(f"  {k:10s} -> {v}")
    print(ONLY_INCREASES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--condition", default="A", choices=list(CONDITIONS))
    ap.add_argument("--prompt", default="P0", choices=list(PROMPTS))
    ap.add_argument("--dataset", default="WN11")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true", help="both test sets -> gap")
    ap.add_argument("--rank", action="store_true", help="50-way link prediction")
    ap.add_argument("--limit", type=int, default=2000)
    ns = ap.parse_args()

    if ns.plan:
        plan()
        return

    from src.utils.config import load_config, save_result
    cfg = load_config(ns.config)
    cond = CONDITIONS[ns.condition]
    root = cfg["data"]["root"]
    tag = f"ch1-{ns.dataset}-{ns.condition}"
    data_dir = Path(root, f"{ns.dataset}-{ns.condition}", "built")
    adapter = Path(cfg["output"]["adapter_dir"], tag)

    # ---------------------------------------------------------------- build
    if ns.build or (ns.train and not data_dir.exists()):
        from .data import build_condition
        m = build_condition(cond, ns.dataset, root,
                            cfg["data"]["train_triples"], cfg["seed"], ns.prompt)
        print(f"[build] {m['n_train_instances']:,} instances · "
              f"seen coverage {m['seen_coverage']:.1%}")
        print(f"[build] example: {m['example_positive']['instruction'][:110]}")

    # ---------------------------------------------------------------- train
    if ns.train:
        from src.train.sft import train_sft
        print(f"[train] {tag}  ({cond.isolates})")
        s = train_sft(cfg, str(data_dir), str(adapter), run_name=tag)
        s |= {"condition": cond.id, "n_instances": cond.n_instances,
              "isolates": cond.isolates, "reference": cond.reference}
        save_result(cfg, tag, s)

    # ------------------------------------------------------------- evaluate
    if ns.evaluate:
        from .evaluate import evaluate_both
        r = evaluate_both(cfg, cond, ns.dataset, str(adapter), ns.limit)
        save_result(cfg, f"{tag}-eval", r)
        print(f"\n  real {r['acc_real']:.4f} · anon {r['acc_anon']:.4f} · "
              f"GAP {r['gap']:+.4f}")

    # ----------------------------------------------------------------- rank
    if ns.rank:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "chapter1.rank",
                        "--adapter", str(adapter), "--dataset", ns.dataset,
                        "--condition", ns.condition, "--prompt", ns.prompt],
                       check=False)


if __name__ == "__main__":
    main()
