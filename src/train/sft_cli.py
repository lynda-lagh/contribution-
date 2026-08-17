"""
Command-line wrapper around `train_sft`.

    python -m src.train.sft_cli \
        --data data/WN18RR-ind/built/mixed \
        --out  checkpoints/ch3-WN18RR-ind-shared \
        --run-name ch3-WN18RR-ind-shared

WHY THIS FILE EXISTS
--------------------
`chapter1/run.py` calls `train_sft()` in-process, so Chapter 1 never needed a
CLI. Chapter 3 trains ONE shared model from a notebook cell, where a subprocess
is preferable: a crashed training run then cannot take the kernel (and the
already-built prompts) down with it.

⚠️ `--data` must be a directory containing `train_instructions.json`, which is
   what `chapter3.data` writes. Passing the dataset root instead is the obvious
   mistake, so it is checked and reported rather than raising a KeyError forty
   lines later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="directory holding train_instructions.json")
    ap.add_argument("--out", required=True, help="adapter output directory")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="k.k=v",
                    help="config overrides, e.g. --set train.epochs=1 peft.r=16")
    ns = ap.parse_args()

    data = Path(ns.data)
    f = data / "train_instructions.json"
    if not f.exists():
        sibling = sorted(p.name for p in data.glob("*/train_instructions.json"))
        raise SystemExit(
            f"✋ {f} not found.\n"
            f"   --data must be the directory CONTAINING train_instructions.json.\n"
            + (f"   Did you mean one of: {sibling[:6]}\n" if sibling else "")
            + f"   Build it first:\n"
              f"     python -m chapter3.data --dataset <DS> --train-mixed")

    # parse overrides before loading anything heavy
    overrides = {}
    for kv in ns.set:
        if "=" not in kv:
            raise SystemExit(f"--set expects k.k=v, got {kv!r}")
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)            # ints, floats, bools, lists
        except json.JSONDecodeError:
            pass                         # leave as string
        overrides[k] = v

    from src.utils.config import load_config
    from src.train.sft import train_sft

    cfg = load_config(ns.config, overrides or None)

    n = len(json.loads(f.read_text(encoding="utf-8")))
    run = ns.run_name or Path(ns.out).name
    print(f"[sft_cli] {n:,} instances  ·  {cfg['model']['name']}  ·  "
          f"{cfg['peft']['method']}  ·  run={run}")
    print(f"[sft_cli] {data}  ->  {ns.out}")
    if overrides:
        print(f"[sft_cli] overrides: {overrides}")

    summary = train_sft(cfg, str(data), ns.out, run_name=run)

    Path(ns.out).mkdir(parents=True, exist_ok=True)
    print("\n" + json.dumps(summary, indent=2)[:1200])

    # ★ non-zero exit on a degenerate run, so a notebook loop cannot march past it
    curve = (summary or {}).get("curve", {})
    if curve.get("final_train_loss") == 0.0:
        print("\n✋ train_loss is exactly 0.0 — this is the fp16+eager NaN signature, "
              "not a perfect fit. Check model.attn_implementation=sdpa.")
        sys.exit(1)


if __name__ == "__main__":
    main()
