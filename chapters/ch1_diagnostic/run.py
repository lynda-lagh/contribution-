"""
CHAPTER 1 -- What does instruction tuning install: output FORMAT or entity KNOWLEDGE?

Two training runs total. Everything else is inference over the same outputs.

    # 1. build data
    python -m src.data.build_instructions --dataset WN11 --n_triples 10000
    python -m src.data.build_instructions --dataset WN11 --n_triples 10000 --anonymise

    # 2. train (2 runs)
    python -m chapters.ch1_diagnostic.run --dataset WN11
    python -m chapters.ch1_diagnostic.run --dataset WN11 --anonymise

    # 3. score untuned + tuned with all four parsers, then decompose
    python -m chapters.ch1_diagnostic.analyse --dataset WN11

Datasets: WN11 and FB13 only. Both are BINARY triple classification, so chance
is 50 -- which is what makes KG-LLM's untuned scores of 21.1 / 9.1 impossible to
read as a knowledge failure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.train.sft import train_sft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WN11", choices=["WN11", "FB13"])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--anonymise", action="store_true",
                    help="train on the entity-anonymised variant (KG-CF control)")
    ap.add_argument("--peft", default=None, help="override configs/base.yaml peft.method")
    ns = ap.parse_args()

    from src.utils.config import load_config
    cfg = load_config(ns.config)      # ★ seeds python/numpy/torch/HF. This is a
                                      # TRAINING runner -- raw yaml.safe_load left
                                      # LoRA init and data order unseeded.
    if ns.peft:
        cfg["peft"]["method"] = ns.peft

    name = f"{ns.dataset}{'-anon' if ns.anonymise else ''}"
    data_dir = Path(cfg["data"]["root"], name, "built")
    if not (data_dir / "train_instructions.json").exists():
        raise FileNotFoundError(
            f"{data_dir} not built. Run:\n"
            f"  python -m src.data.build_instructions --dataset {ns.dataset} "
            f"--n_triples {cfg['data']['train_triples']}"
            f"{' --anonymise' if ns.anonymise else ''}"
        )

    run_name = f"ch1-{name}-{cfg['peft']['method']}"
    out_dir = Path(cfg["output"]["adapter_dir"], run_name)

    print(f"[ch1] {run_name}")
    print(f"[ch1] data      {data_dir}")
    print(f"[ch1] adapters  {out_dir}   (adapter only, ~20-100 MB)")
    train_sft(cfg, str(data_dir), str(out_dir), run_name=run_name)


if __name__ == "__main__":
    main()
