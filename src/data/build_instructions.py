"""
Build Alpaca-format instruction data -- our fork of KG-LLM's instructions_*.py.

    python -m src.data.build_instructions --dataset WN11 --n_triples 10000 --seed 42

Differences from KG-LLM:
  * stratified-by-relation sampling (rare relations survive)
  * pluggable negative strategy (random | type_consistent | kge_near_miss)
  * optional --anonymise for the contamination control (KG-CF)
  * writes a manifest.json recording EXACTLY what was built, so every run is
    reproducible and the triples-vs-instances distinction is never lost
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .loaders import anonymise, load_kg
from .negatives import make_negatives
from .prompts import NO, YES, PromptConfig, to_alpaca_record, triple_classification_instruction
from .sampling import relation_frequency_report, sample_triples


def build(dataset: str, n_triples: int, seed: int = 42, root: str = "data",
          negatives: str = "random", anonymised: bool = False,
          out_dir: str | None = None, prompt_cfg: PromptConfig | None = None) -> dict:
    kg = load_kg(dataset, root)
    if anonymised:
        kg = anonymise(kg)
    cfg = prompt_cfg or PromptConfig()

    pos = sample_triples(kg.train, n_triples, seed=seed,
                         stratified=True, min_per_relation=10)
    neg = make_negatives(pos, kg, strategy=negatives, seed=seed)

    records, rng = [], random.Random(seed)
    for p, n in zip(pos, neg):
        records.append(to_alpaca_record(triple_classification_instruction(p, kg, cfg), YES))
        records.append(to_alpaca_record(triple_classification_instruction(n, kg, cfg), NO))
    rng.shuffle(records)

    # test set: KG-LLM's test.tsv already carries +1/-1 labels -- no sampling needed
    test_records = [
        {**to_alpaca_record(triple_classification_instruction(t, kg, cfg),
                            YES if t.label == 1 else NO),
         "label": t.label}
        for t in kg.test
    ]

    out = Path(out_dir or f"{root}/{kg.name}/built")
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_instructions.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    (out / "test_instructions.json").write_text(json.dumps(test_records, indent=1), encoding="utf-8")

    manifest = {
        "dataset": kg.name,
        "anonymised": anonymised,
        "seed": seed,
        "negatives_strategy": negatives,
        "n_triples_sampled": len(pos),          # <- what we asked for
        "n_train_instances": len(records),      # <- what the model actually sees (~2x)
        "n_test_instances": len(test_records),
        "kg": kg.describe(),
        "relation_frequency": relation_frequency_report(pos),
        "prompt_config": vars(cfg) | {"extras": sorted(cfg.extras)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\n[build] wrote -> {out}")
    print(f"[build] {len(pos)} triples -> {len(records)} instances "
          f"(1 positive + 1 negative each). Report BOTH numbers.")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="WN11 | FB13 | WN18RR | YAGO3-10")
    ap.add_argument("--n_triples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default="data")
    ap.add_argument("--negatives", default="random",
                    choices=["random", "type_consistent", "kge_near_miss"])
    ap.add_argument("--anonymise", action="store_true",
                    help="KG-CF contamination control: entity names -> entity{i}")
    ap.add_argument("--out_dir", default=None)
    ns = ap.parse_args()

    from ..utils.config import set_all_seeds
    set_all_seeds(ns.seed)          # sampling + negative generation happen HERE,
                                    # before any Trainer exists to seed them

    build(dataset=ns.dataset, n_triples=ns.n_triples, seed=ns.seed, root=ns.root,
          negatives=ns.negatives, anonymised=ns.anonymise, out_dir=ns.out_dir)


if __name__ == "__main__":
    main()
