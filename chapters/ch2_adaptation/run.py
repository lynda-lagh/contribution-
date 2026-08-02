"""
CHAPTER 2 -- Is low-rank a bottleneck for entity memorisation in KGC?

NOT "which PEFT method wins". The question is mechanistic, and BOTH answers are results:

    MoRA margin GROWS with |E|  ->  bottleneck confirmed
    MoRA margin FLAT            ->  bottleneck refuted -- AND this is exactly what
                                    Chapter 1 predicts if tuning installs FORMAT.
                                    MoRA's own paper says it is "comparable on
                                    other tasks", so a null cannot read as failure.

The decisive experiment
-----------------------
|E| is swept by SUBSAMPLING WITHIN YAGO3-10 (induced subgraph), NOT across
FB15k-237/WN18RR/YAGO3-10 -- those differ in relation count (237/11/37), density
and label quality, which would confound |E| with everything else.

Grid (see STATUS.md for the reduced plan):

    LoRA  x 4 |E| points                    = 4   <- the sweep
    MoRA  x 4 |E| points                    = 4   <- the sweep
    BOFT  @ largest |E|                     = 1   <- preservation hypothesis
    frozen probe @ largest |E|              = 1   <- ★ the CONTROL (no training)
    data axis: 3k & 50k triples @ min/max |E|, LoRA+MoRA = 8
    3 seeds on LoRA vs MoRA @ largest |E|   = 4
                                              --
                                              22

Usage
-----
    python -m chapters.ch2_adaptation.run --entities 25000 --triples 10000 --peft lora
    python -m chapters.ch2_adaptation.run --sweep            # print the full plan
    python -m chapters.ch2_adaptation.run --entities 123182 --peft probe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data.build_instructions import build
from src.data.prompts import PromptConfig
from src.train.sft import train_sft

E_POINTS = [10_000, 25_000, 50_000, 123_182]     # YAGO3-10 full = 123,182
T_POINTS = [3_000, 10_000, 50_000]               # 10k is the fixed budget


def plan() -> list[dict]:
    """The 22-run plan. Printed with --sweep so it can be checked before committing."""
    runs: list[dict] = []
    for m in ("lora", "mora"):
        for e in E_POINTS:
            runs.append({"peft": m, "entities": e, "triples": 10_000, "seed": 42})
    runs.append({"peft": "boft", "entities": E_POINTS[-1], "triples": 10_000, "seed": 42})
    runs.append({"peft": "probe", "entities": E_POINTS[-1], "triples": 10_000, "seed": 42})
    # data axis -- only at the |E| extremes; this is what answers
    # "does the data requirement scale with |E|?"
    for m in ("lora", "mora"):
        for e in (E_POINTS[0], E_POINTS[-1]):
            for t in (T_POINTS[0], T_POINTS[-1]):
                runs.append({"peft": m, "entities": e, "triples": t, "seed": 42})
    # seeds only where the headline claim lives
    for m in ("lora", "mora"):
        for s in (43, 44):
            runs.append({"peft": m, "entities": E_POINTS[-1], "triples": 10_000, "seed": s})
    return runs


def run_id(peft: str, entities: int, triples: int, seed: int) -> str:
    return f"ch2-{peft}-E{entities}-T{triples}-s{seed}"


def one_dpo(cfg: dict, sft_adapter: str, entities: int, triples: int, seed: int,
            dataset: str = "YAGO3-10", negatives: str = "type_consistent",
            beta: float = 0.1) -> dict:
    """
    PHASE 2 -- DPO on top of an existing SFT adapter. One run, not a fresh grid.

    ⚠️ The negative strategy is the contribution here. KG-LLM trains on
    `random.choice(all_entities)` -- uniformly random, therefore mostly
    type-violating and trivially separable. We default to `type_consistent`
    so the rejected sample is a PLAUSIBLE near-miss, which is what makes the
    preference signal meaningful.
    """
    from src.data.loaders import load_kg
    from src.data.sampling import sample_triples
    from src.train.dpo import build_preference_pairs, train_dpo

    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = seed

    ds_name = f"{dataset}-E{entities}" if entities else dataset
    kg = load_kg(ds_name, cfg["data"]["root"])
    trip = sample_triples(kg.train, triples, seed=seed, stratified=True, min_per_relation=10)

    pair_path = Path(cfg["data"]["root"], ds_name, "built", f"dpo_pairs_{negatives}.json")
    pairs = build_preference_pairs(kg, trip, strategy=negatives, seed=seed,
                                   out_path=str(pair_path))

    rid = f"ch2-dpo-{negatives}-E{entities}-T{triples}-s{seed}"
    summary = train_dpo(cfg, sft_adapter, pairs,
                        str(Path(cfg["output"]["adapter_dir"], rid)),
                        beta=beta, run_name=rid)
    summary |= {"run_id": rid, "entities": entities, "triples": triples, "seed": seed,
                "peft": "dpo", "dataset": dataset, "sft_adapter": sft_adapter}
    res = Path(cfg["output"]["results_dir"]); res.mkdir(parents=True, exist_ok=True)
    (res / f"{rid}.json").write_text(json.dumps(summary, indent=2))
    return summary


def one(cfg: dict, peft: str, entities: int, triples: int, seed: int,
        dataset: str = "YAGO3-10", negatives: str = "random") -> dict:
    cfg = json.loads(json.dumps(cfg))                 # deep copy
    cfg["seed"] = seed
    cfg["peft"]["method"] = peft
    cfg["data"]["train_triples"] = triples

    rid = run_id(peft, entities, triples, seed)
    data_dir = Path(cfg["data"]["root"], f"{dataset}-E{entities}", "built")

    if not (data_dir / "train_instructions.json").exists():
        print(f"[ch2] building data for |E|={entities} ...")
        from src.data.loaders import load_kg
        from src.data.sampling import entity_subset
        kg = entity_subset(load_kg(dataset, cfg["data"]["root"]), entities, seed=42)
        # persist the subset so every method sees the SAME graph
        sub_root = Path(cfg["data"]["root"], f"{dataset}-E{entities}")
        sub_root.mkdir(parents=True, exist_ok=True)
        (sub_root / "entity2text.txt").write_text(
            "\n".join(f"{k}\t{v}" for k, v in kg.ent2txt.items()), encoding="utf-8")
        (sub_root / "relation2text.txt").write_text(
            "\n".join(f"{k}\t{v}" for k, v in kg.rel2txt.items()), encoding="utf-8")
        (sub_root / "train.tsv").write_text(
            "\n".join(f"{t.head}\t{t.relation}\t{t.tail}" for t in kg.train), encoding="utf-8")
        (sub_root / "test.tsv").write_text(
            # `t.label or 1` would turn a legitimate 0 into 1; be explicit. Triples
            # from a KG without test labels (WN18RR/YAGO3-10) are all positives.
            "\n".join(f"{t.head}\t{t.relation}\t{t.tail}\t"
                      f"{1 if t.label is None else t.label}" for t in kg.test),
            encoding="utf-8")
        build(dataset=f"{dataset}-E{entities}", n_triples=triples, seed=seed,
              root=cfg["data"]["root"], negatives=negatives,
              prompt_cfg=PromptConfig())

    out_dir = Path(cfg["output"]["adapter_dir"], rid)

    if peft == "probe":
        from src.train.probe import train_probe
        tr = json.loads((data_dir / "train_instructions.json").read_text(encoding="utf-8"))
        te = json.loads((data_dir / "test_instructions.json").read_text(encoding="utf-8"))
        summary = train_probe(cfg, tr, te[:2000], output_dir=str(out_dir))
    else:
        summary = train_sft(cfg, str(data_dir), str(out_dir), run_name=rid)

    summary |= {"run_id": rid, "entities": entities, "triples": triples,
                "seed": seed, "peft": peft, "dataset": dataset}
    res = Path(cfg["output"]["results_dir"]); res.mkdir(parents=True, exist_ok=True)
    (res / f"{rid}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--peft", default="lora",
                    choices=["lora", "mora", "boft", "probe", "dpo"])
    ap.add_argument("--entities", type=int, default=25_000)
    ap.add_argument("--triples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negatives", default="random",
                    choices=["random", "type_consistent", "kge_near_miss"])
    ap.add_argument("--sft-adapter", default=None,
                    help="required for --peft dpo: the Phase-1 checkpoint to start from")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO KL strength")
    ap.add_argument("--sweep", action="store_true", help="print the plan and exit")
    ns = ap.parse_args()

    if ns.sweep:
        runs = plan()
        print(f"CHAPTER 2 PLAN -- {len(runs)} runs\n" + "-" * 58)
        for i, r in enumerate(runs, 1):
            print(f"{i:>3}. {run_id(**r)}")
        print("-" * 58)
        print("Probe costs no training. Chapter 4 reuses every checkpoint above.")
        return

    from src.utils.config import env_report, load_config
    cfg = load_config(ns.config, seed=ns.seed)      # ★ seeds python/numpy/torch/HF
    print(f"[ch2] env: {env_report()}")

    if ns.peft == "dpo":
        if not ns.sft_adapter:
            raise SystemExit(
                "--peft dpo requires --sft-adapter, e.g.\n"
                "  --sft-adapter checkpoints/ch2-mora-E123182-T10000-s42\n"
                "DPO is Phase 2: it continues training the winning SFT adapter.")
        neg = ns.negatives if ns.negatives != "random" else "type_consistent"
        if ns.negatives == "random":
            print("[ch2] note: switching negatives random -> type_consistent for DPO. "
                  "Random corruptions are trivially separable (they are KG-LLM's "
                  "baseline, and the weakness this contribution targets).")
        one_dpo(cfg, ns.sft_adapter, ns.entities, ns.triples, ns.seed,
                dataset=ns.dataset, negatives=neg, beta=ns.beta)
    else:
        one(cfg, ns.peft, ns.entities, ns.triples, ns.seed,
            dataset=ns.dataset, negatives=ns.negatives)


if __name__ == "__main__":
    main()
