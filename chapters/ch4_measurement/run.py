"""
CHAPTER 4 -- Can the model know when to stay silent, and can we trust the reason?

★ ZERO TRAINING. Every step here is inference over checkpoints produced by
Chapters 2 and 3. That is why this is the safest chapter and must never be cut.

Pipeline
--------
    1. ONE sampling loop  -> Hits@K  AND  sampling-disagreement uncertainty
    2. compare THREE confidence sources (log-prob | P(True) | disagreement)
    3. calibrate each (temperature / isotonic) -> ECE, Brier, reliability
    4. abstention -> RISK-COVERAGE CURVE           (verified 0/188)
    5. hallucination type 1 (OOV) and ★ type 2 (type violation, never measured)
    6. emit the three computable traces for the review queue

    python -m chapters.ch4_measurement.run --adapter checkpoints/ch2-lora-... --dataset YAGO3-10
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.loaders import load_kg
from src.data.prompts import ALPACA_NO_INPUT
from src.eval.abstention import full_report as abstention_report
from src.eval.calibration import compare_sources
from src.eval.hallucination import hallucination_report
from src.infer.generate import (confidence_sources, greedy_predictions, hits_at_k,
                                sample_predictions)
from src.infer.scoring import sequence_logprob


def _norm(s: str) -> str:
    """Surface normalisation. 'Alan_Turing' / 'alan turing,  a mathematician' -> 'alan turing'."""
    s = s.strip().split(",")[0].strip().replace("_", " ").lower()
    return " ".join(s.split())


_YES = re.compile(r"\b(yes|true|correct)\b", re.I)
_NO = re.compile(r"\b(no|not|n't|never|false|incorrect)\b", re.I)


def clean_verdict(text: str) -> str | None:
    """
    A CLEAN yes/no reading -- deliberately NOT one of Chapter 1's parsers.

    Chapter 1's StrictParser and LenientParser exist to REPRODUCE the literature's
    scoring, bug included (LenientParser's `find("no")` fires inside "know",
    "cannot", "another"). They are objects of study. Scoring Chapter 4 with one of
    them would import that bug into our own results.

    Three rules, in this order, each one earning its place:

    1. REFUSAL first. "I don't know" contains "n't"; every naive rule reads it as a
       negative prediction. That conversion is the exact phenomenon Chapter 1
       measures (KG-LLM Table VI; Knit Fig. 1 lists "I don't know." as a
       hallucination) -- so here it must be None, an abstention, never a "no".

    2. NEGATION before affirmation. The gold negative is "No, this is not true.",
       which CONTAINS the token "true". Testing yes-markers first therefore reads
       the gold negative as positive and silently inflates accuracy. Negation scope
       wins over the affirmative token inside it.

    3. Word boundaries throughout, so "know" is not "no" and "Nope" is not "no".
    """
    from src.eval.parse import is_refusal
    if is_refusal(text):
        return None
    if _NO.search(text):
        return "no"
    if _YES.search(text):
        return "yes"
    return None


def correctness(preds: list[str], golds: list[str], *, binary: bool) -> list[bool]:
    """
    binary=True  (WN11 / FB13): compare VERDICTS, not string prefixes.
    binary=False (WN18RR / YAGO3-10): normalised match on the entity name -- the
                 same rule `hits_at_k` applies, so hits@1 and hits@3 stay consistent.
    """
    if binary:
        return [(v := clean_verdict(pr)) is not None and v == clean_verdict(g)
                for pr, g in zip(preds, golds)]
    return [bool(_norm(pr)) and _norm(pr) == _norm(g) for pr, g in zip(preds, golds)]


def load(base: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(adapter or base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float32, attn_implementation="sdpa").cuda()  # fp16 -> NaN
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    return m.eval(), tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--adapter", required=True, help="a Ch2/Ch3 checkpoint")
    ap.add_argument("--limit", type=int, default=2000, help="fixed test subset")
    ap.add_argument("--k", type=int, default=10, help="samples per prompt")
    ns = ap.parse_args()

    from src.utils.config import load_config
    cfg = load_config(ns.config)      # ★ seeds everything. Chapter 4 SAMPLES (k=10,
                                      # top-p .95): without a seed the disagreement
                                      # confidence source is not reproducible.
    base = cfg["model"]["name"]
    tag = Path(ns.adapter).name
    res_dir = Path(cfg["output"]["results_dir"], f"ch4_{tag}")
    res_dir.mkdir(parents=True, exist_ok=True)

    kg = load_kg(ns.dataset, cfg["data"]["root"])
    test = json.loads(Path(cfg["data"]["root"], ns.dataset, "built",
                           "test_instructions.json").read_text(encoding="utf-8"))[: ns.limit]
    prompts = [ALPACA_NO_INPUT.format(instruction=r["instruction"]) for r in test]
    golds = [r.get("output", "") for r in test]
    is_binary = ns.dataset in ("WN11", "FB13")

    # ★ ALIGNMENT GUARD. Steps 5 (hallucination) zips `kg.test` against `greedy`
    # POSITIONALLY. test_instructions.json is written from kg.test in order, so this
    # holds -- but if the built data is ever stale relative to the raw KG the two
    # silently desynchronise and every type-violation number becomes meaningless.
    if len(test) > len(kg.test):
        raise SystemExit(
            f"test_instructions.json ({len(test)}) is longer than kg.test "
            f"({len(kg.test)}) -- the built data is stale. Re-run build_instructions.")

    model, tok = load(base, ns.adapter)

    # 1 -- one loop, two uses -------------------------------------------------
    print(f"[ch4] sampling k={ns.k} (top-p .95, top-k 20) ...")
    preds = sample_predictions(model, tok, prompts, k=ns.k)
    greedy = greedy_predictions(model, tok, prompts)

    # ★ CORRECTNESS. The previous rule was `gold.startswith(pred[:3])`, which is
    # wrong in two ways: it is ASYMMETRIC (an empty prediction matches everything,
    # since "".startswith("") is True) and a 3-character prefix collides freely on
    # entity-prediction datasets. Chapter 1 built four parsers precisely so that
    # scoring is never ad hoc -- so we reuse them.
    correct = np.array(correctness(greedy, golds, binary=is_binary))
    metrics = {
        "hits_at_1": float(correct.mean()),
        "hits_at_3": hits_at_k(preds, golds, 3),
        "hits_at_10": hits_at_k(preds, golds, min(ns.k, 10)),
        "note": "MRR is NOT computable: sampling yields a SET, not a ranking",
    }
    print(f"[ch4] Hits@1 {metrics['hits_at_1']:.4f} | @3 {metrics['hits_at_3']:.4f} "
          f"| @10 {metrics['hits_at_10']:.4f}")

    # 2 -- three confidence sources ------------------------------------------
    print("[ch4] sequence log-probabilities ...")
    lps = [sequence_logprob(model, tok, p, g) for p, g in zip(prompts, greedy)]
    lp = np.array(lps)
    lp_norm = (lp - lp.min()) / (lp.max() - lp.min() + 1e-9)
    sources = confidence_sources(preds, logprobs=lp_norm.tolist())

    del model
    torch.cuda.empty_cache()

    # 3 -- calibration --------------------------------------------------------
    print("[ch4] calibration ...")
    calib = compare_sources({k: np.array(v) for k, v in sources.items()}, correct,
                            out_path=str(res_dir / "calibration.json"))

    # 4 -- abstention ---------------------------------------------------------
    best = calib["best_source"]
    print(f"[ch4] abstention on '{best}' ...")
    explanations = [p.explanation() for p in preds]          # ★ trace 2
    abst = abstention_report(np.array(sources[best]), correct, explanations,
                             out_path=str(res_dir / "abstention.json"))

    # 5 -- hallucination ------------------------------------------------------
    print("[ch4] hallucination rates ...")
    rels = [t.relation for t in kg.test[: len(greedy)]]
    gold_ids = [t.tail for t in kg.test[: len(greedy)]]
    hall = hallucination_report(greedy, rels, gold_ids, kg,
                                out_path=str(res_dir / "hallucination.json"))

    # 6 -- summary ------------------------------------------------------------
    summary = {
        "adapter": ns.adapter, "dataset": ns.dataset, "n_test": len(prompts), "k": ns.k,
        "metrics": metrics,
        "best_confidence_source": best,
        "calibration_ranking": calib["ranking"],
        "abstention": abst["risk_coverage"] | {"headline": abst["headline"]},
        "hallucination": {"type1_oov": hall["type1_oov_rate"],
                          "type2_violation": hall["type2_rate"],
                          "plausible_wrong": hall["plausible_wrong_rate"]},
    }
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 66)
    print(f"CHAPTER 4 -- {tag}")
    print("=" * 66)
    print(f"  Hits@1/3/10        {metrics['hits_at_1']:.4f} / "
          f"{metrics['hits_at_3']:.4f} / {metrics['hits_at_10']:.4f}")
    print(f"  best confidence    {best}")
    for r in calib["ranking"]:
        print(f"     {r['source']:22s} ECE {r['best_ece']:.4f} ({r['method']})")
    print(f"  {abst['headline']}")
    print(f"  type1 OOV          {hall['type1_oov_rate']:.1%}  "
          f"(GS-KGC WN18RR baseline 38.9-45.3%)")
    print(f"  ★ type2 violation  {hall['type2_rate']:.1%}  "
          f"(EGIT defined it; never measured before)")
    print(f"\nsaved -> {res_dir}")


if __name__ == "__main__":
    main()
