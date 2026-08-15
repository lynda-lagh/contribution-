"""
CHAPTER 1 — evaluate one checkpoint on BOTH test sets.

★ A single accuracy number cannot express this chapter's claim. Every model is
scored on the REAL test set and the ANONYMISED one, and the GAP between them is
the result:

    gap = acc_real - acc_anon        how much is carried by entity surface forms

Scoring uses the LOGIT parser -- P(Yes) vs P(No) at the first response position,
no generation. Chapter 1 established that all four parsers agree after tuning
(0.9315 across the board), so the format-independent one is the right default
and costs one forward pass instead of sixteen decoded tokens.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load(base: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter or base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float32, attn_implementation="sdpa").cuda()
    if adapter and Path(adapter).exists():
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    return m.eval(), tok


def _score_set(model, tok, records: list[dict], limit: int) -> dict:
    from src.data.prompts import ALPACA_NO_INPUT
    from src.infer.scoring import yes_no_probabilities

    recs = records[:limit]
    prompts = [ALPACA_NO_INPUT.format(instruction=r["instruction"]) for r in recs]
    labels = [r["label"] for r in recs]
    probs = yes_no_probabilities(model, tok, prompts)

    conf = [p_yes for p_yes, _ in probs]
    pred = [1 if c > 0.5 else -1 for c in conf]
    correct = [p == l for p, l in zip(pred, labels)]

    return {
        "accuracy": float(np.mean(correct)),
        "n": len(recs),
        "positive_rate": float(np.mean([p == 1 for p in pred])),
        "confidences": conf,
        "correct": correct,
        "records": recs,
    }


def evaluate_both(cfg: dict, cond, dataset: str, adapter: str, limit: int = 2000) -> dict:
    """
    Score `adapter` on the real AND anonymised test sets.

    ⚠️ Both sets must exist. The anonymised one is not a Chapter 1 curiosity --
    it is standard infrastructure for every chapter, because the gap column is
    how this thesis reports any result at all.
    """
    root = cfg["data"]["root"]
    base = cfg["model"]["name"]

    paths = {
        "real": Path(root, dataset, "built", "test_instructions.json"),
        "anon": Path(root, f"{dataset}-anon", "built", "test_instructions.json"),
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing test set(s) {missing}.\n"
            f"  real: python -m src.data.build_instructions --dataset {dataset} "
            f"--n_triples {cfg['data']['train_triples']} --seed {cfg['seed']}\n"
            f"  anon: same command with --anonymise")

    model, tok = _load(base, adapter)
    out = {}
    for key, p in paths.items():
        recs = json.loads(p.read_text(encoding="utf-8"))
        r = _score_set(model, tok, recs, limit)
        out[f"acc_{key}"] = r["accuracy"]
        out[f"positive_rate_{key}"] = r["positive_rate"]
        out[f"_{key}"] = r
        print(f"  [{key:4s}] acc {r['accuracy']:.4f}  "
              f"(predicted Yes {r['positive_rate']:.1%})")

    out["gap"] = out["acc_real"] - out["acc_anon"]
    out["condition"] = cond.id
    out["isolates"] = cond.isolates
    out["memorisation_share"] = (
        out["gap"] / (out["acc_real"] - 0.5) if out["acc_real"] > 0.5 else None)

    # ★ seen/unseen, computed here so it is never a separate manual step
    from .analysis import calibration_by_familiarity, seen_unseen
    rr = out["_real"]
    if rr["records"] and "seen_head" in rr["records"][0]:
        out["seen_unseen"] = seen_unseen(rr["records"], rr["correct"])
        out["calibration"] = calibration_by_familiarity(
            [max(c, 1 - c) for c in rr["confidences"]], rr["correct"], rr["records"])

    # trim the bulky arrays before saving
    for k in ("_real", "_anon"):
        out[k] = {kk: vv for kk, vv in out[k].items()
                  if kk not in ("records", "correct", "confidences")}
    return out
