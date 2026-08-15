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

    # ★ DECIDE THE WAY LogitParser DOES: p_yes >= p_no, not p_yes > 0.5.
    #   `yes_no_probabilities` returns raw token probabilities that need not sum
    #   to 1 -- other tokens carry mass -- so a fixed 0.5 threshold is not the
    #   same rule as the parser's and the two would disagree on some records.
    conf = [p_yes for p_yes, _ in probs]                 # directional, P(Yes)
    pred = [1 if p_yes >= p_no else -1 for p_yes, p_no in probs]
    correct = [p == l for p, l in zip(pred, labels)]

    return {
        "accuracy": float(np.mean(correct)),
        "n": len(recs),
        "positive_rate": float(np.mean([p == 1 for p in pred])),
        # p_yes / p_no are the raw quantities; everything downstream derives from
        # them. `confidences` is kept as P(Yes) for backwards compatibility --
        # ⚠️ note that the OLDER chapters/ch1_diagnostic/analyse.py wrote an
        # UNDIRECTED margin under the same key. Consumers must check for p_yes.
        "p_yes": [float(a) for a, _ in probs],
        "p_no": [float(b) for _, b in probs],
        "confidences": conf,
        "correct": correct,
        "records": recs,
    }


def smi_block(cfg: dict, adapter: str | None, records: list[dict],
              n: int = 600) -> dict:
    """
    ★★ THE THIRD INSTRUMENT — and the wedge against the closest prior work.

    FLAME's claim, measured with SMI alone:

        "these representations reach fine-tuned-level SMI values, indicating that
         fine-tuning primarily aligns representations rather than injecting
         knowledge from the KG training set"

    Our first run: SMI **0.0105 -> 0.0474**, a 3.5x rise. Read alone, that says
    tuning DID install knowledge. Read beside anonymisation -- where 91% of the
    accuracy turned out to be surface form -- it says something sharper:

        ★ SMI CANNOT DISTINGUISH MEMORISATION FROM RELATIONAL KNOWLEDGE.
          Representations became more label-informative, and the information
          they encode is the entity NAME.

    That is the precise limitation of the nearest prior work, and it is what our
    anonymisation control resolves. Reporting SMI *and* the gap together is the
    argument; reporting either alone is not.

    600 samples: SMI needs samples, not all of them, and this is the slow step.
    """
    from src.data.prompts import ALPACA_NO_INPUT
    from src.eval.smi import smi_across_layers

    recs = records[:n]
    prompts = [ALPACA_NO_INPUT.format(instruction=r["instruction"]) for r in recs]
    y = [1 if r["label"] == 1 else 0 for r in recs]
    model, tok = _load(cfg["model"]["name"], adapter)
    try:
        return smi_across_layers(model, tok, prompts, y)
    finally:
        import torch
        del model
        torch.cuda.empty_cache()


def evaluate_both(cfg: dict, cond, dataset: str, adapter: str, limit: int = 2000,
                  with_smi: bool = False) -> dict:
    """
    Score `adapter` on the real AND anonymised test sets.

    ⚠️ Both sets must exist. The anonymised one is not a Chapter 1 curiosity --
    it is standard infrastructure for every chapter, because the gap column is
    how this thesis reports any result at all.
    """
    root = cfg["data"]["root"]
    base = cfg["model"]["name"]

    # ★ chapter1.data writes to data/{dataset}-{CONDITION}/built/, so the old
    #   data/{dataset}/built/ and data/{dataset}-anon/built/ never existed and
    #   this function could not find a test set at all. Resolve by condition,
    #   keeping the legacy names as a fallback.
    #
    #   The pair must be matched on EVERYTHING except names, or the gap measures
    #   two changes at once:
    #     untyped conditions (A, B, S) -> real = A, anon = B
    #     typed conditions  (C, D, E, G) -> real = G, anon = C   (both carry tags)
    typed = getattr(cond, "types", False)
    want = {"real": [f"{dataset}-G" if typed else f"{dataset}-A", dataset],
            "anon": [f"{dataset}-C" if typed else f"{dataset}-B",
                     f"{dataset}-anon"]}

    paths, missing = {}, {}
    for key, names in want.items():
        cands = [Path(root, n, "built", "test_instructions.json") for n in names]
        hit = next((p for p in cands if p.exists()), None)
        if hit is None:
            missing[key] = cands
        else:
            paths[key] = hit

    if missing:
        msg = [f"missing test set(s): {', '.join(missing)}"]
        for k, cands in missing.items():
            msg.append(f"  {k}: looked in " + " · ".join(str(c.parent) for c in cands))
        need = "G C" if typed else "A B"
        msg.append(f"\n  build them:  python -m chapter1.data --condition {need} "
                   f"--dataset {dataset}")
        raise FileNotFoundError("\n".join(msg))

    print(f"  [pair] real <- {paths['real'].parent.parent.name}   "
          f"anon <- {paths['anon'].parent.parent.name}"
          + ("   (typed pair)" if typed else ""))

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

    # ---- ★ SMI: the third instrument -------------------------------------
    if with_smi:
        from src.eval.smi import compare
        print("  [smi] tuned ...")
        s_tuned = smi_block(cfg, adapter, out["_real"]["records"])
        print("  [smi] untuned baseline ...")
        s_base = smi_block(cfg, None, out["_real"]["records"])
        out["smi"] = {"tuned": s_tuned, "untuned": s_base,
                      "comparison": compare(s_base, s_tuned)}
        c = out["smi"]["comparison"]
        print(f"  [smi] {c['smi_untuned']:.5f} -> {c['smi_tuned']:.5f} "
              f"({c['relative_change']:+.2f}x)")
        # ★ the joint reading — neither number means much alone
        out["smi"]["joint_reading"] = (
            f"SMI rose {c['relative_change']:+.1f}x while the anonymisation gap is "
            f"{out['gap']:+.4f}. Representations became more label-informative AND "
            f"the information is surface form -> SMI cannot separate memorisation "
            f"from relational knowledge. This is FLAME's limitation, resolved."
            if c["delta"] > 0 and out["gap"] > 0.15 else
            f"SMI delta {c['delta']:+.5f}, gap {out['gap']:+.4f} — the two "
            f"instruments do NOT point the same way. Investigate before writing.")
        print(f"  [smi] {out['smi']['joint_reading']}")

    # trim the bulky arrays before saving
    for k in ("_real", "_anon"):
        out[k] = {kk: vv for kk, vv in out[k].items()
                  if kk not in ("records", "correct", "confidences")}
    return out
