"""
Frozen-probe control (FLAME-style) -- Chapter 2's most important baseline.

    "FLAME ... extracts context-aware hidden states from intermediate layers of
     frozen LLMs to train data-efficient KGC classifiers ... first to achieve
     fine-tuned performance with 188x memory efficiency and 26.11x speedup."

Why this is a control and not a competitor
------------------------------------------
Chapter 2 asks whether low-rank adaptation is a bottleneck for entity memorisation.
A flat MoRA-vs-LoRA result is AMBIGUOUS on its own -- it could mean capacity does
not matter, or that the whole setup is insensitive.

The frozen probe disambiguates: if LoRA, MoRA and BOFT all merely MATCH a linear
probe on frozen representations, then no adaptation method is installing knowledge,
and Chapter 1's diagnosis is confirmed from a second direction.

Cost: no gradient ever reaches the LLM. Minutes, not hours.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.prompts import ALPACA_NO_INPUT
from ..eval.smi import extract_representations, sliced_mutual_information


def train_probe(cfg: dict, train_records: list[dict], test_records: list[dict],
                layer: int | None = None, pooling: str = "last",
                output_dir: str = "checkpoints/probe") -> dict:
    """
    train_records / test_records : [{"instruction": ..., "output"/"label": ...}]
    Labels: +1 / -1 (test) or derived from the "Yes"/"No" output string (train).
    """
    mcfg = cfg["model"]
    tok = AutoTokenizer.from_pretrained(mcfg["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"], torch_dtype=torch.float16,
        attn_implementation=mcfg.get("attn_implementation", "eager")).cuda().eval()

    if layer is None:                      # FLAME uses INTERMEDIATE layers
        layer = model.config.num_hidden_layers // 2

    def prep(recs):
        prompts = [ALPACA_NO_INPUT.format(instruction=r["instruction"]) for r in recs]
        y = np.array([r["label"] if "label" in r
                      else (1 if r["output"].strip().startswith("Yes") else -1)
                      for r in recs])
        return prompts, y

    tr_p, tr_y = prep(train_records)
    te_p, te_y = prep(test_records)

    print(f"[probe] extracting layer {layer} ({pooling} pooling) ...")
    Xtr = extract_representations(model, tok, tr_p, layer=layer, pooling=pooling)
    Xte = extract_representations(model, tok, te_p, layer=layer, pooling=pooling)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=cfg["seed"])
    clf.fit(Xtr_s, tr_y)

    pred = clf.predict(Xte_s)
    prob = clf.predict_proba(Xte_s)                     # -> Chapter 4 calibration
    pos_idx = list(clf.classes_).index(1)

    smi = sliced_mutual_information(Xtr, tr_y)

    out = {
        "method": "frozen_probe",
        "model": mcfg["name"],
        "layer": layer,
        "pooling": pooling,
        "n_train": len(tr_y),
        "n_test": len(te_y),
        "accuracy": float(accuracy_score(te_y, pred)),
        "smi_train": smi,
        "trainable_params": int(Xtr.shape[1] + 1),      # vs millions for LoRA
        "note": "no gradient reaches the LLM",
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(output_dir, "probe.npz"),
                        coef=clf.coef_, intercept=clf.intercept_, mu=mu, sd=sd,
                        classes=clf.classes_,
                        test_probs=prob[:, pos_idx], test_labels=te_y)
    Path(output_dir, "probe_summary.json").write_text(json.dumps(out, indent=2))

    del model
    torch.cuda.empty_cache()
    print(json.dumps(out, indent=2))
    return out


def layer_sweep(cfg: dict, train_records: list[dict], test_records: list[dict],
                layers: list[int] | None = None) -> dict:
    """
    Which depth carries the KGC signal? FLAME's claim is that intermediate layers
    do -- the final layer is specialised for next-token prediction.
    Cheap: representations are extracted once per layer, no training of the LLM.
    """
    from transformers import AutoConfig
    n = AutoConfig.from_pretrained(cfg["model"]["name"]).num_hidden_layers
    layers = layers or [n // 4, n // 2, 3 * n // 4, n]
    res = {f"layer_{L}": train_probe(cfg, train_records, test_records, layer=L,
                                     output_dir=f"checkpoints/probe_L{L}")
           for L in layers}
    best = max(res.items(), key=lambda kv: kv[1]["accuracy"])
    res["best"] = {"layer": best[0], "accuracy": best[1]["accuracy"]}
    print(f"\n[probe] best: {best[0]} acc={best[1]['accuracy']:.4f}")
    return res
