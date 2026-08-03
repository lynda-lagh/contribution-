"""
EVALUATE EVERY TRAINED ADAPTER -- one table, one JSON, no generation.

    python -m scripts.eval_all                 # every checkpoint found
    python -m scripts.eval_all --limit 2000    # the frozen test subset
    python -m scripts.eval_all --only ch2      # substring filter

Why logit scoring and not chapters/ch4
--------------------------------------
Chapter 4 SAMPLES (k=10, top-p .95) because it needs sampling-disagreement as a
confidence source. That costs ~100 min per adapter. For plain accuracy on a
binary task none of it is needed: P("Yes") vs P("No") from ONE forward pass is
the same decision the model would make, and `src/infer/scoring.py` already
implements it batched. Minutes, not hours.

★ WHAT THIS SCRIPT REFUSES TO HIDE
----------------------------------
YAGO3-10 and WN18RR ship test.tsv WITHOUT a label column, so
`loaders._read_triples(has_label=True)` leaves every label None and
`build_instructions.build()` writes every test record as YES. The test set is
then 100% POSITIVE, and on a single-class set:

    * "accuracy" is really RECALL on positives,
    * chance is NOT 0.5 -- a model that always answers Yes scores 1.000,
    * precision/abstention numbers are close to free.

WN11 and FB13 are unaffected: their test.tsv carries real +1/-1.

So every row is stamped with the class balance, and single-class rows are
labelled `RECALL-ONLY (not comparable to 0.5)`. A number you cannot compare is
worse than no number, because it will end up in a table anyway.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")          # see src/__init__.py
os.environ.setdefault("USE_FLAX", "0")

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.prompts import ALPACA_NO_INPUT
from src.infer.scoring import yes_no_probabilities


# ---------------------------------------------------------------- adapter -> data
def dataset_for(adapter_name: str) -> str | None:
    """
    Map a checkpoint name to the dataset variant it was TRAINED on.

    This must be exact. Scoring an adapter against prompts built with different
    enrichment measures DISTRIBUTION SHIFT, not quality -- the same trap
    ch1/analyse.py documents for the anonymised arm.

        ch1-WN11-lora                  -> WN11
        ch1-WN11-anon-lora             -> WN11-anon
        ch2-lora-E123182-T10000-s42    -> YAGO3-10-E123182
        ch3-YAGO3-10-L2-lora           -> YAGO3-10-L2
    """
    p = adapter_name.split("-")
    if adapter_name.startswith("ch1-"):
        return "-".join(p[1:-1])                      # WN11 | WN11-anon
    if adapter_name.startswith("ch2-"):
        for part in p:
            if part.startswith("E") and part[1:].isdigit():
                return f"YAGO3-10-{part}"
        return None
    if adapter_name.startswith("ch3-"):
        for part in p:
            if part in ("L0", "L1", "L2", "L3", "L4"):
                return f"YAGO3-10-{part}"
    return None


def load_test(root: str, dataset: str, limit: int):
    f = Path(root, dataset, "built", "test_instructions.json")
    if not f.exists():
        return None
    rows = json.loads(f.read_text(encoding="utf-8"))[:limit]
    prompts = [ALPACA_NO_INPUT.format(instruction=r["instruction"]) for r in rows]
    # gold verdict from the stored answer string, not from the name of anything
    gold = ["yes" if r.get("output", "").strip().lower().startswith("yes") else "no"
            for r in rows]
    return prompts, gold


# ---------------------------------------------------------------- metrics
def score(gold: list[str], pred: list[str]) -> dict:
    """Accuracy plus the per-class detail that a single-class test set makes essential."""
    n = len(gold)
    tp = sum(g == "yes" and p == "yes" for g, p in zip(gold, pred))
    tn = sum(g == "no" and p == "no" for g, p in zip(gold, pred))
    fp = sum(g == "no" and p == "yes" for g, p in zip(gold, pred))
    fn = sum(g == "yes" and p == "no" for g, p in zip(gold, pred))
    n_pos, n_neg = gold.count("yes"), gold.count("no")
    single = n_pos == 0 or n_neg == 0

    acc = (tp + tn) / n if n else 0.0
    recall = tp / n_pos if n_pos else None
    specificity = tn / n_neg if n_neg else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)
    # ★ balanced accuracy is the honest headline when classes are skewed;
    #   undefined (None) when one class is absent, rather than silently = accuracy
    balanced = ((recall + specificity) / 2
                if recall is not None and specificity is not None else None)

    return {
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "single_class_test_set": single,
        "accuracy": acc,
        "balanced_accuracy": balanced,
        "recall_on_positives": recall,
        "specificity_on_negatives": specificity,
        "precision": precision,
        "f1": f1,
        "predicted_yes_rate": pred.count("yes") / n if n else 0.0,
        "chance": None if single else 0.5,
        "interpretation": ("RECALL-ONLY: test set is single-class, so this is NOT "
                           "comparable to chance 0.5 (always-Yes scores 1.000)"
                           if single else "balanced test set: comparable to chance 0.5"),
    }


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@torch.no_grad()
def evaluate(base: str, adapter: str | None, prompts: list[str], gold: list[str],
             batch_size: int = 8, device: str = "cuda", dtype: str = "fp32") -> dict:
    tok = AutoTokenizer.from_pretrained(adapter or base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ★ DTYPE IS A MEMORY *AND* CORRECTNESS SETTING.
    #
    # fp32 is the default because the chapters' numbers rest on these logits.
    # But fp32 weights for Qwen2.5-1.5B are ~6.2 GB and will NOT fit a 4 GB card.
    #
    # fp16 IS SAFE HERE -- but only with sdpa. The smoke test's dtype probe
    # measured exactly this on a T4:
    #
    #     dtype  attn    finite  max|logit|
    #     fp16   eager    False       nan   <-- the NaN everyone warns about
    #     fp16   sdpa      True      27.2   <-- indistinguishable from fp32
    #     fp32   sdpa      True      27.3
    #
    # So the danger is EAGER, not fp16. attn_implementation is pinned to sdpa
    # below, which is what makes --dtype fp16 defensible on a small GPU.
    # bf16 is offered for Ampere+ cards (RTX 30xx); it is NOT native on T4.
    m = AutoModelForCausalLM.from_pretrained(
        base, dtype=DTYPES[dtype], attn_implementation="sdpa").to(device)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()

    probs = yes_no_probabilities(m, tok, prompts, batch_size=batch_size, device=device)
    pred = ["yes" if py >= pn else "no" for py, pn in probs]

    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    out = score(gold, pred)
    out["mean_p_yes"] = sum(p for p, _ in probs) / len(probs) if probs else 0.0
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--limit", type=int, default=2000,
                    help="test items; keep IDENTICAL across adapters for paired tests")
    ap.add_argument("--only", default=None, help="substring filter, e.g. ch2 or L3")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--untuned", action="store_true",
                    help="also score the BASE model -- the floor every adapter must beat")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="fp32 needs ~6.2 GB; use fp16 on a 4-6 GB card (safe with sdpa)")
    ns = ap.parse_args()

    device = pick_device(ns.device)
    print(f"[eval] device={device}  dtype={ns.dtype}  batch={ns.batch_size}")
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        gb = total / 1e9
        need = {"fp32": 6.2, "fp16": 3.1, "bf16": 3.1}[ns.dtype]
        print(f"[eval] GPU {torch.cuda.get_device_name(0)}  {gb:.1f} GB total, "
              f"{free/1e9:.1f} GB free | weights need ~{need} GB")
        if need > free / 1e9:
            raise SystemExit(
                f"\n{ns.dtype} weights (~{need} GB) do not fit in {free/1e9:.1f} GB free.\n"
                f"  * on a 4 GB card:  --dtype fp16 --batch-size 1\n"
                f"  * still tight?     --device cpu   (slow but correct)\n"
                f"fp16 is SAFE with sdpa -- see the dtype probe note in evaluate().")

    from src.utils.config import load_config
    cfg = load_config(ns.config)
    base = cfg["model"]["name"]
    root = cfg["data"]["root"]
    ckpt_root = Path(cfg["output"]["adapter_dir"])
    res_dir = Path(cfg["output"]["results_dir"]); res_dir.mkdir(parents=True, exist_ok=True)

    adapters = sorted(d.name for d in ckpt_root.iterdir()
                      if d.is_dir() and (d / "adapter_config.json").exists()) \
        if ckpt_root.exists() else []
    if ns.only:
        adapters = [a for a in adapters if ns.only in a]

    print("=" * 78)
    print(f"EVALUATION -- {len(adapters)} adapter(s), limit={ns.limit}, logit scoring")
    print("=" * 78)
    if not adapters:
        raise SystemExit(f"no adapters with adapter_config.json under {ckpt_root}/")

    rows: list[dict] = []

    # ★ the untuned floor. Chapter 1 found the untuned model is FAR from useless
    #   (WN11 logit 0.692), so "the adapter beat chance" is the wrong bar --
    #   "the adapter beat the untuned model on the same prompts" is the right one.
    if ns.untuned:
        seen = set()
        for a in adapters:
            ds = dataset_for(a)
            if ds is None or ds in seen:
                continue
            seen.add(ds)
            data = load_test(root, ds, ns.limit)
            if data is None:
                continue
            print(f"\n[untuned] {ds} ...")
            r = evaluate(base, None, *data, batch_size=ns.batch_size,
                         device=device, dtype=ns.dtype)
            rows.append({"adapter": f"UNTUNED ({ds})", "dataset": ds,
                         "chapter": "baseline", "peft": "none", **r})
            print(f"          acc {r['accuracy']:.4f}  yes-rate {r['predicted_yes_rate']:.3f}")

    for a in adapters:
        ds = dataset_for(a)
        if ds is None:
            print(f"\n[skip] {a}: cannot infer its dataset")
            continue
        data = load_test(root, ds, ns.limit)
        if data is None:
            print(f"\n[skip] {a}: data/{ds}/built/test_instructions.json not found")
            continue
        print(f"\n[eval] {a}  on  {ds} ...")
        r = evaluate(base, str(ckpt_root / a), *data, batch_size=ns.batch_size,
                     device=device, dtype=ns.dtype)
        rows.append({"adapter": a, "dataset": ds,
                     "chapter": a.split("-")[0], "peft": a.split("-")[-1], **r})
        flag = "  ⚠️ single-class" if r["single_class_test_set"] else ""
        print(f"       acc {r['accuracy']:.4f}  yes-rate {r['predicted_yes_rate']:.3f}"
              f"  (+{r['n_pos']}/-{r['n_neg']}){flag}")

    # ---- the frozen probe is a RESULT, not an adapter: fold it in ---------
    probe = Path(res_dir, "ch2-probe-E123182-T10000-s42.json")
    if probe.exists():
        p = json.loads(probe.read_text(encoding="utf-8"))
        rows.append({"adapter": "ch2-probe (frozen, no training)",
                     "dataset": "YAGO3-10-E123182", "chapter": "ch2", "peft": "probe",
                     "accuracy": p.get("accuracy"), "n": p.get("n_test"),
                     "trainable_params": p.get("trainable_params"),
                     "single_class_test_set": True,
                     "interpretation": "from ch2 probe run; same single-class test set"})

    payload = {
        "base_model": base,
        "limit": ns.limit,
        "scoring": "logit comparison P(Yes) vs P(No), no generation",
        "n_adapters": len(rows),
        "warning": ("Rows with single_class_test_set=true are RECALL on positives, "
                    "not accuracy against chance 0.5. YAGO3-10/WN18RR ship no test "
                    "labels, so build_instructions writes an all-positive test set."),
        "rows": rows,
    }
    out_json = res_dir / "evaluation_summary.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- markdown table, paste-ready for the thesis ----------------------
    md = ["| adapter | dataset | acc | bal.acc | recall+ | spec- | yes-rate | test |",
          "|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v:.4f}"
    for r in rows:
        md.append(f"| {r['adapter']} | {r['dataset']} | {f(r.get('accuracy'))} | "
                  f"{f(r.get('balanced_accuracy'))} | {f(r.get('recall_on_positives'))} | "
                  f"{f(r.get('specificity_on_negatives'))} | "
                  f"{f(r.get('predicted_yes_rate'))} | "
                  f"{'⚠️ 1-class' if r.get('single_class_test_set') else 'balanced'} |")
    out_md = res_dir / "evaluation_summary.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    print("\n" + "=" * 78)
    print("\n".join(md))
    print("=" * 78)
    n_single = sum(bool(r.get("single_class_test_set")) for r in rows)
    if n_single:
        print(f"\n⚠️  {n_single}/{len(rows)} rows use a SINGLE-CLASS test set.")
        print("    Those are recall on positives. An always-Yes model scores 1.000,")
        print("    so they cannot be read against chance = 0.5, and they cannot be")
        print("    compared with WN11/FB13 rows. Generate test negatives to fix.")
    print(f"\nsaved -> {out_json}\nsaved -> {out_md}")


if __name__ == "__main__":
    main()
