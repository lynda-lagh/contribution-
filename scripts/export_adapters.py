"""
★ EXPORT THE FINE-TUNED ADAPTERS — the small files, not the 3 GB base model.

WHAT ACTUALLY GOT TRAINED
-------------------------
    trainable params: 1,089,536  ||  all params: 1,544,803,840  ||  0.0705%

LoRA trains 0.07% of the network. The fine-tuned artefact is therefore about
**4 MB**, not 3 GB — `adapter_model.safetensors` plus a small JSON. The base
model is never modified and is re-downloaded from the hub on load.

WHY THE FOLDERS ARE STILL BIG
-----------------------------
`save_strategy="steps"` makes HF Trainer write `checkpoint-250/`,
`checkpoint-500/` … each carrying optimiser state, scheduler state and RNG state.
Once training has finished and `load_best_model_at_end` has written the winning
adapter to the top level, those are dead weight. `--prune` removes them.

`tokenizer.save_pretrained` also writes ~10 MB of vocabulary per adapter, and it
is byte-identical across all of them because the base model never changes. The
export drops it and records the base model id instead.

WHAT YOU GET
------------
    export/adapters/
        MANIFEST.json                  <- provenance + metrics for every adapter
        USAGE.md                       <- the six lines that load one
        ch1-YAGO3-10-A/
            adapter_config.json
            adapter_model.safetensors
            train_summary.json         <- fit verdict, loss curve, VRAM, runtime

    python -m scripts.export_adapters                 # export, report sizes
    python -m scripts.export_adapters --zip           # + one portable archive
    python -m scripts.export_adapters --prune         # + delete checkpoint-*/
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

# the only files that matter. Everything else is reproducible or redundant.
KEEP = ("adapter_config.json", "adapter_model.safetensors",
        "adapter_model.bin", "train_summary.json")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def metrics_for(name: str, results: Path) -> dict:
    """Pull whatever this adapter scored, so the export is self-describing."""
    out = {}
    for f in results.glob("*.json"):
        if name.replace("ch1-", "") not in f.stem:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for k in ("acc_real", "acc_anon", "gap", "memorisation_share",
                  "positive_rate_real", "positive_rate_anon"):
            if isinstance(d, dict) and k in d:
                out[k] = d[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="export/adapters")
    ap.add_argument("--zip", action="store_true", help="also write one archive")
    ap.add_argument("--prune", action="store_true",
                    help="DELETE checkpoint-*/ from the source dirs afterwards")
    ns = ap.parse_args()

    src, out = Path(ns.checkpoints), Path(ns.out)
    if not src.is_dir():
        raise SystemExit(f"{src} does not exist — nothing to export.")
    out.mkdir(parents=True, exist_ok=True)

    dirs = sorted(d for d in src.iterdir()
                  if d.is_dir() and (d / "adapter_config.json").exists())
    if not dirs:
        partial = [d.name for d in src.iterdir()
                   if d.is_dir() and list(d.glob("checkpoint-*"))]
        raise SystemExit(
            "no finished adapters found.\n"
            + (f"  These look interrupted (checkpoint-*/ but no adapter at the "
               f"top level): {', '.join(partial)}\n"
               "  The adapter is only written after trainer.train() returns.\n"
               if partial else "")
            + f"  Looked in {src.resolve()}")

    manifest, before_total, after_total = {}, 0, 0
    print(f"{'adapter':28s} {'source':>10s} {'exported':>10s}  fit")
    print("-" * 72)

    for d in dirs:
        before = dir_size(d)
        dest = out / d.name
        dest.mkdir(parents=True, exist_ok=True)
        for fn in KEEP:
            f = d / fn
            if f.exists():
                shutil.copy2(f, dest / fn)
        after = dir_size(dest)
        before_total += before
        after_total += after

        cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
        summary = {}
        if (d / "train_summary.json").exists():
            summary = json.loads((d / "train_summary.json").read_text(encoding="utf-8"))
        verdict = (summary.get("curve") or {}).get("verdict", "")
        short = ("overfit" if "OVERFIT" in verdict.upper() else
                 "underfit" if "UNDERFIT" in verdict.upper() else
                 "good" if "GOOD" in verdict.upper() else "—")

        manifest[d.name] = {
            "base_model": cfg.get("base_model_name_or_path"),
            "peft_type": cfg.get("peft_type"),
            "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
            "lora_dropout": cfg.get("lora_dropout"),
            "target_modules": sorted(cfg.get("target_modules") or []),
            "trainable_params": summary.get("trainable_params"),
            "n_instances": summary.get("n_instances"),
            "train_runtime_s": summary.get("train_runtime_s"),
            "peak_vram_gb": summary.get("peak_vram_gb"),
            "fit_verdict": verdict,
            "best_eval_loss": (summary.get("curve") or {}).get("best_eval_loss"),
            "metrics": metrics_for(d.name, Path(ns.results)),
            "size_bytes": after,
        }
        print(f"{d.name:28s} {human(before):>10s} {human(after):>10s}  {short}")

    base = {m["base_model"] for m in manifest.values() if m["base_model"]}
    (out / "MANIFEST.json").write_text(json.dumps({
        "note": "LoRA adapters only. The base model is NOT included and is "
                "downloaded from the hub on load.",
        "base_models": sorted(base),
        "n_adapters": len(manifest),
        "adapters": manifest,
    }, indent=2), encoding="utf-8")

    (out / "USAGE.md").write_text(f"""# Reusing these adapters

These are **LoRA adapters**, about 4 MB each. The base model is not here — it is
downloaded from the hub the first time you load it.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "{sorted(base)[0] if base else 'Qwen/Qwen2.5-1.5B-Instruct'}"

tok   = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype="float16", attn_implementation="sdpa", device_map="cuda:0")
model = PeftModel.from_pretrained(model, "export/adapters/{dirs[0].name}")
model.eval()
```

⚠️ **`attn_implementation="sdpa"` is not optional.** fp16 with `eager` returns
NaN on Qwen2.5 — it surfaces as `train_loss=0.0` and `grad_norm=nan`, which looks
like a finished run rather than a broken one.

## Swapping adapters without reloading the base model

```python
model.load_adapter("export/adapters/ch1-YAGO3-10-B", adapter_name="B")
model.set_adapter("B")
```

## Inside this project

```bash
python -m chapter1.run  --dataset YAGO3-10 --condition A --evaluate
python -m chapter1.rank --adapter export/adapters/ch1-YAGO3-10-A \\
                        --dataset YAGO3-10 --condition A --limit 500
```

`MANIFEST.json` records, per adapter: base model, LoRA hyper-parameters, training
runtime, peak VRAM, the fit verdict read off the learning curve, and whatever
accuracies were logged for it.
""", encoding="utf-8")

    print("-" * 72)
    saved = before_total - after_total
    print(f"{'TOTAL':28s} {human(before_total):>10s} {human(after_total):>10s}"
          f"   ({human(saved)} of optimiser state and duplicate vocab dropped)")
    print(f"\nwrote {out}/  ·  MANIFEST.json  ·  USAGE.md")

    if ns.zip:
        arc = shutil.make_archive(str(out.parent / "adapters"), "zip", out)
        print(f"archive: {arc}  ({human(Path(arc).stat().st_size)})")

    if ns.prune:
        freed = 0
        for d in dirs:
            for ck in d.glob("checkpoint-*"):
                if ck.is_dir():
                    freed += dir_size(ck)
                    shutil.rmtree(ck)
        print(f"pruned checkpoint-*/ from {len(dirs)} dirs, freed {human(freed)}")
        print("  (the exported adapters are unaffected — the best model was "
              "already written to the top level)")


if __name__ == "__main__":
    main()
