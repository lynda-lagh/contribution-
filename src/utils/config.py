"""
Config loading, seeding and run identity.

⚠️ The audit found `set_seed` was never called anywhere. TrainingArguments(seed=)
seeds the Trainer, but NOT numpy, NOT python's `random`, and NOT the data
sampling / negative generation that happens before training starts. With three
seeds planned on the headline comparison, that silently breaks the one thing
significance testing depends on.

`load_config()` seeds everything, once, at entry.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import yaml


def set_all_seeds(seed: int) -> None:
    """python · numpy · torch · CUDA · HF. Call once at the start of every run."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        pass


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str = "configs/base.yaml", overrides: dict | None = None,
                seed: int | None = None) -> dict:
    """
    Load, apply overrides, seed everything, and return the config.

    Overrides use dotted keys:  {"peft.method": "mora", "data.train_triples": 3000}
    """
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    for dotted, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value

    if seed is not None:
        cfg["seed"] = seed
    set_all_seeds(cfg["seed"])
    return cfg


def run_id(**parts: Any) -> str:
    """Stable, sortable run identifier: ch2-lora-E123182-T10000-s42"""
    return "-".join(f"{k}{v}" if len(k) <= 2 else str(v) for k, v in parts.items() if v is not None)


def save_result(cfg: dict, name: str, payload: dict) -> Path:
    """
    One JSON per run, with the FULL config embedded.

    Without the config in the file, a results directory becomes unreadable within
    a month -- you cannot tell which run used which batch size.
    """
    d = Path(cfg["output"]["results_dir"])
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    # ★ `env` is stamped on EVERY result, not just where someone remembered to.
    # Chapter 2 spans two peft environments; without this the cross-environment
    # control cannot be checked after the fact, and Kaggle sessions are not stable.
    p.write_text(json.dumps({"config": cfg, "env": env_report(), **payload},
                            indent=2, default=str), encoding="utf-8")
    return p


def load_results(pattern: str, results_dir: str = "results") -> list[dict]:
    """Glob and load -- used by the chapter analysis scripts."""
    out = []
    for f in sorted(Path(results_dir).glob(pattern)):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")) | {"_file": f.name})
        except Exception as e:
            print(f"[utils] skipping {f}: {e}")
    return out


def env_report() -> dict:
    """Record the environment with every run -- Kaggle sessions are not stable."""
    rep: dict[str, Any] = {}
    try:
        import torch
        rep |= {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "n_gpu": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpus": [torch.cuda.get_device_name(i)
                     for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            # T4 = Turing (SM 7.5) -> fp16 only. Recorded because it changes results.
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        }
    except ImportError:
        pass
    for mod in ("transformers", "peft", "trl", "datasets"):
        try:
            rep[mod] = __import__(mod).__version__
        except Exception:
            rep[mod] = None
    rep |= peft_env()
    return rep


def peft_env() -> dict:
    """
    ★ WHICH PEFT ENVIRONMENT IS THIS?

    MoRA and BOFT cannot be installed together. MoRA exists only in the
    `kongds/MoRA` fork, which is pinned to peft 0.9.0; BOFT is in official peft
    and did not exist in 0.9.0. Both occupy the import name `peft`, so pip
    installing one overwrites the other.

    Chapter 2 therefore runs across TWO environments, which is a confound unless
    it is recorded. This stamps every result JSON with the environment that
    produced it, so `scripts/verify_env_control.py` can check afterwards that the
    LoRA arm agrees across both -- and so a reviewer asking "which peft version
    produced this number?" has an answer.
    """
    import inspect
    out = {"peft_env": "unknown", "has_mora": False, "has_boft": False}
    try:
        import peft
        out["has_mora"] = "use_mora" in inspect.signature(peft.LoraConfig.__init__).parameters
        out["has_boft"] = hasattr(peft, "BOFTConfig")
        if out["has_mora"] and not out["has_boft"]:
            out["peft_env"] = "mora-fork"        # kongds/MoRA, peft 0.9.0
        elif out["has_boft"] and not out["has_mora"]:
            out["peft_env"] = "official"         # huggingface/peft
        elif out["has_mora"] and out["has_boft"]:
            out["peft_env"] = "both"             # upstream merged MoRA -> one env is enough
    except Exception:
        pass
    return out


def usable_peft_methods() -> list[str]:
    """Which --peft values this environment can actually run."""
    e = peft_env()
    m = ["lora", "probe", "dpo"]                 # always available
    if e["has_mora"]:
        m.append("mora")
    if e["has_boft"]:
        m.append("boft")
    return m
