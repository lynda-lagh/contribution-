"""
Supervised fine-tuning -- model-agnostic fork of KG-LLM's lora_finetune_*.py.

Changes from the original, each deliberate:

  * AutoModel/AutoTokenizer            (their LlamaTokenizer assert fails on modern
                                        transformers; delete it)
  * eval_strategy=                     (evaluation_strategy= is renamed)
  * DYNAMIC padding                    (theirs pads every sequence to cutoff_len=50,
                                        which both wastes compute and truncates)
  * effective batch 32, not 128        (128 was tuned on 112k-316k triples; at 10k
                                        it yields 156 steps/epoch, so warmup=100
                                        would be 32% of training)
  * torch_compile off by default       (unreliable on T4)
  * pluggable PEFT: lora | mora | boft
  * adapter-only checkpointing         (KEPT from KG-LLM -- their state_dict
                                        monkey-patch; ~20-100 MB, fits /kaggle/working)

Loss is masked to the response only -- also KG-LLM's behaviour, and it is the
reason Chapter 1 exists: the model is trained to emit the fixed string
"Yes, this is true." / "No, this is not true.". That is format compliance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.prompts import ALPACA_NO_INPUT


# ------------------------------------------------------------------ collator
@dataclass
class DynamicCausalCollator:
    """
    Pads to the longest sequence IN THE BATCH (not to cutoff_len).
    This is what lets us set a generous cutoff without paying for it, and it
    removes any need to tune sequence length.
    """
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ------------------------------------------------------------------ tokenise
def make_tokenize_fn(tokenizer, cutoff_len: int):
    def fn(rec: dict) -> dict:
        prompt = ALPACA_NO_INPUT.format(instruction=rec["instruction"])
        n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        full = tokenizer(prompt + rec["output"], truncation=True,
                         max_length=cutoff_len)["input_ids"]
        n_prompt = min(n_prompt, len(full))
        return {
            "input_ids": full,
            # loss on the RESPONSE only -- KG-LLM's behaviour, kept
            "labels": [-100] * n_prompt + full[n_prompt:],
            "attention_mask": [1] * len(full),
        }
    return fn


# ------------------------------------------------------------------ PEFT
def _cast_trainable_to_fp32(model):
    """
    ★ REQUIRED WHENEVER THE BASE MODEL IS LOADED IN fp16.

    The base model is fp16, so PEFT creates the adapter tensors in fp16 too.
    `Trainer(fp16=True)` then wraps the step in a GradScaler, and
    `scaler.unscale_()` REFUSES fp16 gradients:

        ValueError: Attempting to unscale FP16 gradients.

    Mixed precision means fp16 *activations* with fp32 *master weights*. The
    trainable parameters must therefore be fp32; only the frozen backbone stays
    fp16. This is what `prepare_model_for_kbit_training` does for QLoRA, and it
    is just as necessary for plain fp16.

    ⚠️ MoRA fails SILENTLY without this instead of raising: loss collapses to
    exactly 0.0 with grad_norm=nan, which looks like a finished run. Any check
    that only asks "is the loss non-NaN" will pass it.

    Layer norms are also promoted: they accumulate sums over the hidden
    dimension and are the classic fp16 overflow site.
    """
    import torch as _t
    n = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(_t.float32)
            n += 1
    for name, module in model.named_modules():
        if "norm" in name.lower() and hasattr(module, "weight"):
            if module.weight is not None and module.weight.dtype == _t.float16:
                module.weight.data = module.weight.data.to(_t.float32)
    if n == 0:
        raise RuntimeError(
            "attach_peft produced a model with NO trainable parameters. "
            "Check `target_modules` matches this architecture "
            "(Qwen2.5 uses q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj)."
        )
    return model


def attach_peft(model, method: str, cfg: dict):
    """
    Returns a PEFT-wrapped model whose trainable parameters are fp32-safe.

    Keep `target_modules` identical across methods -- it is the control that makes
    LoRA / MoRA / BOFT comparable at all.
    """
    method = method.lower()

    if method == "lora":
        from peft import LoraConfig, get_peft_model
        return _cast_trainable_to_fp32(get_peft_model(model, LoraConfig(
            r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
            target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM")))

    if method == "mora":
        # peft-mora is a FORK of peft: pip install git+https://github.com/kongds/MoRA.git#subdirectory=peft-mora
        # MoRA's premise is exactly Chapter 2's hypothesis: "the low-rank updating
        # mechanism in LoRA may limit the ability of LLMs to learn and memorize
        # new knowledge", and it is "comparable on other tasks".
        from peft import LoraConfig, get_peft_model
        kw = dict(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                  target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM")
        try:
            return _cast_trainable_to_fp32(get_peft_model(model, LoraConfig(
                **kw, use_mora=True, mora_type=cfg.get("mora_type", 6))))
        except TypeError as e:
            raise RuntimeError(
                "Installed `peft` has no `use_mora`. Install the fork:\n"
                "  pip install git+https://github.com/kongds/MoRA.git#subdirectory=peft-mora\n"
                "and verify it does not break BOFT."
            ) from e

    if method == "boft":
        # Official HF PEFT. Multiplicative ORTHOGONAL updates (vs LoRA's additive
        # low-rank) -- tests the knowledge-PRESERVATION hypothesis, which is why
        # the forgetting measurement is what makes BOFT worth including.
        try:
            from peft import BOFTConfig, get_peft_model
        except ImportError as e:
            raise RuntimeError(
                "`BOFTConfig` is missing from the installed peft.\n"
                "This is almost always the peft-mora fork: it is built on peft 0.9.0, "
                "which predates BOFT, and installing it OVERWRITES official peft.\n"
                "  -> MoRA and BOFT cannot coexist in one environment.\n"
                "  -> Run them in SEPARATE sessions (see DEPLOY.md), and run LoRA in "
                "both as a cross-environment control."
            ) from e
        return _cast_trainable_to_fp32(get_peft_model(model, BOFTConfig(
            boft_block_size=cfg.get("boft_block_size", 4),
            boft_n_butterfly_factor=cfg.get("boft_n_butterfly_factor", 2),
            target_modules=cfg["target_modules"],
            boft_dropout=cfg["dropout"], bias="none", task_type="CAUSAL_LM")))

    raise ValueError(f"unknown PEFT method: {method}")


# ------------------------------------------------------------------ main
def train_sft(cfg: dict, data_dir: str, output_dir: str, run_name: str = "run") -> dict:
    from datasets import Dataset

    mcfg, tcfg, pcfg, tokcfg = cfg["model"], cfg["train"], cfg["peft"], cfg["tokenizer"]

    tokenizer = AutoTokenizer.from_pretrained(
        mcfg["name"], trust_remote_code=mcfg.get("trust_remote_code", False))
    if tokenizer.pad_token is None:                      # Qwen has no pad token
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"],
        torch_dtype=torch.float16,                       # T4: fp16, NO bf16
        attn_implementation=mcfg.get("attn_implementation", "eager"),
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )
    model.config.use_cache = False
    if tcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = attach_peft(model, pcfg["method"], pcfg)
    model.print_trainable_parameters()
    model = model.cuda()

    records = json.loads(Path(data_dir, "train_instructions.json").read_text(encoding="utf-8"))
    ds = Dataset.from_list(records).shuffle(seed=cfg["seed"])
    ds = ds.map(make_tokenize_fn(tokenizer, tokcfg["cutoff_len"]),
                remove_columns=ds.column_names)

    val_size = min(cfg["data"]["val_size"], max(1, len(ds) // 10))
    split = ds.train_test_split(test_size=val_size, seed=cfg["seed"])

    args = transformers.TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=tcfg["micro_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum_steps"],   # -> effective 32
        num_train_epochs=tcfg["epochs"],
        learning_rate=float(tcfg["learning_rate"]),
        warmup_steps=tcfg["warmup_steps"],
        fp16=True,
        optim=tcfg.get("optim", "adamw_torch"),
        logging_steps=tcfg.get("logging_steps", 20),
        eval_strategy="steps",                                  # renamed from evaluation_strategy
        eval_steps=tcfg.get("eval_steps", 250),
        save_strategy="steps",
        save_steps=tcfg.get("save_steps", 250),
        save_total_limit=tcfg.get("save_total_limit", 2),
        load_best_model_at_end=False,
        torch_compile=bool(tcfg.get("torch_compile", False)),   # off on T4
        report_to=tcfg.get("report_to", "none"),
        seed=cfg["seed"],
        remove_unused_columns=False,
    )

    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DynamicCausalCollator(tokenizer.pad_token_id),
    )

    result = trainer.train()

    # ADAPTER-ONLY checkpoint (KG-LLM's trick, kept): ~20-100 MB, not the base model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "run_name": run_name,
        "peft_method": pcfg["method"],
        "model": mcfg["name"],
        "n_instances": len(records),
        "train_runtime_s": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "steps": result.metrics.get("train_steps_per_second"),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9
                         if torch.cuda.is_available() else None),
        "output_dir": output_dir,
    }
    Path(output_dir, "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
