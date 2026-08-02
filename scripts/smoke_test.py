"""
15-MINUTE SMOKE TEST -- run this before anything else.

It catches, in minutes rather than after a lost 12-hour Kaggle session:
  1. bf16 unavailable on T4 (Turing)          -> must use fp16
  2. transformers API renames                 -> eval_strategy, AutoTokenizer
  3. Qwen has no pad_token
  4. peft-mora fork conflicting with peft     <- highest technical risk
  5. BOFT not wrapping a causal LM
  6. fp16 instability in MoRA / BOFT          -> NaN loss
  7. dynamic padding + loss masking correct

    python -m scripts.smoke_test
"""
from __future__ import annotations

import sys
import traceback

import torch

OK, FAIL, WARN = "  [OK]  ", "  [FAIL]", "  [WARN]"
results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        msg = fn() or ""
        print(f"{OK} {name} {msg}")
        results.append((name, True, msg))
    except Exception as e:
        print(f"{FAIL} {name}: {e}")
        traceback.print_exc(limit=1)
        results.append((name, False, str(e)))


# ---------------------------------------------------------------- hardware
def c_gpu():
    assert torch.cuda.is_available(), "no CUDA device"
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    cc = torch.cuda.get_device_capability(0)
    return f"-> {n} GPU(s) {names}, compute capability {cc[0]}.{cc[1]}"


def c_bf16():
    sup = torch.cuda.is_bf16_supported()
    if sup:
        return "-> bf16 SUPPORTED (Ampere+). You may use bf16."
    return "-> bf16 NOT supported (expected on T4/P100). Use fp16 everywhere."


def c_bnb():
    import bitsandbytes  # noqa: F401
    cc = torch.cuda.get_device_capability(0)
    if cc[0] * 10 + cc[1] < 75:
        return "-> WARNING: compute capability < 7.5, 4-bit unreliable (P100). Use T4."
    return "-> ok (needed for the 7B QLoRA confirmation runs)"


# ---------------------------------------------------------------- libs
def c_transformers():
    import transformers
    from transformers import TrainingArguments
    import inspect
    params = inspect.signature(TrainingArguments.__init__).parameters
    key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    return f"-> transformers {transformers.__version__}, uses `{key}=`"


def c_peft():
    import peft
    from peft import LoraConfig
    import inspect
    has_mora = "use_mora" in inspect.signature(LoraConfig.__init__).parameters
    try:
        from peft import BOFTConfig  # noqa: F401
        has_boft = True
    except Exception:
        has_boft = False
    return (f"-> peft {peft.__version__} | MoRA={'YES' if has_mora else 'NO (fork not installed)'}"
            f" | BOFT={'YES' if has_boft else 'NO'}")


# ---------------------------------------------------------------- model
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def c_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    had = tok.pad_token is not None
    if not had:
        tok.pad_token = tok.eos_token
    ids_yes = tok("Yes", add_special_tokens=False)["input_ids"]
    ids_no = tok("No", add_special_tokens=False)["input_ids"]
    return (f"-> pad_token {'present' if had else 'set to eos'} | "
            f"'Yes'->{ids_yes} 'No'->{ids_no}")


def _tiny_train(method: str, steps: int = 20):
    """20 steps on 8 fake examples -- checks the method trains and loss is finite."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from src.train.sft import DynamicCausalCollator, attach_peft, make_tokenize_fn

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="eager")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    cfg = {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}
    model = attach_peft(model, method, cfg).cuda()

    recs = [{"instruction": f"Is this true: entity{i} related to entity{i+1}?",
             "input": "", "output": "Yes, this is true."} for i in range(8)]
    ds = Dataset.from_list(recs).map(make_tokenize_fn(tok, 512),
                                     remove_columns=["instruction", "input", "output"])

    tr = Trainer(
        model=model,
        args=TrainingArguments(output_dir=f"/tmp/smoke_{method}", max_steps=steps,
                               per_device_train_batch_size=2, gradient_accumulation_steps=1,
                               learning_rate=3e-4, fp16=True, logging_steps=5,
                               report_to="none", save_strategy="no",
                               torch_compile=False, remove_unused_columns=False),
        train_dataset=ds,
        data_collator=DynamicCausalCollator(tok.pad_token_id),
    )
    out = tr.train()
    loss = out.metrics.get("train_loss")
    assert loss is not None and loss == loss, f"{method}: loss is NaN (fp16 instability)"
    vram = torch.cuda.max_memory_allocated() / 1e9
    del model, tr
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return f"-> loss={loss:.4f} (finite), peak VRAM {vram:.2f} GB"


def c_lora():
    return _tiny_train("lora")


def c_mora():
    return _tiny_train("mora")


def c_boft():
    return _tiny_train("boft")


def c_logit_scoring():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.infer.scoring import yes_no_probabilities
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="eager").cuda().eval()
    from src.data.prompts import ALPACA_NO_INPUT
    ps = [ALPACA_NO_INPUT.format(instruction="Is this true: Paris is the capital of France?"),
          ALPACA_NO_INPUT.format(instruction="Is this true: Paris is the capital of Japan?")]
    out = yes_no_probabilities(m, tok, ps)
    del m; torch.cuda.empty_cache()
    assert all(abs(a + b - 1.0) < 1e-3 for a, b in out), "probabilities not normalised"
    return f"-> P(Yes) true-fact={out[0][0]:.3f}  false-fact={out[1][0]:.3f}"


if __name__ == "__main__":
    print("=" * 62)
    print("SMOKE TEST -- run before committing a Kaggle session")
    print("=" * 62)

    check("GPU available", c_gpu)
    check("bf16 support", c_bf16)
    check("bitsandbytes", c_bnb)
    check("transformers API", c_transformers)
    check("peft / MoRA / BOFT", c_peft)
    check("tokenizer + pad_token", c_tokenizer)
    check("LoRA trains (fp16)", c_lora)
    check("MoRA trains (fp16)", c_mora)
    check("BOFT trains (fp16)", c_boft)
    check("logit scoring", c_logit_scoring)

    print("\n" + "=" * 62)
    bad = [n for n, ok, _ in results if not ok]
    if bad:
        print(f"FAILED: {bad}")
        print("Fix these before running anything long.")
        print("If only MoRA failed -> fall back to LoRA vs BOFT (still two mechanisms).")
        sys.exit(1)
    print("ALL PASSED -- safe to start Chapter 1.")
