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
    # ★ More than one VISIBLE device makes HF Trainer wrap the model in
    # DataParallel, and autocast does not reach DP replicas -- fp32 adapters then
    # meet fp16 base weights. Fail here, loudly, rather than 20 steps in.
    assert n == 1, (
        f"{n} GPUs visible {names}. Trainer will use DataParallel and the fp16 "
        f"training checks below WILL fail with 'mat1 and mat2 must have the same "
        f"dtype'.\nRe-run pinned to one device:\n"
        f"    CUDA_VISIBLE_DEVICES=0 python -m scripts.smoke_test\n"
        f"Two T4s are for two INDEPENDENT jobs, not for splitting one."
    )
    return f"-> {n} GPU {names}, compute capability {cc[0]}.{cc[1]}"


def c_bf16():
    cc = torch.cuda.get_device_capability(0)
    native = cc[0] >= 8                    # bf16 is native from Ampere (SM 8.0)
    reported = torch.cuda.is_bf16_supported()
    if native:
        return "-> bf16 native (Ampere+). You may use bf16."
    # ⚠️ is_bf16_supported() returns True on Turing because it counts emulation.
    # T4 has no bf16 tensor cores; using it would be slow and is not what the
    # frozen config specifies.
    return (f"-> bf16 NOT native (SM {cc[0]}.{cc[1]}, Turing). "
            f"torch reports {reported} because it counts emulation -- IGNORE THAT. "
            f"Use fp16, as configs/base.yaml specifies.")


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


def c_forward_dtype():
    """
    ★ WHICH LOADING CONFIG PRODUCES FINITE LOGITS?

    Qwen2/Qwen2.5 were trained in bf16 and carry activation magnitudes that can
    exceed fp16's max (~65504). On Turing there is no native bf16, so pure fp16
    is the obvious choice -- and it overflows: the forward pass returns NaN, the
    loss reads exactly 0.0 and grad_norm reads nan from the first step.

    This probe is INFERENCE ONLY and takes under a minute. It measures the answer
    rather than assuming it, and prints the setting to put in configs/base.yaml.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.data.prompts import ALPACA_NO_INPUT

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt = ALPACA_NO_INPUT.format(
        instruction="Is this true: Paris is the capital of France?")

    combos = [("fp16", torch.float16, "eager"),
              ("fp16", torch.float16, "sdpa"),
              ("fp32", torch.float32, "eager"),
              ("fp32", torch.float32, "sdpa")]

    rows, good = [], []
    for name, dt, attn in combos:
        try:
            m = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=dt, attn_implementation=attn).cuda().eval()
            with torch.no_grad():
                enc = tok(prompt, return_tensors="pt").to("cuda")
                lg = m(**enc).logits[0, -1, :]
            ok = bool(torch.isfinite(lg).all())
            mx = float(lg.abs().max()) if ok else float("nan")
            vram = torch.cuda.max_memory_allocated() / 1e9
            rows.append((name, attn, ok, mx, vram))
            if ok:
                good.append((name, attn))
            del m
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            rows.append((name, attn, False, float("nan"), float("nan")))
            print(f"       {name}/{attn}: {type(e).__name__}: {str(e)[:60]}")

    print()
    print(f"       {'dtype':6s} {'attn':6s} {'finite':>7s} {'max|logit|':>11s} {'VRAM GB':>8s}")
    for name, attn, ok, mx, vram in rows:
        print(f"       {name:6s} {attn:6s} {str(ok):>7s} "
              f"{mx:11.1f} {vram:8.2f}")

    assert good, (
        "NO loading configuration produced finite logits. Something is wrong "
        "beyond dtype -- check the model download."
    )
    best = good[0]
    return (f"-> finite in {len(good)}/4 configs; use dtype={best[0]}, "
            f"attn_implementation={best[1]}")


def _tiny_train(method: str, steps: int = 20):
    """20 steps on 8 fake examples -- checks the method trains and loss is finite."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from src.train.sft import DynamicCausalCollator, attach_peft, make_tokenize_fn

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Matches configs/base.yaml: fp16 weights + SDPA attention.
    # ⚠️ attn="eager" here returns NaN in fp16 -- see the dtype probe above.
    # The fp32 adapters (from _cast_trainable_to_fp32) meet the fp16 base under
    # autocast, which is why this must run on a SINGLE GPU.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa")
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

    # ★ "not NaN" IS NOT ENOUGH.
    #
    # The original assertion was `loss == loss`, which only rejects NaN. MoRA
    # once passed it while reporting train_loss = 0.0000 and grad_norm = nan at
    # every logging step -- i.e. the run had learned nothing and the check said
    # OK. A causal-LM loss of exactly 0.0 is not achievable on real text in 20
    # steps; it means every label was masked, or fp16 collapsed the gradients.
    assert loss is not None, f"{method}: no train_loss reported"
    assert loss == loss, f"{method}: loss is NaN (fp16 instability)"
    assert loss > 1e-6, (
        f"{method}: train_loss is {loss} -- the model did not learn. "
        "Exactly-zero loss means all labels were masked (-100) or fp16 "
        "gradients collapsed. Check _cast_trainable_to_fp32 ran."
    )
    assert loss < 100, f"{method}: train_loss {loss} is implausibly large"

    # every trainable parameter must be fp32, or GradScaler cannot unscale
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and p.dtype == torch.float16]
    assert not bad, (f"{method}: {len(bad)} trainable params still fp16 "
                     f"(e.g. {bad[0]}) -- GradScaler will refuse to unscale them")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vram = torch.cuda.max_memory_allocated() / 1e9
    del model, tr
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return (f"-> loss={loss:.4f} (>0, finite), {n_train/1e6:.2f}M trainable, "
            f"peak VRAM {vram:.2f} GB")


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
    tok.padding_side = "left"          # the function forces "right" internally
    # fp32 for scoring: this is inference, so the memory is affordable and there
    # is no reason to risk fp16 overflow in the measurement Chapter 1 rests on.
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
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
    check("forward-pass dtype probe", c_forward_dtype)
    check("LoRA trains (fp16)", c_lora)
    check("MoRA trains (fp16)", c_mora)
    check("BOFT trains (fp16)", c_boft)
    check("logit scoring", c_logit_scoring)

    print("\n" + "=" * 62)
    bad = [n for n, ok, _ in results if not ok]
    if not bad:
        print("ALL PASSED -- safe to start Chapter 1.")
        sys.exit(0)

    print(f"FAILED: {bad}\n")

    # ---- the one failure that is EXPECTED and is not a bug ----------------
    mora_ok = any(n == "MoRA trains (fp16)" and ok for n, ok, _ in results)
    boft_bad = "BOFT trains (fp16)" in bad
    if mora_ok and boft_bad:
        print("-" * 62)
        print("★ MoRA works and BOFT does not. This is the KNOWN CONFLICT, not a bug")
        print("  in your code:")
        print()
        print("    peft-mora is a FORK of peft 0.9.0. Installing it overwrites")
        print("    official peft, and BOFT did not exist in 0.9.0.")
        print("    The two CANNOT share one environment.")
        print()
        print("  Do NOT try to fix this. Split the work by session:")
        print()
        print("    session A (official peft):  pip install -U peft")
        print("                                --peft lora   --peft boft")
        print("    session B (the fork):       pip install git+https://github.com/"
              "kongds/MoRA.git#subdirectory=peft-mora")
        print("                                --peft mora")
        print()
        print("  ★ Run --peft lora in BOTH sessions. If the two LoRA numbers agree,")
        print("    the peft version is not a confound and the arms are comparable.")
        print("    That control costs one extra run and it is what makes the")
        print("    three-way comparison defensible.")
        print("-" * 62)
        remaining = [b for b in bad if b != "BOFT trains (fp16)"]
        if not remaining:
            print("\nNothing else failed -> you may proceed with LoRA + MoRA now.")
            sys.exit(0)
        print(f"\nStill to fix: {remaining}")

    sys.exit(1)
