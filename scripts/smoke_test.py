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

OK, FAIL, WARN, SKIP = "  [OK]  ", "  [FAIL]", "  [WARN]", "  [n/a] "
# status is one of: "ok" | "expected" | "fail"
results: list[tuple[str, str, str]] = []


MAX_ERR_CHARS = 600   # safety net: no single failure should dump a screen-filling wall of text


def _shorten(text: str, limit: int = MAX_ERR_CHARS) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n       ... [{len(text) - limit} more chars truncated]"


def _configured_butterfly():
    """The boft_n_butterfly_factor the real runs will use, or None if unreadable."""
    try:
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
        return cfg["peft"].get("boft_n_butterfly_factor")
    except Exception:
        return None


def classify(name: str, err: str):
    """
    Is this failure the EXPECTED one for this environment?

    Two checks are *designed* to fail depending on which session you are in --
    printing them in red with a full traceback trains you to ignore red, which is
    exactly the habit that lets a real failure slip through. Recognise them by
    signature and report them calmly instead.

    Returns (one-line headline, indented detail) or None if it is a real failure.
    """
    if name == "MoRA trains (fp16)" and "has no `use_mora`" in err:
        return ("not available in this environment -- expected, MoRA lives in Session B",
                "The official peft has no `use_mora`; the peft-mora fork provides it.\n"
                "       Run MoRA in a FRESH session with the fork installed (ENV='mora').\n"
                "       Nothing to fix -- continue with lora / boft / probe here.")

    if name == "BOFT trains (fp16)" and "downgraded boft_n_butterfly_factor" in err:
        if _configured_butterfly() == 1:
            return ("CUDA kernel unavailable, running the factor=1 fallback you pinned",
                    "peft's fbd_cuda extension will not build here, so BOFT falls back to\n"
                    "       boft_n_butterfly_factor=1 -- which configs/base.yaml already pins\n"
                    "       DELIBERATELY. This is the fallback you chose, working as intended.\n"
                    "       Report that BOFT ran with a single butterfly stage.\n"
                    "       To get factor=2, run the notebook's BOFT CUDA kernel patch cell.")
        # config asks for >1 but peft is silently dropping it: that IS a real failure
        return None

    return None


def check(name: str, fn) -> None:
    try:
        msg = fn() or ""
        print(f"{OK} {name} {msg}")
        results.append((name, "ok", msg))
    except Exception as e:
        full = str(e)
        known = classify(name, full)
        if known:
            # expected -> one calm line, no traceback, no red
            print(f"{SKIP} {name} -> {known[0]}")
            results.append((name, "expected", full))
        else:
            print(f"{FAIL} {name}: {_shorten(full)}")
            traceback.print_exc(limit=1)
            results.append((name, "fail", full))


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
    # OPTIONAL: bitsandbytes is needed only for the 7B QLoRA confirmation run,
    # which is a nice-to-have, not part of the main grid. Never fail the whole
    # smoke test over it -- that would block Chapters 1-4 for a bonus experiment.
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return ("-> NOT INSTALLED (optional). Only the 7B QLoRA confirmation run "
                "needs it; Chapters 1-4 do not. `pip install bitsandbytes` if you "
                "want that run.")
    cc = torch.cuda.get_device_capability(0)
    if cc[0] * 10 + cc[1] < 75:
        return "-> WARNING: compute capability < 7.5, 4-bit unreliable (P100). Use T4."
    return "-> ok (needed for the 7B QLoRA confirmation runs)"


def c_stack_versions():
    """
    ★ The library stack must MATCH ACROSS SESSIONS.

    Chapter 2 spans two peft environments. If one session also has a different
    transformers, the LoRA control cannot isolate peft -- you would be comparing
    two whole stacks. `pip install -U peft` is what causes this: newer peft pulls
    a newer transformers, which then breaks torchao.
    """
    import transformers
    tv = transformers.__version__
    major = int(tv.split(".")[0])
    note = ""
    if major >= 5:
        note = ("  ⚠️ transformers 5.x -- almost certainly pulled in by "
                "`pip install -U peft`. Pin it: pip install 'transformers==4.57.6'")
    try:
        import torchao
        ao = torchao.__version__
        # transformers refuses to import if torchao is present but older than
        # 0.16 -- and torchao is NOT needed for LoRA/BOFT/MoRA training at all.
        # Removing it is safer than chasing a compatible version.
        if tuple(int(x) for x in ao.split(".")[:2]) < (0, 16):
            note += (f"\n       ⚠️ torchao {ao} < 0.16 will make LoRA fail with "
                     f"'Found an incompatible version of torchao'.\n"
                     f"          It is not needed here -> `pip uninstall -y torchao`")
    except Exception:
        ao = "absent (good -- nothing here needs it)"
    return f"-> transformers {tv} | torchao {ao}{note}"


# ---------------------------------------------------------------- libs
def c_transformers():
    import transformers
    from transformers import TrainingArguments
    import inspect
    params = inspect.signature(TrainingArguments.__init__).parameters
    key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    return f"-> transformers {transformers.__version__}, uses `{key}=`"


def c_peft():
    """Identify the environment and print the run plan that belongs to it."""
    import peft
    from src.utils.config import peft_env, usable_peft_methods

    e = peft_env()
    methods = usable_peft_methods()

    print()
    print(f"       peft {peft.__version__}  ->  environment: {e['peft_env'].upper()}")
    print(f"       runnable here: {', '.join(methods)}")
    print()
    if e["peft_env"] == "mora-fork":
        print("       SESSION B (the fork). Run in THIS session:")
        print("           --peft lora      <-- also the cross-env CONTROL")
        print("           --peft mora")
        print("       Then start a FRESH session, set ENV='official', and run")
        print("       --peft boft and --peft probe there.")
    elif e["peft_env"] == "official":
        print("       SESSION A (official peft). Run in THIS session:")
        print("           --peft lora      <-- also the cross-env CONTROL")
        print("           --peft boft")
        print("           --peft probe")
        print("       Then a FRESH session with ENV='mora' for --peft mora.")
    elif e["peft_env"] == "both":
        print("       ★ This peft has BOTH. Upstream must have merged MoRA --")
        print("         no session split needed. Verify before relying on it.")
    else:
        print("       ⚠️ Could not identify the environment. Check peft installed.")
    print()
    print("       Every result JSON is stamped with this environment.")
    print("       After both sessions:  python -m scripts.verify_env_control")

    return (f"-> {e['peft_env']} | MoRA={'YES' if e['has_mora'] else 'NO'}"
            f" | BOFT={'YES' if e['has_boft'] else 'NO'}")


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

    # ★ Read the REAL config, do not hardcode. The smoke test previously built its
    # own dict, so `boft_n_butterfly_factor` fell back to the default of 2 and BOFT
    # failed here even after base.yaml was set to 1 -- the test was checking a
    # configuration that no run would ever use.
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))["peft"]
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
    check("stack versions", c_stack_versions)
    check("peft / MoRA / BOFT", c_peft)
    check("tokenizer + pad_token", c_tokenizer)
    check("forward-pass dtype probe", c_forward_dtype)
    check("LoRA trains (fp16)", c_lora)
    check("MoRA trains (fp16)", c_mora)
    check("BOFT trains (fp16)", c_boft)
    check("logit scoring", c_logit_scoring)

    # ---------------------------------------------------------------- summary
    # Three outcomes, not two. "expected" means a check that is DESIGNED to fail
    # in this environment (MoRA outside the fork; BOFT without its CUDA kernel
    # when base.yaml already pins the fallback). Those are reported calmly so
    # that anything printed as FAIL is always worth reading.
    passed   = [n for n, s, _ in results if s == "ok"]
    expected = [(n, m) for n, s, m in results if s == "expected"]
    failed   = [n for n, s, _ in results if s == "fail"]

    print("\n" + "=" * 62)
    print(f"{len(passed)} passed"
          + (f" | {len(expected)} expected-n/a" if expected else "")
          + (f" | {len(failed)} FAILED" if failed else ""))

    if expected:
        print()
        for n, msg in expected:
            detail = classify(n, msg)[1]
            print(f"  [n/a] {n}\n       {detail}")

    if failed:
        print("\n" + "-" * 62)
        print(f"NEEDS ATTENTION: {failed}")
        print("Scroll up for the traceback of each. Do not start a long run until")
        print("these are resolved -- that is what this script exists to prevent.")
        sys.exit(1)

    print()
    if expected:
        print("Nothing unexpected -- safe to proceed in THIS session.")
    else:
        print("ALL PASSED -- safe to start Chapter 1.")
    sys.exit(0)
