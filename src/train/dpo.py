"""
CHAPTER 2, PHASE 2 -- Direct Preference Optimisation.

Verified gap: preference optimisation applied to KGC is 0 of 188 papers. All nine
DPO/RLHF keyword hits in the corpus are related-work prose, background, or a model
NAME (`dolphin-mistral-dpo`).

Why it belongs here rather than as a separate contribution
----------------------------------------------------------
MoRA/LoRA/BOFT are PARAMETERISATIONS; DPO is an OBJECTIVE. They are orthogonal
axes of the same question, so DPO is a column in the grid, not a fourth chapter.

And it is the test of Chapter 1's own diagnosis:

    SFT teaches "say this."
    DPO teaches "prefer this over that near-miss."

If Chapter 1 shows tuning mostly installs FORMAT, then DPO is precisely the
objective that should install DISCRIMINATION instead.

The data is already there
-------------------------
KG-LLM emits one negative per positive (`instructions_WN11.py`), so the training
set is ALREADY a (chosen, rejected) pair structure. But its negatives are
`random.choice(all_entities)` -- uniformly random, therefore mostly type-violating
and trivially separable. Replacing them with KGE near-misses and type-consistent
corruptions is the contribution; the pairing scaffolding is free.

★ Two outcome measures, so DPO has two chances to show an effect:
      accuracy            and      type-2 hallucination rate
  Flat accuracy with a lower type-2 rate is arguably the BETTER result for a
  quality-oriented thesis -- fewer plausible-but-wrong predictions at no cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.loaders import KG, Triple
from ..data.negatives import make_negatives
from ..data.prompts import ALPACA_NO_INPUT, NO, YES, PromptConfig, triple_classification_instruction
from .sft import assert_single_gpu, load_dtype, resolve_report_to


def build_preference_pairs(kg: KG, triples: list[Triple], strategy: str = "type_consistent",
                           seed: int = 42, cfg: PromptConfig | None = None,
                           out_path: str | None = None) -> list[dict]:
    """
    TRL-format preference pairs:  {"prompt", "chosen", "rejected"}

    We build TWO kinds, because they teach different things:

      (a) answer preference -- same prompt, chosen "Yes...", rejected "No..."
          teaches the model to commit to the correct verdict.

      (b) entity preference -- gold triple vs HARD-NEGATIVE triple, both asked as
          "Is this true: ...?", chosen "Yes" for the gold and "No" for the negative.
          ★ This is where hard negatives matter: a type-consistent or KGE-retrieved
          near-miss forces a discrimination that a random corruption never does.
    """
    cfg = cfg or PromptConfig()
    negs = make_negatives(triples, kg, strategy=strategy, seed=seed)

    pairs: list[dict] = []
    for pos, neg in zip(triples, negs):
        p_instr = triple_classification_instruction(pos, kg, cfg)
        n_instr = triple_classification_instruction(neg, kg, cfg)

        pairs.append({"prompt": ALPACA_NO_INPUT.format(instruction=p_instr),
                      "chosen": YES, "rejected": NO, "kind": "positive"})
        pairs.append({"prompt": ALPACA_NO_INPUT.format(instruction=n_instr),
                      "chosen": NO, "rejected": YES, "kind": "hard_negative",
                      "negative_strategy": strategy})

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(pairs, indent=1), encoding="utf-8")
        print(f"[dpo] {len(pairs)} preference pairs "
              f"({len(triples)} triples, strategy={strategy}) -> {out_path}")
    return pairs


def train_dpo(cfg: dict, sft_adapter: str, pairs: list[dict], output_dir: str,
              beta: float = 0.1, run_name: str = "dpo") -> dict:
    """
    DPO on top of an EXISTING SFT adapter -- Phase 2, one run, not a fresh grid.

    beta : KL strength. 0.1 is the usual default; lower drifts further from the
           SFT policy. ⚠️ DPO is more hyperparameter-sensitive than SFT -- budget
           a week, and keep SFT-only as a complete fallback.
    """
    from datasets import Dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    mcfg, tcfg = cfg["model"], cfg["train"]

    tok = AutoTokenizer.from_pretrained(sft_adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    assert_single_gpu()   # DataParallel breaks autocast for fp32 adapters
    base = AutoModelForCausalLM.from_pretrained(
        # dtype from config (fp16), attention MUST be sdpa -- eager returns NaN
        # in fp16, and DPO's implicit reward is a difference of log-probs, so one
        # NaN silently poisons the entire objective.
        mcfg["name"], dtype=load_dtype(cfg),
        attn_implementation=mcfg.get("attn_implementation", "sdpa"))
    # start from the SFT policy and keep training the same adapter
    model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)
    model.config.use_cache = False
    if tcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    ds = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")}
                            for p in pairs]).shuffle(seed=cfg["seed"])
    split = ds.train_test_split(test_size=min(500, len(ds) // 10), seed=cfg["seed"])

    args = DPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        beta=beta,
        per_device_train_batch_size=max(1, tcfg["micro_batch_size"] // 2),  # DPO holds 2 seqs
        gradient_accumulation_steps=tcfg["grad_accum_steps"] * 2,           # same effective batch
        num_train_epochs=1,                                                 # Phase 2 is short
        learning_rate=5e-6,                                                 # << SFT's 3e-4
        warmup_steps=50,
        fp16=True,
        logging_steps=tcfg.get("logging_steps", 20),
        eval_strategy="steps",
        eval_steps=tcfg.get("eval_steps", 250),
        save_strategy="steps",
        save_steps=tcfg.get("save_steps", 250),
        save_total_limit=2,
        max_length=cfg["tokenizer"]["cutoff_len"],
        max_prompt_length=cfg["tokenizer"]["cutoff_len"] - 32,
        report_to=resolve_report_to(tcfg),
        seed=cfg["seed"],
        torch_compile=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # PEFT: the disabled adapter IS the reference
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tok,
    )
    result = trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)     # adapter only
    tok.save_pretrained(output_dir)

    summary = {
        "run_name": run_name, "objective": "dpo", "beta": beta,
        "init_from": sft_adapter,
        "n_pairs": len(pairs),
        "negative_strategy": pairs[0].get("negative_strategy", "mixed") if pairs else None,
        "train_runtime_s": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9
                         if torch.cuda.is_available() else None),
        "output_dir": output_dir,
        "evaluate_with": ["accuracy", "type2_hallucination_rate"],
        "note": ("A flat accuracy with a LOWER type-2 rate is the quality result: "
                 "fewer plausible-but-wrong predictions at no cost."),
    }
    Path(output_dir, "dpo_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
