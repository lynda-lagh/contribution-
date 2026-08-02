"""
Catastrophic forgetting -- what makes BOFT worth including.

The gap
-------
BiGTex ASSERTS it: "the [pretrained weights] remain untouched, while small,
trainable low-rank matrices are introduced ... This avoids catastrophic
forgetting and significantly reduces the computational cost."

An assertion, in two papers, with no measurement. Nobody in 188 papers measures
forgetting in KGC tuning.

Why it makes BOFT meaningful
----------------------------
BOFT's premise is knowledge PRESERVATION via multiplicative orthogonal updates.
Accuracy on FB15k-237 cannot reveal that. Without this measurement, BOFT is a
decorative third method; with it, BOFT tests a hypothesis.

And it links to Chapter 1: if tuning mostly installs FORMAT while degrading
general ability, that is a cost nobody has priced.

Cost: inference only, on checkpoints that already exist.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Short, domain-neutral probes. Deliberately NOT knowledge-graph text: we are
# asking what the model lost OUTSIDE the tuning distribution.
GENERIC_TEXT = [
    "The capital of France is Paris, a city known for its museums and architecture.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "In machine learning, a loss function measures the difference between the "
    "predicted output and the true target value.",
    "Photosynthesis is the process by which plants convert light energy into "
    "chemical energy stored in glucose.",
    "The Second World War ended in 1945 following the surrender of the Axis powers.",
    "A prime number is a natural number greater than one that has no positive "
    "divisors other than one and itself.",
    "Shakespeare wrote both tragedies and comedies during the late sixteenth and "
    "early seventeenth centuries.",
    "The mitochondrion is often described as the powerhouse of the cell because it "
    "produces adenosine triphosphate.",
]


@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str], device: str = "cuda",
               max_length: int = 256) -> float:
    """Token-weighted perplexity -- lower is better, higher after tuning = forgetting."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=max_length).to(device)
        n = enc["input_ids"].shape[1]
        loss = model(**enc, labels=enc["input_ids"]).loss
        total_nll += loss.item() * n
        total_tok += n
    return math.exp(total_nll / max(total_tok, 1))


def _load(base: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(adapter or base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, attn_implementation="eager").cuda()
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    return m.eval(), tok


def measure_forgetting(base_model: str, adapter_dirs: dict[str, str],
                       texts: list[str] | None = None,
                       out_path: str = "results/forgetting.json") -> dict:
    """
    adapter_dirs : {"lora": "checkpoints/ch2-lora-...", "mora": ..., "boft": ...}

    Reports perplexity BEFORE (base) and AFTER each adaptation. The prediction
    BOFT makes is that its multiplicative orthogonal update degrades general
    ability least.
    """
    texts = texts or GENERIC_TEXT

    model, tok = _load(base_model, None)
    base_ppl = perplexity(model, tok, texts)
    del model; torch.cuda.empty_cache()
    print(f"[forgetting] base perplexity: {base_ppl:.3f}")

    out = {"base_model": base_model, "base_perplexity": base_ppl,
           "n_probe_texts": len(texts), "methods": {}}

    for name, adapter in adapter_dirs.items():
        if not Path(adapter).exists():
            print(f"[forgetting] skip {name}: {adapter} not found")
            continue
        model, tok = _load(base_model, adapter)
        ppl = perplexity(model, tok, texts)
        del model; torch.cuda.empty_cache()

        delta = ppl - base_ppl
        out["methods"][name] = {
            "adapter": adapter,
            "perplexity": ppl,
            "delta": delta,
            "relative_increase": delta / base_ppl,
            "interpretation": ("preserved" if delta <= 0.05 * base_ppl
                               else "degraded" if delta <= 0.25 * base_ppl
                               else "substantial forgetting"),
        }
        print(f"[forgetting] {name:>6}: ppl {ppl:.3f}  "
              f"delta {delta:+.3f} ({delta/base_ppl:+.1%})  "
              f"{out['methods'][name]['interpretation']}")

    if out["methods"]:
        best = min(out["methods"].items(), key=lambda kv: kv[1]["relative_increase"])
        out["least_forgetting"] = best[0]
        out["claim_under_test"] = ("BiGTex asserts LoRA avoids catastrophic forgetting; "
                                   "BOFT claims orthogonal updates preserve pretrained "
                                   "knowledge. Neither is measured anywhere in the corpus.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"[forgetting] saved -> {out_path}")
    return out
