"""
ONE sampling loop, TWO uses.

    sample k answers (top-p 0.95, top-k 20 -- GS-KGC's exact setting)
        |-> Hits@K                     partial comparability restored
        |-> sampling disagreement      = UNCERTAINTY

GS-KGC already runs this loop to compute Hits@3; nobody reuses it for confidence.
Doing so costs nothing and gives an uncertainty signal that is inherently
explainable -- "7 of 10 samples agreed" is a statement a human can act on, which a
softmax margin is not.

⚠️ Note what repeated sampling does and does not give you: a SET, not a RANKING.
Hits@K becomes computable; MRR does not. State that rather than hiding it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch


@dataclass
class SampledPrediction:
    prompt: str
    samples: list[str]
    counts: Counter = field(default_factory=Counter)

    # ---- prediction -------------------------------------------------------
    @property
    def top(self) -> str:
        return self.counts.most_common(1)[0][0] if self.counts else ""

    def top_k(self, k: int = 3) -> list[str]:
        return [a for a, _ in self.counts.most_common(k)]

    # ---- uncertainty ------------------------------------------------------
    @property
    def agreement(self) -> float:
        """Share of samples agreeing with the modal answer. 1.0 = unanimous."""
        n = sum(self.counts.values())
        return self.counts.most_common(1)[0][1] / n if n else 0.0

    @property
    def disagreement(self) -> float:
        return 1.0 - self.agreement

    @property
    def n_distinct(self) -> int:
        return len(self.counts)

    @property
    def entropy(self) -> float:
        """Normalised Shannon entropy over sampled answers -- 0 = certain."""
        import math
        n = sum(self.counts.values())
        if n == 0 or len(self.counts) <= 1:
            return 0.0
        h = -sum((c / n) * math.log2(c / n) for c in self.counts.values())
        return h / math.log2(len(self.counts)) if len(self.counts) > 1 else 0.0

    def explanation(self) -> str:
        """★ Trace 2 -- the abstention reason, ready for the review queue."""
        n = sum(self.counts.values())
        top, cnt = self.counts.most_common(1)[0]
        others = self.counts.most_common(4)[1:]
        s = f"{cnt} of {n} samples agreed on '{top}'"
        if others:
            s += "; competing: " + ", ".join(f"'{a}' ({c})" for a, c in others)
        return s


def _normalise(text: str) -> str:
    return " ".join(text.strip().strip(".").split()).lower()


@torch.no_grad()
def sample_predictions(model, tokenizer, prompts: list[str], k: int = 10,
                       top_p: float = 0.95, top_k: int = 20,
                       max_new_tokens: int = 16, batch_size: int = 4,
                       device: str = "cuda") -> list[SampledPrediction]:
    """k stochastic samples per prompt. Defaults are GS-KGC's."""
    model.eval()
    out: list[SampledPrediction] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        gen = model.generate(**enc, do_sample=True, top_p=top_p, top_k=top_k,
                             num_return_sequences=k, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        plen = enc["input_ids"].shape[1]
        for b, prompt in enumerate(batch):
            texts = [tokenizer.decode(gen[b * k + j][plen:], skip_special_tokens=True)
                     for j in range(k)]
            out.append(SampledPrediction(prompt, texts,
                                         Counter(_normalise(t) for t in texts)))
    return out


@torch.no_grad()
def greedy_predictions(model, tokenizer, prompts: list[str],
                       max_new_tokens: int = 16, batch_size: int = 8,
                       device: str = "cuda") -> list[str]:
    """Deterministic decode -- the Hits@1 number, comparable with KG-LLM."""
    model.eval()
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        plen = enc["input_ids"].shape[1]
        out.extend(tokenizer.decode(g[plen:], skip_special_tokens=True) for g in gen)
    return out


def hits_at_k(preds: list[SampledPrediction], golds: list[str], k: int = 3) -> float:
    """Hits@K from the SET of sampled answers (GS-KGC's approach)."""
    hit = sum(_normalise(g) in [_normalise(a) for a in p.top_k(k)]
              for p, g in zip(preds, golds))
    return hit / len(golds) if golds else 0.0


def confidence_sources(preds: list[SampledPrediction],
                       logprobs: list[float] | None = None,
                       p_true: list[float] | None = None) -> dict[str, list[float]]:
    """
    The three confidence sources Chapter 4 compares. Nobody has compared them:
    GLR contrasts P(True) with Monte-Carlo dropout in PROSE only, and reports no
    calibration metric of any kind.
    """
    out = {
        "sampling_agreement": [p.agreement for p in preds],
        "sampling_entropy": [1.0 - p.entropy for p in preds],   # higher = more certain
    }
    if logprobs is not None:
        out["sequence_logprob"] = logprobs
    if p_true is not None:
        out["p_true"] = p_true
    return out
