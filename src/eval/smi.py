"""
Sliced Mutual Information (SMI) -- Chapter 1's second instrument.

FLAME uses this to argue that "fine-tuning primarily aligns representations
rather than injecting knowledge from the KG training set": frozen intermediate
representations reach fine-tuned-level SMI values.

We reuse the instrument to answer a question FLAME does not ask -- what fine-tuning
DOES install -- by measuring SMI before and after tuning, across PEFT methods, and
across |E|.

Definition
----------
    SMI(X;Y) = E_{theta ~ Unif(S^{d-1})} [ I(theta^T X ; Y) ]

X = intermediate-layer representations (R^d), Y = discrete KGC labels.
Monte-Carlo estimate over m random unit directions; each 1-D mutual information
is estimated with a k-NN (KSG/Ross) estimator.

Why sliced rather than plain MI: MI estimation degrades badly in high dimensions;
projecting to 1-D keeps the estimator in its reliable regime.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif


def random_directions(d: int, m: int, seed: int = 42) -> np.ndarray:
    """m unit vectors sampled uniformly from the (d-1)-sphere."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(m, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sliced_mutual_information(X: np.ndarray, y: np.ndarray, n_projections: int = 128,
                              n_neighbors: int = 3, seed: int = 42) -> dict:
    """
    X : (n, d) representations
    y : (n,)   discrete labels
    Returns mean SMI plus spread, so a difference between conditions can be
    judged against projection variance rather than eyeballed.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n, d = X.shape
    if n < n_neighbors + 1:
        raise ValueError(f"need > {n_neighbors} samples, got {n}")

    # standardise so no single dimension dominates the projections
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    dirs = random_directions(d, n_projections, seed)
    proj = X @ dirs.T                                    # (n, m)

    mis = mutual_info_classif(proj, y, discrete_features=False,
                              n_neighbors=n_neighbors, random_state=seed)
    return {
        "smi": float(mis.mean()),
        "smi_std": float(mis.std()),
        "smi_p95": float(np.percentile(mis, 95)),
        "n_projections": n_projections,
        "n_samples": n,
        "dim": d,
    }


@torch.no_grad()
def extract_representations(model, tokenizer, prompts: list[str], layer: int = -1,
                            pooling: str = "last", batch_size: int = 8,
                            device: str = "cuda", max_length: int = 512) -> np.ndarray:
    """
    Intermediate-layer hidden states.

    layer   : index into output_hidden_states. -1 = final, or e.g. len//2 for middle.
              FLAME uses INTERMEDIATE layers -- the final layer is specialised for
              next-token prediction and is not where task information is richest.
    pooling : "last" (last non-pad token) or "mean" (masked mean over tokens)
    """
    model.eval()
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states[layer]   # (B,T,d)
        mask = enc["attention_mask"]
        if pooling == "mean":
            m = mask.unsqueeze(-1).to(hs.dtype)
            pooled = (hs * m).sum(1) / m.sum(1).clamp_min(1e-9)
        else:
            idx = mask.sum(1) - 1
            pooled = hs[torch.arange(hs.size(0), device=hs.device), idx, :]
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def smi_across_layers(model, tokenizer, prompts: list[str], labels: list[int],
                      layers: list[int] | None = None, **kw) -> dict:
    """
    SMI profile over depth. FLAME's point is that *intermediate* layers carry the
    task information -- so report the profile, not a single number.
    """
    n_layers = model.config.num_hidden_layers
    layers = layers or [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
    y = np.asarray(labels)
    out = {}
    for L in layers:
        X = extract_representations(model, tokenizer, prompts, layer=L, **kw)
        out[f"layer_{L}"] = sliced_mutual_information(X, y)
        print(f"  [smi] layer {L:>3}: SMI = {out[f'layer_{L}']['smi']:.5f}")
    best = max(out.items(), key=lambda kv: kv[1]["smi"])
    out["best_layer"] = best[0]
    out["best_smi"] = best[1]["smi"]
    return out


def compare(smi_untuned: dict, smi_tuned: dict) -> dict:
    """
    The Chapter 1 claim, quantified.

    delta ~ 0  -> fine-tuning did NOT add task-relevant information to the
                  representation, so what it installed lies downstream (output
                  format). This reproduces FLAME's finding and, combined with the
                  logit decomposition, says what fine-tuning did instead.
    delta > 0  -> fine-tuning genuinely enriched the representation.
    """
    a, b = smi_untuned["best_smi"], smi_tuned["best_smi"]
    return {
        "smi_untuned": a,
        "smi_tuned": b,
        "delta": b - a,
        "relative_change": (b - a) / a if a else None,
        "interpretation": ("representation unchanged -> tuning installed FORMAT"
                           if abs(b - a) < 0.1 * max(a, 1e-9)
                           else "representation enriched -> tuning installed KNOWLEDGE"),
    }
