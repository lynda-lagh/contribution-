"""
★ TURN THE TRIPLE CLASSIFIER INTO A RANKER — real KGC, no retraining.

    python -m chapter1.rank --adapter checkpoints/ch1-A --dataset WN11

Chapter 1 trains a binary classifier: "Is this true: h r t?" -> Yes / No.
That is a legitimate KGC subtask, but it COMPLETES nothing -- and the thesis is
about enrichment. A jury will ask.

The fix needs no new training. For a query (h, r, ?), score every candidate t by

        P(Yes | h, r, t)

and sort. The model was trained to emit exactly that judgement, so we are using
it as designed. Ranking gives Hits@1 / Hits@3 / Hits@10 -- and, crucially, MRR.

★★ THIS REMOVES A STATED LIMITATION OF THE WHOLE THESIS.
   The spec says repeatedly "MRR is NOT computable: sampling yields a SET, not a
   ranking". True for GENERATIVE decoding. Scoring candidates produces an
   ordering, so MRR is computable after all.

PRECEDENT: KG-BERT ranks entities by its classification score. The reranking
papers in the corpus (KICGPT, ColKGC) build on the same idea. This is a standard
protocol, not an invention.

⚠️ PROTOCOL (R12). Full ranking over WN11's 38,588 entities is ~38k forward
passes PER QUERY -- infeasible. We use N-WAY ranking: the true tail plus N-1
sampled negatives. **50-way Hits@1 is NOT the same quantity as full-ranking
Hits@1.** Every table caption must say "50-way". All comparisons here are
internal (A vs B vs C under an identical protocol), which is what the argument
needs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.data.loaders import KG, Triple, anonymise, load_kg
from src.data.prompts import ALPACA_NO_INPUT

from .conditions import CONDITIONS, PROMPTS
from .data import demo_pool, render

# ⚠️ torch and the scoring helpers are imported INSIDE the functions that need a
# GPU. `sample_candidates` and `metrics` are pure and must stay importable on a
# machine with no torch, so the test suite can run anywhere in ~2 seconds.


def sample_candidates(true_tail: str, all_entities: list[str], n: int,
                      rng: random.Random, filter_out: set[str]) -> list[str]:
    """
    True tail + (n-1) sampled negatives.

    ⚠️ FILTERED setting: candidates that are genuinely true for this (h, r) are
    excluded, otherwise a correct alternative answer counts as an error. This is
    standard and it matters -- unfiltered numbers look worse for no good reason.
    """
    cands = [true_tail]
    seen = {true_tail} | filter_out
    guard = 0
    while len(cands) < n and guard < n * 50:
        e = rng.choice(all_entities)
        guard += 1
        if e not in seen:
            seen.add(e)
            cands.append(e)
    rng.shuffle(cands)
    return cands


def rank_queries(model, tokenizer, kg: KG, queries: list[Triple],
                 n_way: int = 50, seed: int = 42, prompt_id: str = "P0",
                 types: dict | None = None, batch_size: int = 32) -> list[dict]:
    import torch
    from src.infer.scoring import yes_no_probabilities
    torch.set_grad_enabled(False)

    rng = random.Random(seed)
    ents = list(kg.ent2txt)
    pv = PROMPTS[prompt_id]
    demos = demo_pool(kg) if pv.demonstrations else None

    # true tails per (h, r) -> the filtered setting
    known: dict[tuple[str, str], set[str]] = {}
    for t in kg.train:
        known.setdefault((t.head, t.relation), set()).add(t.tail)

    out = []
    for qi, q in enumerate(queries):
        cands = sample_candidates(
            q.tail, ents, n_way, rng,
            filter_out=known.get((q.head, q.relation), set()) - {q.tail})

        prompts = [
            ALPACA_NO_INPUT.format(instruction=render(
                Triple(q.head, q.relation, c, None), kg, pv, types,
                demos.get(q.relation) if demos else None))
            for c in cands]

        probs = yes_no_probabilities(model, tokenizer, prompts, batch_size=batch_size)
        scores = np.array([p_yes for p_yes, _ in probs])

        order = np.argsort(-scores)                    # descending P(Yes)
        ranked = [cands[i] for i in order]
        rank = ranked.index(q.tail) + 1                # 1-based

        out.append({
            "head": q.head, "relation": q.relation, "tail": q.tail,
            "rank": rank, "n_way": n_way,
            "score_true": float(scores[cands.index(q.tail)]),
            "score_top": float(scores[order[0]]),
            # margin = the confidence source Chapter 4 consumes
            "margin": float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0,
            "top1": ranked[0],
        })
        if (qi + 1) % 100 == 0:
            print(f"    {qi+1}/{len(queries)} queries")
    return out


def metrics(ranks: list[dict]) -> dict:
    r = np.array([x["rank"] for x in ranks])
    n = len(r)
    return {
        "n_queries": n,
        "n_way": ranks[0]["n_way"] if ranks else 0,
        "hits@1": float((r <= 1).mean()),
        "hits@3": float((r <= 3).mean()),
        "hits@10": float((r <= 10).mean()),
        "MRR": float((1.0 / r).mean()),          # ★ computable — see module docstring
        "mean_rank": float(r.mean()),
        "protocol": f"{ranks[0]['n_way'] if ranks else 0}-way, filtered",
        "warning": "N-way Hits@K is NOT comparable to full-ranking Hits@K (R12). "
                   "State the protocol in every caption.",
    }


def load(base: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter or base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float32, attn_implementation="sdpa").cuda()
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    return m.eval(), tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--adapter", default=None, help="omit for the untuned baseline")
    ap.add_argument("--dataset", default="WN11")
    ap.add_argument("--condition", default="A", choices=list(CONDITIONS))
    ap.add_argument("--prompt", default="P0", choices=list(PROMPTS))
    ap.add_argument("--n-way", type=int, default=50)
    ap.add_argument("--limit", type=int, default=500,
                    help="queries. 500 x 50 = 25k forward passes ~ 20 min on a T4")
    ap.add_argument("--tag", default=None)
    ns = ap.parse_args()

    from src.utils.config import load_config, save_result
    cfg = load_config(ns.config)
    cond = CONDITIONS[ns.condition]

    kg = load_kg(ns.dataset, cfg["data"]["root"])
    if cond.anonymise:
        kg = anonymise(kg)

    types = None
    if cond.types or PROMPTS[ns.prompt].types:
        from src.routing.types import entity_types
        types = entity_types(kg)

    # ★ Only POSITIVE test triples are valid link-prediction queries.
    #
    # ⚠️ WN11 and FB13 carry ±1 labels, so filter on label == 1.
    #    WN18RR and YAGO3-10 do NOT -- `load_kg` sets label=None for them, since
    #    KG-LLM used those graphs for link/relation prediction rather than triple
    #    classification. Every triple in their test file IS a true fact, so an
    #    unlabelled test set means "all positives", not "no queries".
    #    Without this branch YAGO3-10 would silently yield 0 queries.
    labelled = any(t.label is not None for t in kg.test)
    queries = [t for t in kg.test if (t.label == 1 if labelled else True)][: ns.limit]
    if not queries:
        raise SystemExit(
            f"no positive queries in {ns.dataset}. labelled={labelled}; "
            f"first test triple label={kg.test[0].label if kg.test else 'no test set'}")
    print(f"[rank] {len(queries)} queries · {ns.n_way}-way · condition {ns.condition} "
          f"· prompt {ns.prompt} · {'anon' if cond.anonymise else 'real'}")
    print(f"[rank] test labels: {'±1 present' if labelled else 'absent -> all positives'} "
          f"| {len(kg.ent2txt):,} entities in the candidate pool")

    model, tok = load(cfg["model"]["name"], ns.adapter)
    ranks = rank_queries(model, tok, kg, queries, ns.n_way, cfg["seed"], ns.prompt, types)
    m = metrics(ranks)

    tag = ns.tag or f"ch1rank-{ns.dataset}-{ns.condition}-{ns.prompt}"
    save_result(cfg, tag, {"metrics": m, "condition": ns.condition,
                           "prompt": ns.prompt, "adapter": ns.adapter,
                           "ranks": ranks[:200]})

    print("\n" + "=" * 60)
    print(f"{tag}   ({m['protocol']})")
    print("=" * 60)
    for k in ("hits@1", "hits@3", "hits@10", "MRR", "mean_rank"):
        print(f"  {k:10s} {m[k]:.4f}")
    print(f"\n  ⚠️ {m['warning']}")


if __name__ == "__main__":
    main()
