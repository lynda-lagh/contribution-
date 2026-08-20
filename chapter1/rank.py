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

    # ★★ BUG FIX. render() consults pv.types ONLY. A typed CONDITION (C, D, E,
    #    G) carries its switch in cond.types, not in the prompt variant, so P0
    #    rendered NO tag at all: every typed arm was ranked on untyped prompts
    #    while its adapter had been trained on typed ones. Same train/test
    #    mismatch as the condition-S bug, one level up.
    #
    #    chapter1/data.py folded the condition switch into the variant when the
    #    identical bug hit the builder (see build_condition); rank.py never did.
    #
    #    Evidence it was live, from the 2026-08-20 run: untuned G came out
    #    BYTE-IDENTICAL to untuned A (MRR 0.6072, H@1 0.4720, MR 4.7180) and
    #    untuned C identical to untuned B (MRR 0.0844, MR 27.2280). If the tags
    #    had been rendered those pairs could not have matched to four decimals.
    pv = PROMPTS[prompt_id]
    if types is not None and not pv.types:
        from dataclasses import replace
        pv = replace(pv, types=True)
    demos = demo_pool(kg) if pv.demonstrations else None

    # ── P5-P7 context, built ONCE ───────────────────────────────────────────
    needs_ctx = pv.relation_desc or pv.neighbours or pv.paths
    index = ctx_rng = None
    if needs_ctx:
        from .context import GraphIndex, assert_no_leak
        index = GraphIndex(kg)
        ctx_rng = random.Random(seed)
        if pv.neighbours:
            # ★ prove the guard before scoring 25,000 prompts with it
            print(f"[context] leak check: {assert_no_leak(kg, index)}")

    # true tails per (h, r) -> the filtered setting
    #
    # ★ BUG FIX. This scanned kg.train ONLY, while the paper states "filtered
    #   against train u valid u test" and src.data.loaders.all_true() exists to
    #   do exactly that. Under-filtering lets a genuinely true tail that lives
    #   in valid/test be sampled as a distractor, where it can outrank the gold
    #   and be scored as an error. The bias is conservative -- it depresses MRR
    #   in every arm -- but the protocol was not the one reported.
    known: dict[tuple[str, str], set[str]] = {}
    for t in (*kg.train, *getattr(kg, "valid", ()), *kg.test):
        known.setdefault((t.head, t.relation), set()).add(t.tail)

    # ✋ prove the tag reaches the prompt before spending 25,000 forward passes
    if types is not None:
        probe = render(Triple(queries[0].head, queries[0].relation, ents[0], None),
                       kg, pv, types, None)
        if "[" not in probe:
            raise SystemExit(
                "\n\u2717 typed condition, but the rendered prompt carries no tag:\n"
                f"    {probe[:120]}\n"
                "  The adapter was trained WITH tags; ranking it without them is a\n"
                "  train/test mismatch. Check pv.types and build_types().")
        print(f"[rank] typed prompt confirmed: {probe[:90]}")

    from src.utils.progress import eta_note, track
    eta_note(len(queries) * n_way, 0.012, "forward passes")

    out = []
    n_ctx = 0                    # queries that actually received a context block
    for qi, q in enumerate(track(queries, f"ranking {n_way}-way",
                                 total=len(queries), unit="query")):
        cands = sample_candidates(
            q.tail, ents, n_way, rng,
            filter_out=known.get((q.head, q.relation), set()) - {q.tail})

        # the head-side context is identical for all 50 candidates, so build
        # it once per query rather than once per prompt
        head_ctx = ""
        if needs_ctx:
            from .context import describe_relation, neighbour_block, path_block
            bits = []
            if pv.relation_desc:
                bits.append(describe_relation(q.relation, kg))
            if pv.neighbours:
                bits.append(neighbour_block(index, kg, q.head, q.relation,
                                            pv.neighbours, ctx_rng,
                                            gold=q.tail))
            head_ctx = " ".join(b for b in bits if b)
            n_ctx += bool(head_ctx)

        prompts = []
        for c in cands:
            ctx = head_ctx
            if pv.paths:                       # path context is per-candidate
                from .context import path_block
                pb = path_block(index, kg, q.head, c, q.relation, pv.paths)
                ctx = (ctx + " " + pb).strip() if pb else ctx
            prompts.append(ALPACA_NO_INPUT.format(instruction=render(
                Triple(q.head, q.relation, c, None), kg, pv, types,
                demos.get(q.relation) if demos else None,
                context=ctx or None)))

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
            # ★ the model's actual ANSWER LIST, not just its winner. Costs a few
            #   hundred bytes per query and is the only way to show what the
            #   model does rather than assert it -- KG-LLM's Table VI is
            #   qualitative and is one of the most-read parts of that paper.
            #   Surface forms are stored too, because under anonymisation the
            #   ID is unreadable and under permutation it is misleading.
            "top5": [{"id": cands[i],
                      "text": kg.ent2txt.get(cands[i], cands[i]),
                      "score": float(scores[i])} for i in order[:5]],
            "gold_text": kg.ent2txt.get(q.tail, q.tail),
            "head_text": kg.ent2txt.get(q.head, q.head),
            "relation_text": kg.rel2txt.get(q.relation, q.relation),
        })

    # ★ COVERAGE GUARD. A context variant with an empty block IS P0. If most
    #   queries get nothing, "P7 makes no difference" is a statement about
    #   path availability, not about whether paths help — and the two read
    #   identically in a results table.
    if needs_ctx:
        cov = n_ctx / max(1, len(queries))
        print(f"[context] {prompt_id}: {n_ctx}/{len(queries)} queries got a "
              f"non-empty block ({cov:.1%})")
        if cov < 0.5:
            print(f"[context] ⚠️ under half the queries have any context, so "
                  f"{prompt_id} is mostly IDENTICAL to P0. Report this "
                  f"coverage next to the metric or the null is uninterpretable.")
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

    # ✋ A local path that does not exist is not an error to transformers: it
    #   falls back to treating the string as a HuggingFace repo id, hits the
    #   Hub, and dies with a 401 that names a repo nobody ever created. That
    #   error says "Unauthorized" when the real cause is "wrong directory".
    #   Fail here instead, and say where we looked.
    if adapter is not None:
        from pathlib import Path
        ad = Path(adapter)
        if not ad.is_dir():
            import os
            raise SystemExit(
                f"\n\u2717 adapter directory not found: {adapter}\n"
                f"  resolved to : {ad.resolve()}\n"
                f"  working dir : {os.getcwd()}\n"
                "  Pass an ABSOLUTE path, or cd to the directory that contains\n"
                "  checkpoints/. Nothing was downloaded and nothing was scored.")
        if not (ad / "adapter_config.json").exists():
            raise SystemExit(
                f"\n\u2717 {adapter} exists but holds no adapter_config.json.\n"
                f"  contents: {sorted(p.name for p in ad.iterdir())[:8]}\n"
                "  This is not a PEFT checkpoint directory.")

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
    ap.add_argument("--require-semantic", action="store_true",
                    help="★ refuse to rank a typed condition with INDUCED tags. "
                         "Pass this whenever the adapter was trained with it.")
    ns = ap.parse_args()

    from src.utils.config import load_config, save_result
    cfg = load_config(ns.config)
    cond = CONDITIONS[ns.condition]

    kg = load_kg(ns.dataset, cfg["data"]["root"])
    if cond.anonymise:
        kg = anonymise(kg)

    # ★ BUG FIX. This branch was missing, so `--condition S` ranked on the
    #   REAL graph: the S adapter (trained on a deranged world) was scored on
    #   undamaged names. That is a train/test MISMATCH, not the permuted-name
    #   control, and it silently corrupted m(S) and therefore the A->S
    #   "binding" term. The seed must match chapter1/data.py's builder so the
    #   evaluation graph carries the SAME derangement the model trained on.
    elif getattr(cond, "shuffle", False):
        from src.data.loaders import shuffle_surface_forms
        kg = shuffle_surface_forms(kg, seed=cfg["seed"])

    types = None
    if cond.types or PROMPTS[ns.prompt].types:
        # ★ BUG FIX. This called src.routing.types.entity_types -> INDUCED tags
        #   ([playsFor::tail]), while chapter1/data.py TRAINED condition C on
        #   EXOGENOUS semantic classes ([football_team]) via build_types. The
        #   adapter was therefore ranked with a tag inventory it had never seen:
        #   the same train/test mismatch as the condition-S bug, one level up,
        #   and --require-semantic could not reach here to stop it.
        #   build_types is the single source of truth for what a tag IS.
        from .data import build_types
        types = build_types(kg, cond, ns.dataset, cfg["data"]["root"],
                            require_semantic=ns.require_semantic)

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
    surface = ("anon" if cond.anonymise
               else "PERMUTED" if getattr(cond, "shuffle", False) else "real")
    print(f"[rank] {len(queries)} queries · {ns.n_way}-way · condition {ns.condition} "
          f"· prompt {ns.prompt} · surface form: {surface}")
    # The ID is identical in every condition -- permutation and anonymisation
    # happen in ent2txt, at render time. Printing the ID here can never detect a
    # failed permutation: it shows a real name under S even on a correct run.
    # Print what the model will actually read.
    _h = kg.ent2txt.get(queries[0].head, queries[0].head)
    print(f"[rank] example query head: id={queries[0].head!r} -> rendered as {_h!r}")
    if surface == "PERMUTED" and _h == queries[0].head.replace("_", " "):
        raise SystemExit(
            "\n\u2717 condition S, but the rendered head equals its own id.\n"
            "  The permutation did not reach ent2txt; S is not a matched control.")
    print(f"[rank] test labels: {'±1 present' if labelled else 'absent -> all positives'} "
          f"| {len(kg.ent2txt):,} entities in the candidate pool")

    model, tok = load(cfg["model"]["name"], ns.adapter)
    ranks = rank_queries(model, tok, kg, queries, ns.n_way, cfg["seed"], ns.prompt, types)
    m = metrics(ranks)

    tag = ns.tag or f"ch1rank-{ns.dataset}-{ns.condition}-{ns.prompt}"
    # ★ was ranks[:200], which threw away 60% of the per-query records and
    #   widened every bootstrap CI by ~1.6x for no saving worth having.
    save_result(cfg, tag, {"metrics": m, "condition": ns.condition,
                           "prompt": ns.prompt, "adapter": ns.adapter,
                           "surface_form": surface, "ranks": ranks})

    print("\n" + "=" * 60)
    print(f"{tag}   ({m['protocol']})")
    print("=" * 60)
    for k in ("hits@1", "hits@3", "hits@10", "MRR", "mean_rank"):
        print(f"  {k:10s} {m[k]:.4f}")
    print(f"\n  ⚠️ {m['warning']}")


if __name__ == "__main__":
    main()
