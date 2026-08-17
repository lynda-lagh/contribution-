"""
Build the prompts for one (policy, budget) pair.

    python -m chapter3.data --dataset WN18RR-ind --policy S0_uniform --budget 120
    python -m chapter3.data --dataset WN18RR-ind --all --budget 120 --direction both
    python -m chapter3.data --dataset WN18RR-ind --train-mixed        # the shared model

TWO KINDS OF OUTPUT
-------------------
    train_instructions.json   for fine-tuning
    queries.json              prefix / suffix straddling the predicted slot
    allocations.json          ★ what each policy DECIDED and WHY — the audit trail

★ DIRECTION. `--direction both` builds `(h, r, ?)` and `(?, r, t)`. CATS and
  RealKGC both report both directions; a one-directional table invites the
  reviewer to assume the easy side was chosen. For head prediction the context
  describes the TAIL — the entity actually given — because describing the head
  would be describing the answer.

★ `--train-mixed` builds the training set for the SHARED model: each example gets
  a RANDOM budget and a RANDOM policy. That is P28's context-corruption idea
  repurposed — the model sees many different context subsets, so no policy is out
  of distribution at evaluation time. One training run then serves every
  (policy, budget) cell.

⚠️ Every policy at a given budget must spend the SAME budget on the SAME
   candidate blocks. The only difference is priority order. `--verify` checks
   that the prompts actually differ between policies: if they are byte-identical
   the comparison measures nothing, which is exactly how Chapter 1's condition C
   silently became condition B.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .budget import allocate, token_counter
from .policies import BUDGETS, POLICIES
from .sources import GraphIndex, assert_no_leak, candidate_blocks


def build_one(kg, queries, policy, budget, rel_desc, types, count,
              n_neighbours=5, n_demos=3, direction="tail", index=None,
              check_leaks=True):
    """
    One (policy, budget, direction) cell -> prompts + the allocation audit trail.

    ★ PREFIX / SUFFIX, not a finished string. The slot being predicted sits
      BETWEEN them, so one representation serves all three tasks:

        tail      "<ctx> Is this true: dog _hypernym "  +CAND+  "?"
        head      "<ctx> Is this true: "                +CAND+  " _hypernym animal?"
        relation  "<ctx> Is this true: dog "            +CAND+  " animal?"

      The old code baked the tail into `prefix`, which is why head prediction
      was impossible without rebuilding everything.

    ⚠️ For head prediction the CONTEXT DESCRIBES THE TAIL, because the tail is
       the entity we are given. Allocating context about the head would be
       describing the answer — a leak, and an easy one to miss.
    """
    from src.data.prompts import YES

    # ★ built ONCE. Building it per query is the 468x bug from Chapter 1, and
    #   at 160k calls over WN18RR's 86k training triples it costs hours of CPU
    #   before a single forward pass.
    if index is None:
        index = GraphIndex(kg)

    records, qrows, allocs = [], [], []
    for t in queries:
        # the entity we are GIVEN, and therefore the one context describes
        anchor = t.head if direction == "tail" else t.tail
        gold = t.tail if direction == "tail" else t.head
        blocks = candidate_blocks(kg, anchor, t.relation, rel_desc, types,
                                  count, n_neighbours, n_demos, index=index,
                                  exclude=(t.head, t.relation, t.tail))
        # ⚠️ independent re-check: the inference graph makes the gold reachable,
        #    so this guard is not redundant with the exclusion above
        if check_leaks:
            assert_no_leak(blocks, kg, anchor, t.relation, gold)
        a = allocate(blocks, budget, policy, count=count)

        ctx = " ".join(a.text_blocks())
        lead = f"{ctx}\nIs this true: " if ctx else "Is this true: "
        h = kg.ent2txt.get(t.head, t.head)
        r = kg.rel2txt.get(t.relation, t.relation)
        tl = kg.ent2txt.get(t.tail, t.tail)

        if direction == "tail":
            prefix, suffix, gold_surface = f"{lead}{h} {r} ", "?", tl
        else:
            prefix, suffix, gold_surface = lead, f" {r} {tl}?", h

        # the relation-prediction slot, built from the same allocation so the
        # confusion matrix is measured at the SAME cost as the ranking table
        rel_prefix, rel_suffix = f"{lead}{h} ", f" {tl}?"

        records.append({"instruction": prefix + gold_surface + suffix,
                        "input": "", "output": YES})
        qrows.append({"head": t.head, "relation": t.relation, "tail": t.tail,
                      "direction": direction,
                      "prefix": prefix, "suffix": suffix,
                      "rel_prefix": rel_prefix, "rel_suffix": rel_suffix,
                      "context_tokens": a.spent})
        allocs.append({"query": f"{t.head}|{t.relation}|{t.tail}",
                       "direction": direction, "anchor": anchor,
                       "kept": [{"kind": b.kind, "target": b.target,
                                 "tokens": b.tokens, "text": b.text} for b in a.kept],
                       "dropped": [{"kind": b.kind, "target": b.target,
                                    "tokens": b.tokens} for b in a.dropped],
                       "spent": a.spent, "budget": a.budget,
                       "utilisation": a.utilisation,
                       "reasons": a.reasons,
                       "tokens_by_kind": a.summary()["tokens_by_kind"]})
    return records, qrows, allocs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="WN18RR-ind")
    ap.add_argument("--root", default="data")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--policy", nargs="+", default=["S0_uniform"],
                    choices=list(POLICIES), metavar="P")
    ap.add_argument("--budget", type=int, nargs="+", default=[120])
    ap.add_argument("--all", action="store_true", help="every policy")
    ap.add_argument("--direction", nargs="+", default=["tail"],
                    choices=("tail", "head", "both"),
                    help="★ CATS and RealKGC report both; one direction looks cherry-picked")
    ap.add_argument("--train-mixed", action="store_true",
                    help="★ training set with random policy+budget per example")
    ap.add_argument("--no-inference-graph", action="store_true",
                    help="⚠️ neighbours from train only. In the inductive setting "
                         "this leaves unseen entities with NO neighbours and "
                         "collapses the ladder — for ablation only")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true", default=True)
    ns = ap.parse_args()

    from src.data.loaders import load_kg
    from src.routing.types import entity_types
    from src.utils.config import load_config

    cfg = load_config(ns.config)
    kg = load_kg(ns.dataset, ns.root)
    count = token_counter(cfg["model"]["name"])

    rd = Path(ns.root, ns.dataset, "relation_descriptions.json")
    rel_desc = json.loads(rd.read_text(encoding="utf-8")) if rd.exists() else {}
    if not rel_desc:
        print("⚠️ no relation_descriptions.json — that block will be absent.\n"
              "   python -m chapter3.sources --dataset {} --generate".format(ns.dataset))
    types = entity_types(kg, method="induced")

    queries = kg.test[: ns.limit]
    print(f"[data] {ns.dataset}: {len(queries):,} queries · "
          f"{len(kg.rel2txt)} relations · {len(rel_desc)} relation descriptions")

    # ★ ONE index for the whole run — see GraphIndex's docstring for the two
    #   bugs this replaces (no inductive neighbours, and the per-call rebuild)
    index = GraphIndex(kg, use_inference_graph=not ns.no_inference_graph)
    with_nb = sum(1 for t in queries[:200]
                  if index.neighbours_of(t.head, (t.head, t.relation, t.tail), 5))
    print(f"[data] graph index: {index.n_support:,} inference-graph facts added "
          f"for unseen entities")
    print(f"[data] {with_nb}/{min(200, len(queries))} sampled queries have a "
          f"neighbours block ({with_nb/max(1,min(200,len(queries))):.0%})")
    if with_nb == 0:
        print("  ✋ NO query has neighbours. S1/S2/S4/S5 all discriminate on that")
        print("     block, so the ladder cannot differ from the baseline and any")
        print("     'specificity does not pay' conclusion would be an artefact.")
        print("     Check the split: inductive test entities need an inference graph.")
        raise SystemExit(1)

    out_root = Path(ns.root, ns.dataset, "built")
    out_root.mkdir(parents=True, exist_ok=True)

    dirs = ("tail", "head") if "both" in ns.direction else tuple(ns.direction)

    # ---------------------------------------------------- the shared model set
    if ns.train_mixed:
        rng = random.Random(ns.seed)
        pols = [p for k, p in POLICIES.items() if k != "ORACLE"]
        train_q = kg.train[: ns.limit * 5]
        recs = []
        for t in train_q:
            p = rng.choice(pols)
            b = rng.choice([x for x in BUDGETS if x > 0])
            # ★ mix the DIRECTION too, or the shared model only ever learns to
            #   complete tails and head prediction is out of distribution — which
            #   would show up as a direction effect that is really a training gap
            d_ = rng.choice(dirs) if len(dirs) > 1 else dirs[0]
            r, _, _ = build_one(kg, [t], p, b, rel_desc, types, count,
                                direction=d_, index=index)
            recs.extend(r)
        d = out_root / "mixed"
        d.mkdir(parents=True, exist_ok=True)
        (d / "train_instructions.json").write_text(json.dumps(recs, indent=1),
                                                   encoding="utf-8")
        print(f"[data] mixed training set: {len(recs):,} examples "
              f"(random policy + budget + direction per example) -> {d}")
        print("   ★ P28's context-corruption idea repurposed: the model sees many")
        print("     context subsets, so no policy is out of distribution later.")
        return

    # ------------------------------------------------------ the evaluation set
    pols = list(POLICIES) if ns.all else ns.policy
    seen_prompts = {}
    for direction in dirs:
        for budget in ns.budget:
            for pid in pols:
                recs, qrows, allocs = build_one(kg, queries, POLICIES[pid], budget,
                                                rel_desc, types, count,
                                                direction=direction, index=index)
                d = out_root / f"{pid}_B{budget}_{direction}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "train_instructions.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
                (d / "queries.json").write_text(json.dumps(qrows, indent=1), encoding="utf-8")
                (d / "allocations.json").write_text(json.dumps(allocs, indent=1), encoding="utf-8")

                spent = [q["context_tokens"] for q in qrows]
                mean = sum(spent) / max(1, len(spent))
                print(f"  {pid:14s} B={budget:<4d} {direction:4s} mean {mean:6.1f} tok "
                      f"({mean/budget if budget else 0:5.1%} of budget)  -> {d.name}")
                seen_prompts.setdefault((direction, budget), {})[pid] = tuple(
                    q["prefix"] + q["suffix"] for q in qrows[:200])

    # ⚠️ the guard that Chapter 1 needed and did not have.
    #
    # ★ It reports, it does not abort. A collision at B=30 says nothing about
    #   B=120: if the smallest budget can only afford one block then every
    #   policy keeps that block and the prompts coincide BY ARITHMETIC. Killing
    #   the whole build there would discard the budgets that do discriminate.
    #   We therefore mark each cell informative or not, and fail only if NONE is.
    if ns.verify:
        print(f"\n{'='*74}\n[guard] DO THE POLICIES ACTUALLY DIFFER?\n{'='*74}")
        informative, dead = [], []
        for (direction, budget), per in sorted(seen_prompts.items()):
            if budget == 0:
                print(f"  B={budget:<4d} {direction:4s}  — floor, identical by definition")
                continue
            uniq = len(set(per.values()))
            # ★ A cell is informative when the policies SEPARATE, not when every
            #   pair happens to differ. Two policies keyed on features that are
            #   constant on this graph will always coincide; that is a finding
            #   about the graph (pre-registered as "S3 ~= S1"), not a broken run.
            #   The cell only dies when ALL policies collapse to one prompt.
            ok = uniq > 1
            (informative if ok else dead).append((direction, budget))
            print(f"  B={budget:<4d} {direction:4s}  {uniq}/{len(per)} distinct "
                  f"{'✓ informative' if ok else '✗ ALL IDENTICAL — measures nothing'}")
            groups = {}
            for k, v in per.items():
                groups.setdefault(v, []).append(k)
            for v, ks in groups.items():
                if len(ks) > 1:
                    print(f"        ≡ {', '.join(ks)}   (same feature on this graph)")

        # ---- block inventory: what could be allocated, and what fits -------
        t = queries[0]
        blocks = candidate_blocks(kg, t.head, t.relation, rel_desc, types,
                                  count, index=index,
                                  exclude=(t.head, t.relation, t.tail))
        print(f"\n  BLOCK INVENTORY for one query ({t.head}, {t.relation}, ?):")
        if not blocks:
            print("    (none at all — nothing can be allocated)")
        for b in sorted(blocks, key=lambda b: b.tokens):
            fits = ", ".join(str(x) for x in ns.budget if b.tokens <= x) or "none"
            print(f"    {b.kind:22s} {b.tokens:>5d} tok   fits in B = {fits}")

        # ---- ★ FEATURE VARIANCE: why two policies can never disagree -------
        #  A policy routes on a feature. If that feature is constant across the
        #  graph the policy has nothing to decide, and any two policies keyed on
        #  constant features produce the same ordering. This is the same defect
        #  that made Chapter 1's first quality-band router flat by construction
        #  (95.7% of entities in one band), so it is measured here rather than
        #  discovered after a GPU sweep.
        import statistics as _st
        feats = {"has_description": [], "label_words": [], "type_entropy": [],
                 "degree": []}
        for q in queries[:400]:
            bl = candidate_blocks(kg, q.head, q.relation, rel_desc, types, count,
                                  index=index,
                                  exclude=(q.head, q.relation, q.tail))
            if not bl:
                continue
            m = bl[0].meta
            for k in feats:
                if m.get(k) is not None:
                    feats[k].append(m[k])

        print(f"\n  FEATURE VARIANCE across {len(queries[:400])} queries "
              f"(a constant feature cannot route):")
        POLICY_OF = {"has_description": "S1_property", "label_words": "S3_quality",
                     "type_entropy": "S2_type", "degree": "(diagnostic only)"}
        for k, vals in feats.items():
            if not vals:
                print(f"    {k:18s} — absent"); continue
            if all(isinstance(v, bool) for v in vals):
                rate = sum(vals) / len(vals)
                flat = rate > 0.97 or rate < 0.03
                print(f"    {k:18s} {rate:6.1%} true      "
                      f"{'✗ CONSTANT -> ' + POLICY_OF[k] + ' cannot decide' if flat else '✓ splits'}")
            else:
                lo, hi = min(vals), max(vals)
                sd = _st.pstdev(vals) if len(vals) > 1 else 0.0
                flat = sd < 1e-6 or lo == hi
                print(f"    {k:18s} min {lo:<8.3g} max {hi:<8.3g} sd {sd:<8.3g}"
                      f"{'  ✗ CONSTANT -> ' + POLICY_OF[k] + ' cannot decide' if flat else '  ✓ varies'}")
        print(f"\n    ★ Two policies keyed on constant features WILL coincide. Report")
        print(f"      that as a property of the graph — it is pre-registered in")
        print(f"      policies.INTERPRETATION as \"S3 ~= S1\" — not as a failed run.")

        print(f"\n  {len(informative)} informative cell(s), {len(dead)} not.")
        if not informative:
            raise SystemExit(
                "✋ NO budget produces distinct prompts, so nothing in this grid can\n"
                "   be measured. Either the blocks are too few (check the inventory\n"
                "   above) or every budget is too small to force a choice. Raise the\n"
                "   budgets, or add block kinds, before spending GPU.")
        print(f"  ★ report results for the informative budgets and state that the")
        print(f"    others could not discriminate — that is a finding about cost,")
        print(f"    not a failed run.")
        (out_root / "informative_cells.json").write_text(
            json.dumps({"informative": [[d, b] for d, b in informative],
                        "not_informative": [[d, b] for d, b in dead]}, indent=1),
            encoding="utf-8")


if __name__ == "__main__":
    main()
