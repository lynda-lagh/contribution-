"""
CHAPTER 1 — data construction for every condition and prompt variant.

    python -m chapter1.data --condition A --dataset WN11
    python -m chapter1.data --all --dataset WN11

★ ONE builder for the whole grid. Serialization is a CONTROLLED VARIABLE: 16
distinct input representations exist across the corpus and no paper compares two
of them, so anything that is not the variable under test is held byte-identical.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from src.data.loaders import KG, Triple, anonymise, load_kg, shuffle_surface_forms
from src.data.negatives import build_relation_type_index, make_negatives
from src.data.prompts import ALPACA_NO_INPUT, NO, YES
from src.routing.types import entity_types
from src.utils.progress import phase, track

from .conditions import CONDITIONS, PROMPTS, STRUCTURAL_INSTRUCTION, Condition, PromptVariant


# =============================================================================
#  PROMPT RENDERING  — the only place a prompt string is ever produced
# =============================================================================
def render(t: Triple, kg: KG, pv: PromptVariant,
           types: dict[str, str] | None = None,
           demos: list[Triple] | None = None,
           context: str | None = None) -> str:
    """
    KG-LLM's question, verbatim, plus exactly the blocks `pv` switches on.

    P0   Is this true: {h} {r} {t}?
    P1   Is this true: {h} [Person] {r} {t} [Location]?
    P2   Is this true: {h} {r} {t}? Before answering, consider whether …
    P3   both
    P4   Other triples using {r}: … | Is this true: {h} {r} {t}?
    """
    h = kg.ent2txt.get(t.head, t.head)
    r = kg.rel2txt.get(t.relation, t.relation)
    tl = kg.ent2txt.get(t.tail, t.tail)

    if pv.types and types:
        th, tt = types.get(t.head), types.get(t.tail)
        h = f"{h} [{th}]" if th else h
        tl = f"{tl} [{tt}]" if tt else tl

    parts = []
    # ★ P5-P7 context block. Built by chapter1/context.py, which owns the
    #   inverse-relation guard; render() must never build it itself, or the
    #   guard can be bypassed by a new call site.
    if context:
        parts.append(context)
    if pv.demonstrations and demos:
        # ★ RealKGC's mechanism: DEMONSTRATE the relation's type pattern rather
        # than assert it. The model compares the query against real instances.
        ex = "; ".join(f"{kg.ent2txt.get(d.head, d.head)} {r} "
                       f"{kg.ent2txt.get(d.tail, d.tail)}"
                       for d in demos[:pv.demonstrations])
        parts.append(f"Other triples using {r}: {ex}.")

    parts.append(f"Is this true: {h} {r} {tl}?")

    if pv.instruction:
        parts.append(STRUCTURAL_INSTRUCTION)

    return " ".join(parts)


# =============================================================================
#  DEMONSTRATION POOL
# =============================================================================
def build_types(kg: KG, cond, dataset: str, root: str = "data",
                max_other: float = 0.5,
                max_other_semantic: float = 0.10,
                require_semantic: bool = False) -> dict[str, str]:
    """
    ★★ TYPES THAT ACTUALLY CARRY INFORMATION — with a hard guard.

    THE BUG THIS EXISTS TO PREVENT
    ------------------------------
    `entity_types(kg, method="auto")` picks the POS method, which reads a
    part-of-speech marker out of the identifier (`stool_NN_2`, WN18RR style).

        WN11 identifiers look like `__east_indian_1` -- NO POS marker.
        Result: 1 distinct type, OTHER = 100.0% of 38,588 entities.

    Every optional type block then renders nothing, so condition C produced a
    prompt BYTE-IDENTICAL to condition B:

        [B] Is this true: entity712 has instance entity21807?
        [C] Is this true: entity712 has instance entity21807?     <-- same

    C, D, E and G would all have silently run as B, and the ladder would have
    "shown" that types do not help. Anonymisation makes it worse (`entity712`
    carries nothing at all), but the failure is present on the RAW graph too.

    THE FIX
    -------
    Fall back to INDUCED types: an entity's type is the set of relation
    positions it occupies (`_has_instance::head`, `_type_of::tail`, ...).
    Derived from graph structure, which anonymisation PRESERVES -- relations are
    never hidden, only entities. So induced types are identical before and after
    anonymisation, which is exactly what conditions B->C need.

    ⚠️ Say so in the paper. An induced type is "participates in these relations",
    which is weaker than a semantic type like Person/Location. On WN11 it is the
    only type signal available. **FB13 has real semantic types** -- that is why
    the type conditions belong there.
    """
    # ★★ EXOGENOUS FIRST. An induced type is the (relation, position) pair the
    #    entity occupies most often -- computed FROM the edges under test, so
    #    the tag partly restates the question. That circularity is measurable:
    #    a one-line rule scored 62.4% on YAGO3-10 with no model at all.
    #    src.routing.semantic_types supplies types from OUTSIDE the graph
    #    (WordNet supersenses, NELL's ontology), which is what condition C needs
    #    in order to ask "can knowing WHAT a thing is replace knowing its NAME?"
    #    Types are keyed on the entity ID, which anonymise() and
    #    shuffle_surface_forms() both leave untouched -- so C and G see the
    #    identical tag inventory, as the design requires.
    from src.routing.semantic_types import coverage, semantic_types
    try:
        t = semantic_types(kg, dataset, root=root)
        r = coverage(t)
        print(f"[types] {dataset} {cond.id}: EXOGENOUS (semantic) -> "
              f"{r['n_distinct']} distinct, OTHER={r['other_rate']:.1%}, "
              f"largest={r['largest_share']:.1%}")
        # ★ A SEMANTIC source is held to a STRICTER bar than the induced
        #   fallback. max_other=0.5 is the "is this usable at all" gate for
        #   induced tags; an exogenous source that leaves a third of the graph
        #   as OTHER is not the resource the paper describes, and the paper
        #   states the 90% figure. YAGO3-10 measures 0.1%, so this costs
        #   nothing today and refuses a degraded type file tomorrow.
        if r["other_rate"] <= max_other_semantic and r["n_distinct"] >= 2:
            return t
        msg = (f"semantic types too sparse for {dataset}: "
               f"OTHER={r['other_rate']:.1%} > {max_other_semantic:.0%} "
               f"(coverage below {1 - max_other_semantic:.0%}), "
               f"{r['n_distinct']} distinct")
        if require_semantic:
            raise SystemExit(f"\n✋ {msg}\n   --require-semantic is set, so "
                             f"falling back to induced types is refused.")
        print(f"[types] ✋ {msg}; falling back")
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        if require_semantic:
            raise SystemExit(
                f"\n✋ no exogenous semantic types for {dataset}.\n"
                f"   {type(exc).__name__}: {exc}\n\n"
                f"   --require-semantic is set, so the induced fallback is\n"
                f"   REFUSED. Either produce the type source, or drop the flag\n"
                f"   and report conditions C/G as bounding INDUCED types only.") from exc
        print(f"[types] no exogenous source for {dataset}: {exc}"[:200])

    # ── endogenous fallback ─────────────────────────────────────────────────
    # ⚠️ Only reachable when require_semantic is False. Everything below
    #    measures a DIFFERENT quantity from the branch above; the two must
    #    never be mixed inside one reported table.
    for method in ("auto", "induced"):
        t = entity_types(kg, method=method)
        n = len(t) or 1
        other = sum(1 for v in t.values() if v in (None, "OTHER")) / n
        distinct = len({v for v in t.values() if v not in (None, "OTHER")})
        print(f"[types] {dataset} {cond.id}: method={method} (ENDOGENOUS) -> "
              f"{distinct} distinct, OTHER={other:.1%}")
        if other <= max_other and distinct >= 2:
            print(f"[types] ⚠️ these are INDUCED types, derived from the edges "
                  f"under test. Report condition {cond.id} as bounding what "
                  f"induced types can do — NOT what semantic types could.")
            return t

    raise SystemExit(
        f"\n★ TYPE EXTRACTION FAILED for {dataset} (condition {cond.id}).\n"
        f"  No method produced usable types: >{max_other:.0%} of entities are OTHER.\n"
        f"  Condition {cond.id} would render prompts IDENTICAL to the no-types\n"
        f"  condition, and the experiment would silently measure nothing.\n\n"
        f"  Options:\n"
        f"    1. Use FB13 — its relations (profession, nationality, place_of_birth)\n"
        f"       carry real domain/range, which is where `Person -bornIn-> Location`\n"
        f"       actually lives. WN11's ids (`__east_indian_1`) have no type marker.\n"
        f"    2. Supply types from an external source (WordNet lexnames, YAGO's\n"
        f"       type hierarchy) and pass them in.\n"
        f"  Refusing to build a condition that cannot differ from its baseline.")


def demo_pool(kg: KG, k: int = 50) -> dict[str, list[Triple]]:
    """Per relation, a few real training triples to show as examples (P4)."""
    pool: dict[str, list[Triple]] = {}
    for t in kg.train:
        pool.setdefault(t.relation, [])
        if len(pool[t.relation]) < k:
            pool[t.relation].append(t)
    return pool


# =============================================================================
#  BUILD
# =============================================================================
def build_condition(cond: Condition, dataset: str, root: str, n_triples: int,
                    seed: int, prompt_id: str = "P0", out_root: str | None = None,
                    require_semantic: bool = False,
                    min_context: float = 0.5) -> dict:
    from src.data.sampling import sample_triples
    from src.utils.config import set_all_seeds
    set_all_seeds(seed)

    kg = load_kg(dataset, root)
    if cond.anonymise:
        kg = anonymise(kg)          # ⚠️ anonymises train AND test consistently
    elif getattr(cond, "shuffle", False):
        # ★ real names kept, permuted across entities. Same vocabulary, same
        # lengths, same readability -- only the name↔entity binding is gone.
        kg = shuffle_surface_forms(kg, seed=seed)

    # ★ CONDITION-level and PROMPT-level `types` are two different switches.
    #
    #   cond.types  -- this condition TRAINS with type tags   (C, D, E, G)
    #   pv.types    -- this prompt VARIANT shows type tags    (P1, P3)
    #
    # `render` only consults `pv.types`, so condition C at prompt P0 computed the
    # types and then silently dropped them -- C came out byte-identical to B.
    # Fold the condition's switch into the effective variant so one flag governs
    # rendering and the two can never disagree again.
    pv = PROMPTS[prompt_id]
    if cond.types and not pv.types:
        pv = replace(pv, types=True)

    # ✋ FAIL FAST. This check used to sit next to the test-set construction,
    #    i.e. AFTER sample_triples, build_relation_type_index and the whole
    #    training-instance loop -- minutes of work on YAGO3-10 before refusing.
    #    Nothing above this point depends on the labels, so refuse here.
    n_lab = sum(1 for t in kg.test if t.label is not None)
    if n_lab == 0:
        raise SystemExit(
            f"\n✋ {dataset}'s test set carries NO ±1 labels, so every test\n"
            f"   instance would be built as a NEGATIVE and the triple-\n"
            f"   classification arm would silently measure nothing.\n\n"
            f"   Generate them first:\n"
            f"     python -m scripts.make_test_negatives --dataset {dataset} "
            f"--strategy type_consistent\n"
            f"     python -m chapter1.validate --dataset {dataset}\n\n"
            f"   (Ranking does not need labels -- chapter1.rank treats an\n"
            f"    unlabelled test set as all-positives, which is correct there.)")

    types = (build_types(kg, cond, dataset, root,
                     require_semantic=require_semantic)
         if pv.types else None)
    demos = demo_pool(kg) if pv.demonstrations else None

    # ── P5/P6/P7 context, for TRAINING as well as inference ─────────────────
    # ★ WITHOUT THIS, `--prompt P6` built P0 PROMPTS UNDER A P6 LABEL. render()
    #   takes `context` and the builder never passed it, so the neighbour block
    #   silently vanished. The differentiation guard below only inspected type
    #   tags, so nothing complained: a 40-minute run would have produced an
    #   adapter that never saw a single neighbour.
    ctx_index = ctx_rng = None
    needs_ctx = pv.relation_desc or pv.neighbours or pv.paths
    if needs_ctx:
        from .context import GraphIndex, assert_no_leak
        ctx_index = GraphIndex(kg)
        ctx_rng = random.Random(seed)
        if pv.neighbours:
            print(f"[context] leak check: {assert_no_leak(kg, ctx_index)}")

    def _ctx(t) -> str | None:
        """The context block for one triple, or None. Same guard as ranking."""
        if not needs_ctx:
            return None
        from .context import describe_relation, neighbour_block, path_block
        bits = []
        if pv.relation_desc:
            bits.append(describe_relation(t.relation, kg))
        if pv.neighbours:
            bits.append(neighbour_block(ctx_index, kg, t.head, t.relation,
                                        pv.neighbours, ctx_rng, gold=t.tail))
        if pv.paths:
            bits.append(path_block(ctx_index, kg, t.head, t.tail, t.relation,
                                   pv.paths))
        return " ".join(b for b in bits if b) or None

    # ---- training instances -------------------------------------------------
    pos = sample_triples(kg.train, n_triples, seed=seed,
                         stratified=True, min_per_relation=10)

    records: list[dict] = []
    rng = random.Random(seed)
    n_out = len(pos) * (1 + cond.n_negatives)
    print(f"  building {len(pos):,} positives × (1 + {cond.n_negatives} negatives) "
          f"= {n_out:,} instances")

    # ★ Build the relation type index ONCE. It scans the entire training graph
    #   (1,079,040 triples on YAGO3-10). Rebuilding it inside the loop made
    #   condition D take ~1.5 h and condition E ~9 h; with it hoisted, both are
    #   minutes. Conditions A/B/C were never affected because `random` negatives
    #   do not touch the index.
    neg_index = (build_relation_type_index(kg)
                 if cond.negatives == "type_consistent" else None)
    if neg_index is not None:
        print(f"  [negatives] type index built once over {len(kg.train):,} "
              f"training triples -> {len(neg_index)} relations")

    for p in track(pos, f"[{cond.id}] train instances", total=len(pos), unit="triple"):
        d = demos.get(p.relation) if demos else None
        records.append({"instruction": render(p, kg, pv, types, d, _ctx(p)),
                        "input": "", "output": YES})
        # ★ n_negatives per positive -- the axis D/E move
        for _ in range(cond.n_negatives):
            neg = make_negatives([p], kg, strategy=cond.negatives,
                                 seed=rng.randrange(1 << 30),
                                 type_index=neg_index)[0]
            # ★ the negative gets its OWN context: same head, but the path
            #   block depends on the corrupted tail. Reusing the positive's
            #   block would put the true answer's evidence behind a false
            #   statement, which is a leak in the opposite direction.
            records.append({"instruction": render(neg, kg, pv, types, d, _ctx(neg)),
                            "input": "", "output": NO})
    rng.shuffle(records)

    # ---- test instances -----------------------------------------------------
    #
    # ✋ THE UNLABELLED-TEST TRAP.
    #    `YES if t.label == 1 else NO` is silent when label is None: `None == 1`
    #    is False, so EVERY test instance becomes a negative. The build then
    #    succeeds, training succeeds, and evaluate._score_set compares a ±1
    #    prediction against None -- never equal -- so accuracy comes out near 0
    #    and nothing anywhere says why.
    #
    #    Datasets that ship ±1 labels: WN11, FB13.
    #    Datasets that do NOT: YAGO3-10, WN18RR, and every CATS inductive split
    #    (NELL-995-ind, WN18RR-ind) -- those are link-prediction splits where
    #    every test row is a true fact.
    #
    #    Refuse rather than build. `label` is also normalised to a plain int so
    #    nothing downstream has to guess.
    if n_lab < len(kg.test):
        print(f"  ⚠️ {len(kg.test) - n_lab:,}/{len(kg.test):,} test triples have no "
              f"label and are DROPPED; a partially labelled set is not balanced")

    test = [{**{"instruction": render(t, kg, pv, types,
                                      demos.get(t.relation) if demos else None,
                                      _ctx(t)),
                "input": "", "output": YES if t.label == 1 else NO},
             "label": int(t.label), "head": t.head,
             "relation": t.relation, "tail": t.tail}
            for t in track([t for t in kg.test if t.label is not None],
                           f"[{cond.id}] test instances",
                           total=n_lab, unit="triple")]

    # ---- ★ seen/unseen split, computed at build time ------------------------
    # We train on 10k of WN11's 112,581 triples, so roughly half the test
    # entities were never seen. That accidental inductive split is an
    # INDEPENDENT test of memorisation -- see analysis.seen_unseen().
    seen = {e for t in pos for e in (t.head, t.tail)}
    for r in test:
        r["seen_head"] = r["head"] in seen
        r["seen_tail"] = r["tail"] in seen
        r["seen_both"] = r["seen_head"] and r["seen_tail"]

    # ★★ CONTEXT GUARD — the prompt-level twin of the type guard below.
    #    A P5/P6/P7 run whose block comes out empty IS a P0 run. It trains
    #    fine, evaluates fine, and reports "context makes no difference" when
    #    the context was never there. Refuse below 50%, warn below 90%, and
    #    record the coverage in the manifest either way so the number can be
    #    reported next to the result.
    ctx_cov = 0.0
    if needs_ctx:
        probe = [_ctx(t) for t in pos[:min(300, len(pos))]]
        ctx_cov = sum(1 for c in probe if c) / max(1, len(probe))
        bare = sum(1 for r, t in zip(records[:200], pos[:200], strict=False)
                   if r["instruction"] == render(t, kg, PROMPTS["P0"], None, None))
        if ctx_cov < min_context:
            raise SystemExit(
                f"\n✋ {prompt_id} produced a context block for only "
                f"{ctx_cov:.1%} of training triples.\n"
                f"   Below half, this run is mostly IDENTICAL to P0 and would\n"
                f"   report 'context does not help' when context was absent.\n"
                f"   P7 in particular needs a 2-hop path to exist; on a sparse\n"
                f"   graph it often does not.\n\n"
                f"   Either use P5/P6, or lower the bar DELIBERATELY and report\n"
                f"   the coverage beside the metric:\n"
                f"     python -m chapter1.data --prompt {prompt_id} "
                f"--min-context 0.05")
        if ctx_cov < 0.9:
            print(f"  ⚠️ {prompt_id}: only {ctx_cov:.1%} of instances carry a "
                  f"context block — report this coverage beside the metric")
        print(f"[guard] {prompt_id}: context on {ctx_cov:.1%} of instances, "
              f"{bare}/200 still byte-identical to P0 ✓")

    # ★★ DIFFERENTIATION GUARD.
    # A condition that renders the same prompts as its baseline measures nothing.
    # This is the failure that produced identical B and C prompts, and it is
    # invisible downstream: training succeeds, accuracy looks plausible, and the
    # ladder reports "types do not help" when types were never present.
    if cond.types:
        # ★ `bare` and `same` used to be computed here and never read -- dead
        #   code that made the guard look like it compared against the untyped
        #   baseline when it only counted brackets. Do the comparison for real:
        #   a typed prompt must DIFFER from the same triple rendered untyped.
        n_check = min(200, len(records))
        identical = sum(
            1 for p in pos[:n_check]
            if render(p, kg, pv, types, demos.get(p.relation) if demos else None)
            == render(p, kg, PROMPTS["P0"], None, None))
        typed_marker = sum(1 for r in records[:n_check] if "[" in r["instruction"])
        if identical:
            raise SystemExit(
                f"\n★ CONDITION {cond.id}: {identical}/{n_check} typed prompts are "
                f"BYTE-IDENTICAL to their untyped rendering. The condition would "
                f"measure nothing. See build_types().")
        if typed_marker == 0:
            raise SystemExit(
                f"\n★ CONDITION {cond.id} HAS types=True BUT NO TYPE TAG APPEARS "
                f"IN ANY PROMPT.\n  It would be identical to the no-types condition. "
                f"See build_types() for why.\n"
                f"  Example: {records[0]['instruction'][:100]}")
        print(f"[guard] {cond.id}: {typed_marker}/{n_check} prompts carry a type "
              f"tag, 0/{n_check} identical to the untyped rendering ✓")

    tag = f"{dataset}-{cond.id}" + (f"-{prompt_id}" if prompt_id != "P0" else "")
    out = Path(out_root or root, tag, "built")
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_instructions.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    (out / "test_instructions.json").write_text(json.dumps(test, indent=1), encoding="utf-8")

    manifest = {
        "condition": cond.id, "prompt": prompt_id, "dataset": dataset, "seed": seed,
        "anonymised": cond.anonymise, "types": cond.types,
        "require_semantic": require_semantic,
        "negatives": cond.negatives, "n_negatives": cond.n_negatives,
        "n_triples_sampled": len(pos),
        "n_train_instances": len(records),      # ⚠️ report alongside triples
        "n_test_instances": len(test),
        # ★ The paper says "test sets are balanced, so accuracy equals balanced
        #   accuracy and chance is 0.5". For a set we GENERATED (YAGO3-10,
        #   NELL-995) that is an assumption, not a fact -- record it so the
        #   claim can be checked instead of trusted.
        "test_positive_rate": sum(r["label"] == 1 for r in test) / max(1, len(test)),
        # ★ a P5/P6/P7 result is uninterpretable without this
        "context_coverage": ctx_cov if needs_ctx else None,
        "n_entities_seen_in_training": len(seen),
        "n_entities_total": len(kg.ent2txt),
        "seen_coverage": len(seen) / max(1, len(kg.ent2txt)),
        "test_seen_both_rate": sum(r["seen_both"] for r in test) / max(1, len(test)),
        "isolates": cond.isolates, "reference": cond.reference,
        "example_positive": records[0] if records else None,
        "example_test": test[0] if test else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    pr = manifest["test_positive_rate"]
    if abs(pr - 0.5) > 0.02:
        print(f"  ⚠️ test set is {pr:.1%} positive, not 50%. Accuracy is then NOT "
              f"balanced accuracy and chance is NOT 0.5 — report balanced "
              f"accuracy, or regenerate with --n-negatives 1")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    # nargs="+" so `--condition D E` works. Previously one value only, which
    # silently looked like a typo: argparse reported "unrecognized arguments: E".
    ap.add_argument("--condition", default=["A"], nargs="+", choices=list(CONDITIONS),
                    metavar="ID", help="one or more of: " + " ".join(CONDITIONS))
    ap.add_argument("--prompt", default="P0", choices=list(PROMPTS))
    ap.add_argument("--dataset", default="WN11")
    ap.add_argument("--root", default="data")
    ap.add_argument("--n_triples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true", help="build every condition at P0")
    ap.add_argument("--min-context", type=float, default=0.5,
                    help="refuse a P5/P6/P7 build whose context block is empty "
                         "for more than this share of instances. Lower it only "
                         "if you will report the coverage.")
    ap.add_argument("--require-semantic", action="store_true",
                    help="★ refuse to fall back to INDUCED types. Use this for "
                         "any run whose numbers go in the paper: it turns a "
                         "silent downgrade into a hard stop.")
    ns = ap.parse_args()

    todo = list(CONDITIONS) if ns.all else list(dict.fromkeys(ns.condition))
    if len(todo) > 1:
        print(f"[build] {len(todo)} conditions: {' '.join(todo)}")
    for cid in todo:
        m = build_condition(CONDITIONS[cid], ns.dataset, ns.root,
                            ns.n_triples, ns.seed, ns.prompt,
                            require_semantic=ns.require_semantic,
                            min_context=ns.min_context)
        print(f"\n[{cid}] {m['n_triples_sampled']:,} triples -> "
              f"{m['n_train_instances']:,} instances | "
              f"seen coverage {m['seen_coverage']:.1%} | "
              f"test seen-both {m['test_seen_both_rate']:.1%}")
        print(f"     {m['example_positive']['instruction'][:110]}")


if __name__ == "__main__":
    main()
