"""
★★ DATA AUDIT — check the data itself, at every stage, before believing a number.

    python -m chapter1.audit_data --dataset YAGO3-10
    python -m chapter1.audit_data --dataset YAGO3-10 --conditions A S B C G

`preflight.py` asks "can this run start?". This asks "is the data telling the
truth?". They are different questions and both have already failed silently
here: the type-tag leak (62.4% from a one-line rule) and the S permutation bug
were both DATA problems that every code-level check passed straight over.

STAGES
  1  RAW GRAPH        duplicates · self-loops · train/test leakage · coverage
  2  SPLIT            transductive vs inductive · unseen-entity share
  3  LABELS           balance · per-relation skew · negatives that are true
  4  BUILT INSTANCES  count · positive rate · prompt length · empty text
  5  CONDITION PAIRS  are the arms actually different, and matched?
  6  TYPES            inventory · OTHER rate · concentration · anon-invariance
  7  CANDIDATE POOL   gold present · filtering · reproducibility

Severity: FAIL stops you, WARN is a number you must report, INFO is context.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

FAIL, WARN, INFO, OK = "✗ FAIL", "⚠ WARN", "· info", "✓"


class Audit:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.warns: list[str] = []

    def head(self, title: str) -> None:
        print(f"\n{'─' * 76}\n{title}\n{'─' * 76}")

    def ok(self, msg: str) -> None:
        print(f"  {OK} {msg}")

    def info(self, msg: str) -> None:
        print(f"  {INFO} {msg}")

    def warn(self, msg: str) -> None:
        print(f"  {WARN} {msg}")
        self.warns.append(msg)

    def fail(self, msg: str) -> None:
        print(f"  {FAIL} {msg}")
        self.fails.append(msg)

    def check(self, cond: bool, ok_msg: str, bad_msg: str, hard: bool = True) -> bool:
        if cond:
            self.ok(ok_msg)
        else:
            (self.fail if hard else self.warn)(bad_msg)
        return cond


# =============================================================================
def stage_raw(a: Audit, kg, ds: str) -> None:
    a.head("1 · RAW GRAPH")
    a.info(f"{len(kg.ent2txt):,} entities · {len(kg.rel2txt)} relations · "
           f"{len(kg.train):,} train · {len(kg.test):,} test")

    tr = [(t.head, t.relation, t.tail) for t in kg.train]
    te = [(t.head, t.relation, t.tail) for t in kg.test]
    dup_tr, dup_te = len(tr) - len(set(tr)), len(te) - len(set(te))
    a.check(dup_tr == 0, "no duplicate training triples",
            f"{dup_tr:,} duplicate training triples — they up-weight whatever "
            f"they contain and inflate any per-relation figure", hard=False)
    a.check(dup_te == 0, "no duplicate test triples",
            f"{dup_te:,} duplicate test triples — accuracy double-counts them",
            hard=False)

    loops = sum(1 for h, _, t in tr + te if h == t)
    a.check(loops == 0, "no self-loops",
            f"{loops:,} self-loops (h == t) — trivially satisfiable", hard=False)

    # ★ THE ONE THAT MATTERS. A POSITIVE test triple also present in train is
    #   memorisable by definition and inflates the real-name arm specifically,
    #   which is the numerator of the memorisation share.
    pos_te = {(t.head, t.relation, t.tail) for t in kg.test if t.label != -1}
    leak = pos_te & set(tr)
    a.check(not leak, "no positive test triple appears in train",
            f"{len(leak):,}/{len(pos_te):,} positive test triples ALSO appear in "
            f"train ({len(leak) / max(1, len(pos_te)):.2%}) — these are memorisable "
            f"by construction and inflate condition A", hard=False)

    ents = set(kg.ent2txt)
    missing = {e for h, _, t in tr + te for e in (h, t)} - ents
    a.check(not missing, "every triple's entities are in entity2text",
            f"{len(missing):,} entity ids appear in triples but NOT in "
            f"entity2text — render() falls back to the raw id, so those prompts "
            f"silently carry an unreadable token")
    blank = sum(1 for v in kg.ent2txt.values() if not str(v).strip())
    a.check(blank == 0, "no blank surface forms",
            f"{blank:,} entities have an EMPTY name — the prompt reads "
            f"'Is this true:  died in X?'")


def stage_split(a: Audit, kg) -> None:
    a.head("2 · SPLIT — transductive or inductive?")
    tr_e = {e for t in kg.train for e in (t.head, t.tail)}
    te_e = {e for t in kg.test for e in (t.head, t.tail)}
    unseen = te_e - tr_e
    share = len(unseen) / max(1, len(te_e))
    a.info(f"{len(te_e):,} test entities · {len(unseen):,} unseen in train "
           f"({share:.1%})")
    if share < 0.02:
        a.ok("fully transductive — memorisation is directly useful here, which "
             "is exactly why the diagnostic is needed")
    elif share > 0.5:
        a.warn(f"{share:.0%} of test entities are unseen — this is closer to an "
               f"INDUCTIVE split. The memorisation gap should be smaller, and "
               f"that is a finding, not a bug. Say which regime you are in.")
    else:
        a.info("mixed regime — report the unseen share alongside the gap")


def stage_labels(a: Audit, kg) -> None:
    a.head("3 · LABELS")
    lab = [t.label for t in kg.test]
    n_lab = sum(1 for x in lab if x is not None)
    if not a.check(n_lab > 0, f"{n_lab:,} labelled test triples",
                   "test set carries NO ±1 labels — chapter1.data would build "
                   "every instance as a NEGATIVE"):
        return
    pos = sum(1 for x in lab if x == 1) / n_lab
    a.check(abs(pos - 0.5) <= 0.02, f"balanced: {pos:.1%} positive",
            f"{pos:.1%} positive, not 50% — accuracy is NOT balanced accuracy "
            f"and chance is NOT 0.5")

    per = defaultdict(lambda: [0, 0])
    for t in kg.test:
        if t.label is not None:
            per[t.relation][0 if t.label == 1 else 1] += 1
    skew = {r: p / max(1, p + n) for r, (p, n) in per.items() if p + n >= 30}
    bad = {r: v for r, v in skew.items() if abs(v - 0.5) > 0.15}
    a.check(not bad, f"per-relation balance holds across {len(skew)} relations",
            f"{len(bad)} relation(s) are skewed: " +
            ", ".join(f"{r} {v:.0%} pos" for r, v in
                      sorted(bad.items(), key=lambda x: -abs(x[1] - .5))[:5]) +
            " — a per-relation table will mislead", hard=False)

    # closed-world: a generated negative that is actually a recorded fact.
    # ★ Severity is proportional. WN11 ships 7 such rows out of 10,542 (0.07%),
    #   which is a defect to REPORT, not a reason to stop. Above 1% the label
    #   noise is large enough to move the accuracy itself.
    tr = {(t.head, t.relation, t.tail) for t in kg.train}
    negs = {(t.head, t.relation, t.tail) for t in kg.test if t.label == -1}
    bad = negs & tr
    frac = len(bad) / max(1, len(negs))
    a.check(not bad, "no test negative is a recorded training fact",
            f"{len(bad):,}/{len(negs):,} test NEGATIVES ({frac:.2%}) appear as "
            f"TRUE triples in train — the label is wrong and the model is "
            f"punished for being right",
            hard=frac > 0.01)


def stage_built(a: Audit, root: str, ds: str, conds: list[str]) -> dict[str, list]:
    a.head("4 · BUILT INSTANCES")
    built = {}
    for c in conds:
        p = Path(root, f"{ds}-{c}", "built")
        tr_f, te_f = p / "train_instructions.json", p / "test_instructions.json"
        if not tr_f.exists():
            a.warn(f"{c}: not built yet ({p})")
            continue
        tr = json.loads(tr_f.read_text(encoding="utf-8"))
        te = json.loads(te_f.read_text(encoding="utf-8"))
        built[c] = (tr, te)

        pos = sum(1 for r in tr if r["output"].strip().lower().startswith("yes"))
        rate = pos / max(1, len(tr))
        lens = sorted(len(r["instruction"]) for r in tr)
        empty = sum(1 for r in tr if not r["instruction"].strip())
        a.info(f"{c}: {len(tr):,} train · {len(te):,} test · {rate:.1%} positive "
               f"· prompt chars p50={lens[len(lens) // 2]} p99={lens[int(.99 * len(lens))]}")
        if empty:
            a.fail(f"{c}: {empty:,} EMPTY prompts")
        if abs(rate - 1 / (1 + 1)) > 0.02 and abs(rate - 1 / (1 + 6)) > 0.02:
            a.warn(f"{c}: training positive rate {rate:.1%} matches neither 1:1 "
                   f"nor 1:6 — check n_negatives")
        # the model was trained at 1:1 but tested at 1:1 too; a mismatch is the
        # condition-E failure mode
        te_pos = sum(1 for r in te if r.get("label") == 1) / max(1, len(te))
        if abs(te_pos - 0.5) > 0.02:
            a.warn(f"{c}: TEST positive rate {te_pos:.1%} — not balanced")
        if abs(rate - te_pos) > 0.15:
            a.warn(f"{c}: trained at {rate:.0%} positive but tested at "
                   f"{te_pos:.0%} — the model learns the training prior and "
                   f"argmax at 0.5 is the wrong operating point (this is what "
                   f"collapsed condition E). Report AUC as well as accuracy.")
    return built


def stage_pairs(a: Audit, built: dict) -> None:
    a.head("5 · CONDITION PAIRS — different, and matched?")
    if len(built) < 2:
        a.warn("fewer than two conditions built — nothing to compare")
        return

    def firsts(c, k=400):
        return [r["instruction"] for r in built[c][0][:k]]

    seen: dict[str, str] = {}
    for c in built:
        sig = str(hash(tuple(firsts(c))))
        if sig in seen:
            a.fail(f"{c} and {seen[sig]} produce IDENTICAL prompts — one of them "
                   f"measures nothing")
        else:
            seen[sig] = c
            a.ok(f"{c}: prompts differ from every earlier condition")

    # the test sets of a matched pair must align row-for-row
    keys = {c: [(r.get("head"), r.get("relation"), r.get("tail"))
                for r in built[c][1]] for c in built}
    ref = next(iter(keys))
    for c, v in keys.items():
        if v != keys[ref]:
            a.fail(f"{c}'s test rows are not in the same order as {ref}'s — the "
                   f"gap would compare different triples")
    if all(v == keys[ref] for v in keys.values()):
        a.ok(f"all {len(keys)} test sets align row-for-row")


def stage_types(a: Audit, kg, ds: str, root: str) -> None:
    a.head("6 · TYPES")
    from src.data.loaders import anonymise, shuffle_surface_forms
    try:
        from src.routing.semantic_types import coverage, semantic_types
        t = semantic_types(kg, ds, root=root)
        kind = "EXOGENOUS (semantic)"
    except Exception as exc:                                    # noqa: BLE001
        a.warn(f"no exogenous types ({type(exc).__name__}) — falling back to "
               f"INDUCED, which are derived from the edges under test")
        from src.routing.types import entity_types
        from src.routing.semantic_types import coverage
        t = entity_types(kg, method="induced")
        kind = "ENDOGENOUS (induced)"
    r = coverage(t)
    a.info(f"{kind}: {r['n_distinct']} types · OTHER {r['other_rate']:.1%} · "
           f"largest {r['largest_share']:.1%} · top5 {r['top5_share']:.1%}")
    a.check(r["other_rate"] <= 0.5, "OTHER rate is usable",
            f"OTHER {r['other_rate']:.1%} — the typed conditions are near-vacuous")
    a.check(r["n_distinct"] >= 2, f"{r['n_distinct']} distinct types",
            "fewer than 2 distinct types — the tag carries no information")
    if r["largest_share"] > 0.5:
        a.warn(f"one type covers {r['largest_share']:.0%} of entities — the tag "
               f"is close to constant and bounds what C and G can show")

    # ★ the invariance C and G depend on
    for name, fn in (("anonymise", anonymise),
                     ("permute", lambda k: shuffle_surface_forms(k, seed=42))):
        try:
            from src.routing.semantic_types import semantic_types as st
            t2 = st(fn(kg), ds, root=root)
        except Exception:                                       # noqa: BLE001
            from src.routing.types import entity_types
            t2 = entity_types(fn(kg), method="induced")
        n = sum(1 for k in t if t.get(k) != t2.get(k))
        a.check(n == 0, f"types are invariant under {name}",
                f"{n:,} types CHANGE under {name} — C and G are not a matched "
                f"pair and the comparison is meaningless")


def stage_candidates(a: Audit, kg, seed: int, n_way: int) -> None:
    a.head("7 · CANDIDATE POOL")
    import random

    from .rank import sample_candidates
    known = defaultdict(set)
    for t in (*kg.train, *getattr(kg, "valid", ()), *kg.test):
        known[(t.head, t.relation)].add(t.tail)
    ents = list(kg.ent2txt)
    if len(ents) < n_way:
        a.fail(f"only {len(ents)} entities but n_way={n_way}")
        return

    qs = [t for t in kg.test if t.label in (1, None)][:200]
    rng = random.Random(seed)
    short = gold_missing = contaminated = 0
    for q in qs:
        filt = known[(q.head, q.relation)] - {q.tail}
        c = sample_candidates(q.tail, ents, n_way, rng, filter_out=filt)
        short += len(c) < n_way
        gold_missing += q.tail not in c
        contaminated += len(set(c) & filt) > 0
    a.check(gold_missing == 0, f"gold answer present in all {len(qs)} pools",
            f"{gold_missing} pools LACK the gold answer — rank is undefined")
    a.check(short == 0, f"every pool has exactly {n_way} candidates",
            f"{short} pools are SHORT — chance MRR is then not 0.0900", hard=False)
    a.check(contaminated == 0, "no other true answer leaked into a pool",
            f"{contaminated} pools contain a triple that is TRUE — a correct "
            f"alternative is scored as an error")

    # reproducibility: the same seed must give the same pools
    r1, r2 = random.Random(seed), random.Random(seed)
    same = all(sample_candidates(q.tail, ents, n_way, r1,
                                 known[(q.head, q.relation)] - {q.tail}) ==
               sample_candidates(q.tail, ents, n_way, r2,
                                 known[(q.head, q.relation)] - {q.tail})
               for q in qs[:20])
    a.check(same, "candidate sampling is reproducible at a fixed seed",
            "candidate pools are NOT reproducible")
    a.warn("pools are drawn from ONE shared RNG, so they depend on --limit and "
           "on query order. Chapter 3 fixed this with per-query seeds; chapter 1 "
           "did not. Arms are comparable only if run at the same --limit.")


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--root", default="data")
    ap.add_argument("--conditions", nargs="+", default=["A", "S", "B", "C", "G"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-way", type=int, default=50)
    ap.add_argument("--skip-built", action="store_true")
    ns = ap.parse_args()

    from src.data.loaders import load_kg
    a = Audit()
    print("=" * 76)
    print(f"DATA AUDIT — {ns.dataset}")
    print("=" * 76)

    kg = load_kg(ns.dataset, ns.root)
    stage_raw(a, kg, ns.dataset)
    stage_split(a, kg)
    stage_labels(a, kg)
    built = {} if ns.skip_built else stage_built(a, ns.root, ns.dataset, ns.conditions)
    if built:
        stage_pairs(a, built)
    stage_types(a, kg, ns.dataset, ns.root)
    stage_candidates(a, kg, ns.seed, ns.n_way)

    print("\n" + "=" * 76)
    if a.fails:
        print(f"✗ {len(a.fails)} FAILURE(S) — do not train on this data:")
        for m in a.fails:
            print(f"    {m.splitlines()[0][:110]}")
    if a.warns:
        print(f"⚠ {len(a.warns)} WARNING(S) — each is a number you must REPORT, "
              f"not a bug to hide:")
        for m in a.warns:
            print(f"    {m.splitlines()[0][:110]}")
    if not a.fails and not a.warns:
        print("✓ every data check passed")
    print("=" * 76)
    sys.exit(1 if a.fails else 0)


if __name__ == "__main__":
    main()
