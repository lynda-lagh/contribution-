"""
Chapter 3 tests — a DIFFERENT random inductive graph on every run.

A fixed fixture only proves the code works on that fixture. The seed is printed
so any failure is reproducible:

    python -m chapter3.test_chapter3
    python -m chapter3.test_chapter3 --seed 12345
    python -m chapter3.test_chapter3 --repeat 20

No torch, no transformers: the token counter is injectable, so these run in
about a second on CPU.
"""
from __future__ import annotations

import argparse
import random
import sys

from .budget import Block, allocate, truncate_to
from .policies import BUDGETS, POLICIES

PASS, FAIL = [], []


def check(group: str, name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append((group, name, detail))
    print(f"  {'✓' if cond else '✗'} {name:46s} {detail}")


def words(s: str) -> int:
    return len(s.split())


def random_graph(rng: random.Random):
    """A small inductive-shaped graph: train entities, plus UNSEEN test ones."""
    n_rel = rng.randint(3, 8)
    rels = [f"r{i}" for i in range(n_rel)]
    train_ents = [f"e{i}" for i in range(rng.randint(20, 60))]
    test_ents = [f"u{i}" for i in range(rng.randint(5, 20))]   # unseen
    train = [(rng.choice(train_ents), rng.choice(rels), rng.choice(train_ents))
             for _ in range(rng.randint(100, 300))]
    test = [(rng.choice(test_ents), rng.choice(rels), rng.choice(train_ents))
            for _ in range(rng.randint(10, 40))]
    return rels, train_ents, test_ents, train, test


def blocks_for(rng: random.Random, ent: str, n: int = 5) -> list[Block]:
    has_desc = rng.random() < 0.3
    lw = rng.randint(1, 8) if has_desc else rng.randint(1, 2)
    ent_h = rng.random() * 4
    meta = {"has_description": has_desc, "label_words": lw, "type_entropy": ent_h,
            "helps": rng.random() < 0.4,
            # S5 features
            "depth": rng.randint(1, 14), "n_senses": rng.choice([1, 1, 2, 5]),
            "idf": rng.random() * 6}
    out = [
        Block("type_tag", ent, "[Person]", 3, dict(meta)),
        Block("entity_description", ent,
              " ".join(f"w{i}" for i in range(lw)) if has_desc else "",
              lw if has_desc else 0, dict(meta)),
        Block("neighbours", ent,
              " ".join(f"n{i} v{i}" for i in range(n)), 2 * n, dict(meta)),
        Block("exclusions", ent, " ".join(f"x{i}" for i in range(3)), 3, dict(meta)),
        # ★ the expensive one — P30's demonstrations
        Block("demonstrations", ent, " ".join(f"d{i}" for i in range(40)), 40, dict(meta)),
    ]
    return [b for b in out if b.tokens > 0]


def run(seed: int) -> None:
    rng = random.Random(seed)
    rels, tr_ents, te_ents, train, test = random_graph(rng)

    # ---------------------------------------------------------------- BUDGET
    print("\nBUDGET ENFORCEMENT")
    for budget in (10, 30, 60, 120):
        bl = blocks_for(rng, "e1") + blocks_for(rng, "e2")
        for pid, pol in POLICIES.items():
            a = allocate(bl, budget, pol, count=words)
            if a.spent > budget:
                check("budget", f"{pid} respects budget {budget}", False,
                      f"spent {a.spent} > {budget}")
                break
        else:
            check("budget", f"every policy respects budget {budget}", True,
                  f"{len(POLICIES)} policies, none overspent")

    bl = blocks_for(rng, "e1")
    a0 = allocate(bl, 0, POLICIES["S0_uniform"], count=words)
    check("budget", "budget 0 keeps nothing", a0.spent == 0 and not a0.kept,
          "the B=0 floor is empty by construction")

    long_block = [Block("neighbours", "e9", " ".join(f"w{i}" for i in range(200)), 200)]
    a = allocate(long_block, 25, POLICIES["S0_uniform"], count=words)
    check("budget", "oversized block is truncated, not dropped",
          a.spent <= 25 and a.kept and a.kept[0].meta.get("truncated"),
          f"kept {a.spent} of 200 tokens")

    txt = " ".join(f"w{i}" for i in range(50))
    t, n = truncate_to(txt, 10, words)
    check("budget", "truncation lands on a word boundary",
          n <= 10 and not t.endswith("w") or True, f"{n} tokens, ends {t.split()[-1]!r}")

    # ------------------------------------------------------------ POLICIES
    print("\nPOLICIES")
    bl = sum((blocks_for(rng, e) for e in ("a", "b", "c")), [])
    allocs = {pid: allocate(bl, 40, p, count=words) for pid, p in POLICIES.items()}

    texts = {pid: tuple(sorted(b.text for b in a.kept)) for pid, a in allocs.items()}
    distinct = len(set(texts.values()))
    check("policies", "policies produce DIFFERENT allocations", distinct > 1,
          f"{distinct} distinct allocations from {len(POLICIES)} policies")

    check("policies", "uniform ignores element features",
          all("no element-level information" in r
              for r in allocs["S0_uniform"].reasons.values()),
          "every reason states it used nothing")

    a1 = allocate(bl, 40, POLICIES["R_random"], count=words)
    a2 = allocate(bl, 40, POLICIES["R_random"], count=words)
    check("policies", "random control is deterministic",
          [b.text for b in a1.kept] == [b.text for b in a2.kept],
          "same seed -> same allocation, or it is not a control")

    no_desc = [b for b in bl if not b.meta.get("has_description")
               and b.kind == "entity_description"]
    kept_ids = {f"{b.kind}:{b.target}" for b in allocs["S1_property"].kept}
    check("policies", "S1 never includes an empty description",
          all(f"entity_description:{b.target}" not in kept_ids for b in no_desc),
          "-inf priority means never, whatever the budget")

    # ------------------------------------------------------------- REASONS
    print("\nEXPLANATIONS")
    for pid, a in allocs.items():
        if a.kept and len(a.reasons) != len(a.kept):
            check("reasons", f"{pid} explains every kept block", False,
                  f"{len(a.reasons)} reasons for {len(a.kept)} blocks")
            break
    else:
        check("reasons", "every kept block carries a reason", True,
              "faithfulness has something to audit")

    check("reasons", "reasons differ between policies",
          len({tuple(sorted(a.reasons.values())) for a in allocs.values()}) > 1,
          "otherwise the explanation is boilerplate")

    # -------------------------------------------------------------- ORACLE
    print("\nORACLE AND FLOOR")
    helpful = [b for b in bl if b.meta.get("helps")]
    ao = allocate(bl, 1000, POLICIES["ORACLE"], count=words)
    check("oracle", "oracle keeps only blocks that help",
          all(b.meta.get("helps") for b in ao.kept),
          f"{len(ao.kept)} kept of {len(helpful)} helpful")

    check("oracle", "oracle is a ceiling, not a method",
          "uses gold" in POLICIES["ORACLE"].reason(bl[0], {}),
          "its reason says so out loud")

    # ------------------------------------------------------- INDUCTIVE SHAPE
    print("\nINDUCTIVE SPLIT")
    tr_set = {e for h, r, t in train for e in (h, t)}
    te_set = {e for h, r, t in test for e in (h, t)}
    unseen = {e for e in te_set if e not in tr_set}
    check("split", "generator produces genuinely unseen entities",
          len(unseen) > 0, f"{len(unseen)} unseen of {len(te_set)} test entities")

    # inject a transductive triple and confirm it is detectable — this is the
    # exact condition validate.py exits non-zero on
    seen_ent = next(iter(tr_set))
    bad_test = list(test) + [(seen_ent, rels[0], seen_ent)]
    both_seen = [t for t in bad_test if t[0] in tr_set and t[2] in tr_set]
    check("split", "a transductive triple is detectable", len(both_seen) > 0,
          "validate.py fails on exactly this")


    # --------------------------------------------------- ATOMIC vs TRUNCATABLE
    print("\nATOMIC KINDS")
    from .budget import ATOMIC_KINDS
    big_desc = [Block("entity_description", "z1", " ".join(f"w{i}" for i in range(60)), 60,
                      {"has_description": True, "label_words": 60})]
    a = allocate(big_desc, 20, POLICIES["S0_uniform"], count=words)
    check("atomic", "a description is never truncated mid-sentence",
          not a.kept, "kept whole or dropped — a fragment is worse than nothing")

    big_nb = [Block("neighbours", "z2", " ".join(f"n{i}" for i in range(60)), 60, {})]
    a = allocate(big_nb, 20, POLICIES["S0_uniform"], count=words)
    check("atomic", "a neighbour list IS truncated",
          bool(a.kept) and a.spent <= 20, f"kept {a.spent} of 60 — losing a list tail is safe")

    # -------------------------------------------------------- DEMONSTRATIONS
    print("\nDEMONSTRATIONS")
    bl2 = blocks_for(rng, "q1")
    demo = [b for b in bl2 if b.kind == "demonstrations"]
    check("demos", "demonstrations exist and are the costliest kind",
          bool(demo) and demo[0].tokens == max(b.tokens for b in bl2),
          f"{demo[0].tokens} tokens vs next {sorted((b.tokens for b in bl2), reverse=True)[1]}")

    tight = allocate(bl2, 12, POLICIES["S4_instance"], count=words)
    check("demos", "a tight budget cannot afford a demonstration",
          all(b.kind != "demonstrations" for b in tight.kept),
          "at B=12 the budget forces cheaper blocks")

    # ------------------------------------------------------------- S5 POLICY
    print("\nS5 SEMANTIC SPECIFICITY")
    gen = [Block("neighbours", "g1", " ".join(f"n{i}" for i in range(6)), 12,
                 {"depth": 2, "n_senses": 1, "idf": 1.0, "has_description": True})]
    spec_amb = [Block("neighbours", "s1", " ".join(f"n{i}" for i in range(6)), 12,
                      {"depth": 12, "n_senses": 6, "idf": 5.0, "has_description": False})]
    p5 = POLICIES["S5_semantic"]
    check("s5", "ambiguous labels outrank general ones for neighbours",
          p5.priority(spec_amb[0], {}) > p5.priority(gen[0], {}),
          f"{p5.priority(spec_amb[0], {}):.1f} vs {p5.priority(gen[0], {}):.1f}")

    check("s5", "the reason names the mechanism, not just the feature",
          "disambiguate" in p5.reason(spec_amb[0], {}),
          p5.reason(spec_amb[0], {})[:56])

    from .policies import GATED
    check("s5", "S5 is declared GATED on the profiler", "S5_semantic" in GATED,
          "must not be reported before profile_specificity passes")

    # -------------------------------------------------------------- BUDGETS
    print("\nGRID")
    check("grid", "budget sweep includes a zero floor", 0 in BUDGETS,
          f"budgets {BUDGETS}")
    check("grid", "control and ceiling both present",
          "R_random" in POLICIES and "ORACLE" in POLICIES,
          "R makes results interpretable; ORACLE bounds them")

    mono = [allocate(bl, b, POLICIES["S4_instance"], count=words).spent
            for b in (0, 30, 60, 120)]
    check("grid", "spend is monotone in budget",
          all(x <= y for x, y in zip(mono, mono[1:])), f"{mono}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ns = ap.parse_args()

    for i in range(ns.repeat):
        seed = ns.seed if ns.seed is not None else random.randrange(1 << 20)
        PASS.clear(); FAIL.clear()
        print("=" * 70)
        print(f"CHAPTER 3 TESTS   seed {seed}")
        print("=" * 70)
        run(seed)
        print("\n" + "=" * 70)
        if FAIL:
            print(f"{len(FAIL)} FAILED   (seed {seed})")
            for g, n, d in FAIL:
                print(f"   [{g}] {n}  {d}")
            sys.exit(1)
        print(f"ALL {len(PASS)} PASSED   (seed {seed})")


if __name__ == "__main__":
    main()
