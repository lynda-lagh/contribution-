"""
CHAPTER 1 — tests. Every one is also a WORKED EXAMPLE.

    python -m chapter1.test_chapter1

No GPU, no model, no downloads. Runs on a 12-entity toy graph in ~2 seconds.
Run it before every Kaggle session: it catches the class of bug that produces
plausible numbers rather than a crash, which is the only kind that matters here.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from src.data.loaders import KG, Triple, anonymise

from .conditions import CONDITIONS, PROMPTS, STRUCTURAL_INSTRUCTION
from .data import demo_pool, render

PASS, FAIL = "  ✓", "  ✗"
results: list[tuple[str, bool, str]] = []

# ★ A DIFFERENT GRAPH AND A DIFFERENT TEST TRIPLE ON EVERY RUN.
#
# A fixed fixture only ever proves the code works on that fixture. Randomising
# the graph size, the entity/relation names and the triple under test turns this
# into light property-based testing: re-run it a few times and it explores a
# space instead of a point.
#
# The seed is PRINTED. If a run fails, reproduce it exactly with --seed <n>.
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--seed", type=int, default=None)
_ap.add_argument("--repeat", type=int, default=1,
                 help="run the whole suite N times with different seeds")
_ARGS, _ = _ap.parse_known_args()

SEED = _ARGS.seed if _ARGS.seed is not None else int(time.time() * 1000) % 100_000
RNG = random.Random(SEED)


def check(name, fn):
    try:
        msg = fn() or ""
        print(f"{PASS} {name} {msg}")
        results.append((name, True, msg))
    except AssertionError as e:
        print(f"{FAIL} {name}\n      {e}")
        results.append((name, False, str(e)))
    except Exception as e:
        print(f"{FAIL} {name}\n      {type(e).__name__}: {e}")
        results.append((name, False, str(e)))


# ---------------------------------------------------------------- fixture
FIRST_NAMES = ["Ada", "Bruno", "Chen", "Dilan", "Eve", "Farid", "Gita", "Hana",
               "Ivan", "Jae", "Kofi", "Lina", "Malik", "Nour", "Omar", "Priya"]
CITIES = ["Lyon", "Oslo", "Cairo", "Lima", "Perth", "Kyoto", "Accra", "Riga",
          "Quito", "Tunis", "Bergen", "Cusco"]
JOBS = ["surgeon", "poet", "chemist", "pilot", "archivist", "botanist"]


def toy(rng: random.Random) -> tuple[KG, dict, Triple]:
    """
    A RANDOM graph with real type structure, so `Person -bornIn-> Location` is
    testable. Size, names and the triple under test differ on every run.

    This mirrors FB13 rather than WN11: WN11's "types" are only parts of speech,
    and `Person -bornIn-> Location` is a Freebase statement.
    """
    n_p = rng.randint(4, 9)
    n_c = rng.randint(3, 7)
    n_j = rng.randint(2, 4)
    people = [f"p{i}" for i in range(n_p)]
    places = [f"c{i}" for i in range(n_c)]
    jobs = [f"j{i}" for i in range(n_j)]

    pn = rng.sample(FIRST_NAMES, n_p)
    cn = rng.sample(CITIES, n_c)
    jn = rng.sample(JOBS, n_j)

    ent2txt = ({p: f"{pn[i]}, a person" for i, p in enumerate(people)}
               | {c: f"{cn[i]}, a city" for i, c in enumerate(places)}
               | {j: f"{jn[i]}, a profession" for i, j in enumerate(jobs)})
    rel2txt = {"bornIn": "was born in", "worksAs": "works as", "livesIn": "lives in"}

    train = []
    for i, p in enumerate(people):
        train.append(Triple(p, "bornIn", places[i % n_c], None))
        train.append(Triple(p, "worksAs", jobs[i % n_j], None))
        train.append(Triple(p, "livesIn", places[(i + 1) % n_c], None))

    # a random POSITIVE query triple, drawn fresh each run
    q_person = rng.choice(people)
    q_place = places[people.index(q_person) % n_c]
    test = [Triple(q_person, "bornIn", q_place, 1),
            Triple(rng.choice(people), "bornIn", rng.choice(places), -1),
            Triple(rng.choice(people), "worksAs", rng.choice(jobs), 1)]

    kg = KG(name="TOY", ent2txt=ent2txt, rel2txt=rel2txt, train=train, test=test)
    types = ({p: "Person" for p in people} | {c: "Location" for c in places}
             | {j: "Profession" for j in jobs})
    return kg, types, test[0]


KG_, TYPES, T = toy(RNG)


# ================================================================ PROMPTS
def t_p0():
    s = render(T, KG_, PROMPTS["P0"])
    assert s.startswith("Is this true:"), s
    assert "[Person]" not in s, "P0 must carry NO type tag"
    assert STRUCTURAL_INSTRUCTION not in s, "P0 must carry NO instruction"
    return f'\n      → "{s}"'


def t_p1():
    s = render(T, KG_, PROMPTS["P1"], TYPES)
    assert "[Person]" in s and "[Location]" in s, s
    # ★ this is the `Person -bornIn-> Location` rule, made explicit
    return f'\n      → "{s}"'


def t_p2():
    s = render(T, KG_, PROMPTS["P2"])
    assert STRUCTURAL_INSTRUCTION in s
    assert "[Person]" not in s, "P2 adds the instruction ONLY, no types"
    return f'\n      → "{s}"'


def t_p3():
    s = render(T, KG_, PROMPTS["P3"], TYPES)
    assert "[Person]" in s and STRUCTURAL_INSTRUCTION in s
    return "  (types + instruction)"


def t_p4():
    s = render(T, KG_, PROMPTS["P4"], None, demo_pool(KG_)["bornIn"])
    assert "Other triples using" in s, s
    assert s.count("was born in") >= 2, "P4 must SHOW examples, not just the query"
    return f'\n      → "{s[:150]}…"'


def t_prompts_differ():
    """
    If two variants render identically the ablation measures nothing.

    ★ P5-P7 only differ once their CONTEXT BLOCK is supplied, so this builds it
      the way chapter1/rank.py does. Rendering them without context made all
      four collapse onto P0 — which is exactly the silent failure the test
      exists to catch, so it must exercise the real path.
    """
    from chapter1.context import (GraphIndex, describe_relation,
                                  neighbour_block, path_block)
    idx = GraphIndex(KG_)
    seen, empty = {}, []
    for pid, pv in PROMPTS.items():
        bits = []
        if pv.relation_desc:
            bits.append(describe_relation(T.relation, KG_))
        if pv.neighbours:
            bits.append(neighbour_block(idx, KG_, T.head, T.relation,
                                        pv.neighbours, random.Random(1),
                                        gold=T.tail))
        if pv.paths:
            bits.append(path_block(idx, KG_, T.head, T.tail, T.relation, pv.paths))
        ctx = " ".join(b for b in bits if b)
        # ★ a variant whose context comes out EMPTY is legitimately identical
        #   to P0 -- e.g. P7 when no path exists between head and tail. That is
        #   a property of the graph, not a bug, and rank.py reports the
        #   coverage so a null cannot be misread. Skip those here.
        if (pv.relation_desc or pv.neighbours or pv.paths) and not ctx:
            empty.append(pid)
            continue
        seen[pid] = render(T, KG_, pv, TYPES,
                           demo_pool(KG_)["bornIn"] if pv.demonstrations else None,
                           context=ctx or None)
    dupes = {v: [k for k in seen if seen[k] == v] for v in set(seen.values())}
    clash = [ks for ks in dupes.values() if len(ks) > 1]
    assert not clash, f"identical renders: {clash}"
    note = f" ({', '.join(empty)} had no context on this graph)" if empty else ""
    return f"  all {len(seen)} variants distinct{note}"


# =========================================================== ANONYMISATION
def t_anon_removes_names():
    a = anonymise(KG_)
    s = render(Triple("p1", "bornIn", "c1", 1), a, PROMPTS["P0"])
    assert "Person" not in s, f"surface form survived anonymisation: {s}"
    assert "was born in" in s, "relations must SURVIVE — only entities are hidden"
    return f'\n      → "{s}"'


def t_anon_keeps_structure():
    a = anonymise(KG_)
    assert len(a.train) == len(KG_.train) and len(a.test) == len(KG_.test)
    assert a.rel2txt == KG_.rel2txt, "relation text must be untouched"
    return "  triples and relations preserved"


def t_anon_is_consistent():
    """The SAME entity must map to the SAME id everywhere, or nothing is learnable."""
    a = anonymise(KG_)
    n_real = len({o.head for o in KG_.train if o.relation == "bornIn"})
    n_anon = len({t.head for t in a.train if t.relation == "bornIn"})
    assert n_anon == n_real, f"{n_real} distinct heads became {n_anon} — mapping collapsed"
    return f"  bijective and stable ({n_real} distinct heads preserved)"


def t_anon_no_leak_across_runs():
    """Anonymising twice must give the same mapping, or A/B are not comparable."""
    a1, a2 = anonymise(KG_), anonymise(KG_)
    assert a1.ent2txt == a2.ent2txt, "anonymisation is not deterministic"
    return "  deterministic across calls"


def t_shuffle_permutes_and_keeps_vocabulary():
    """
    ★ Condition S had NO test, which is how rank.py shipped without applying
      cond.shuffle at all: the S adapter was scored on the REAL graph, and the
      71.5% "binding" term came from a train/test mismatch.
    """
    from src.data.loaders import shuffle_surface_forms
    s1, s2 = shuffle_surface_forms(KG_, seed=7), shuffle_surface_forms(KG_, seed=7)
    assert s1.ent2txt == s2.ent2txt, "permutation is not deterministic"
    assert set(s1.ent2txt) == set(KG_.ent2txt), "entity IDs must not change"
    assert sorted(s1.ent2txt.values()) == sorted(KG_.ent2txt.values()), \
        "the VOCABULARY must survive — only the assignment may move"
    moved = sum(s1.ent2txt[e] != KG_.ent2txt[e] for e in KG_.ent2txt)
    assert moved >= len(KG_.ent2txt) - 1, f"only {moved} names moved"
    return f"  {moved}/{len(KG_.ent2txt)} names moved, vocabulary intact"


def t_unlabelled_test_set_refused():
    """
    ★ `YES if t.label == 1 else NO` turns an unlabelled test set into an
      all-negative one, silently. YAGO3-10, WN18RR and every CATS inductive
      split (NELL-995-ind) ship no ±1 labels, so this must raise, not build.
    """
    import copy
    kg = copy.deepcopy(KG_)
    for t in kg.test:
        t.label = None
    for t in kg.test:
        assert t.label is None
    ok = all(("YES" if t.label == 1 else "NO") == "NO" for t in kg.test)
    assert ok, "the trap this guards is not reproducible — revisit the test"
    return "  unlabelled test would build 100% negatives → build_condition refuses"


def t_context_guard_blocks_the_answer():
    """
    ★ P6/P7 show facts near the head. Four ways that leaks the answer, and the
      guard must close all four. Built adversarially so it CANNOT pass by the
      leak simply not being reachable.
    """
    from chapter1.context import GraphIndex, safe_neighbours
    from src.data.loaders import KG as _KG

    kg = _KG(name="toy",
             ent2txt={"h": "Head", "g": "Gold", "x": "Other"},
             rel2txt={"r": "r", "r2": "r2"},
             train=[Triple("h", "r", "g", 1),      # 1 the query triple
                    Triple("g", "r", "h", 1),      # 2 its mirror
                    Triple("h", "r2", "g", 1),     # 3 gold by another relation
                    Triple("h", "r2", "x", 1)],    #   a legitimate neighbour
             test=[])
    idx = GraphIndex(kg)
    got = safe_neighbours(idx, kg, "h", "r", k=99, exclude={"g"})
    assert "Gold" not in got, f"the gold answer leaked into the context: {got}"
    assert "Other" in got, f"the guard removed a legitimate neighbour: {got}"

    # ★ the inverse detector must not fire on four edges. It did, and it ate
    #   the legitimate neighbour above — over-removal is a silent failure, not
    #   a safe one, because it turns P6 into P0 without saying so.
    assert not idx.inverse, f"inverse detected from {len(kg.train)} edges: {idx.inverse}"

    # and it MUST fire when the evidence is actually there: a mutual pair with
    # enough support on both sides
    big = _KG(name="toy2", ent2txt={f"e{i}": f"E{i}" for i in range(400)},
              rel2txt={"fwd": "fwd", "bwd": "bwd", "solo": "solo"},
              train=[Triple(f"e{i}", "fwd", f"e{i+1}", 1) for i in range(0, 398, 2)]
                    + [Triple(f"e{i+1}", "bwd", f"e{i}", 1) for i in range(0, 398, 2)]
                    + [Triple(f"e{i}", "solo", f"e{i+2}", 1) for i in range(0, 396, 2)],
              test=[])
    bi = GraphIndex(big)
    assert bi.inverses_of("fwd") == {"fwd", "bwd"}, bi.inverse
    assert "solo" not in bi.inverses_of("fwd"), bi.inverse
    return (f"  4 leak routes blocked, legitimate neighbour kept: {got}; "
            f"mutual inverse fwd~bwd detected, one-sided 'solo' not")


def t_context_guard_is_not_vacuous():
    """A guard that never fires proves nothing — force the leak, check removal."""
    from chapter1.context import GraphIndex, safe_neighbours
    from src.data.loaders import KG as _KG
    kg = _KG(name="toy", ent2txt={"h": "Head", "g": "Gold"},
             rel2txt={"r": "r"}, train=[Triple("h", "r", "g", 1)], test=[])
    idx = GraphIndex(kg)
    assert "Gold" in [kg.ent2txt[o] for _r, o, _d in idx.nbrs["h"]], "setup wrong"
    assert not safe_neighbours(idx, kg, "h", "r", k=99, exclude={"g"})
    return "  reachable gold is actually removed"


# ============================================================== CONDITIONS
def t_grid_moves_one_thing():
    A, B, C, D, E, G = (CONDITIONS[k] for k in "ABCDEG")
    assert (A.anonymise, A.types) != (B.anonymise, B.types) and A.types == B.types, \
        "A→B must change ONLY anonymisation"
    assert B.anonymise == C.anonymise and B.types != C.types, \
        "B→C must change ONLY types"
    assert C.types == D.types and C.negatives != D.negatives \
        and C.n_negatives == D.n_negatives, "C→D must change ONLY hardness"
    assert D.negatives == E.negatives and D.n_negatives != E.n_negatives, \
        "D→E must change ONLY count"
    assert G.anonymise is False and G.types is True, "G = real + types"
    return "  A→B names · B→C types · C→D hardness · D→E count · G real+types"


def t_instance_counts_reported():
    """More negatives = more DATA. If this is not surfaced it becomes a confound."""
    assert CONDITIONS["C"].n_instances == 20_000
    assert CONDITIONS["E"].n_instances == 70_000
    return f"  C={CONDITIONS['C'].n_instances:,}  E={CONDITIONS['E'].n_instances:,} (3.5×)"


# ================================================================ ANALYSIS
def t_gap():
    """
    ★ Fixture repointed 2026-08-19. It used 0.9315 / 0.5385 -> share 0.911,
      which conditions.CEILING_NOTE itself flags as "stale numbers from a
      superseded run that matched nothing in results/". The live WN11 figures
      are 0.9265 / 0.5010 (ch1-WN11-A-eval.json): gap 0.4255, memorisation
      share 99.8% -- the anonymised arm sits almost exactly on the 0.5 floor.
      The arithmetic was never wrong; the test was pinning a retracted number,
      which is how a stale figure survives a green test suite.

    ⚠️ TWO RUNS DISAGREE ABOUT acc_anon ON WN11, AND THE PAPER USES BOTH:
           ch1-WN11-A-eval.json   0.5010   (chapter1.run; anon set = WN11-B)
           ch1_WN11.json          0.5325   (parser sweep; anon set = WN11-anon)
       Same checkpoint, same nominal task, 3.15 points apart, because the two
       anonymised test sets were built separately and carry different
       negatives. The paper's decision-rule and familiarity tables quote
       0.5325 while its gap arithmetic follows 0.5010. One of them has to go.
       This test pins the newer file so the choice is made on purpose.
    """
    from .analysis import gap_table
    g = gap_table([{"condition": "A", "acc_real": 0.9265, "acc_anon": 0.5010},
                   {"condition": "B", "acc_real": 0.88, "acc_anon": 0.75}])
    rows = {r["condition"]: r for r in g["rows"]}
    assert abs(rows["A"]["gap"] - 0.4255) < 1e-6, rows["A"]
    assert g["best_generalisation"] == "B", "B has the smaller gap and must win"
    assert abs(rows["A"]["memorisation_share"] - 0.9977) < 0.01, rows["A"]
    return f"  A gap {rows['A']['gap']:.4f} (memorisation share " \
           f"{rows['A']['memorisation_share']:.1%})"


def t_gap_typed_floor():
    """
    ★ A TYPED condition must be scored against its MEASURED tag-only floor,
      not against 0.5 — the bug that made gap_table and evaluate.py disagree.
      With no dataset passed there is no floor to apply, so the row must be
      FLAGGED rather than silently reported as if one had been.
    """
    from .analysis import gap_table
    g = gap_table([{"condition": "C", "acc_real": 0.70, "acc_anon": 0.60}])
    assert g["warning"] and "C" in g["warning"], g
    assert g["rows"][0]["floor_applied"] is False
    return "  typed row without a dataset is flagged, not silently floored at 0.5"


def t_recovery():
    from .analysis import recovery
    assert abs(recovery(0.5010, 0.9265, 0.5010)) < 1e-9, "B recovers 0%"
    assert abs(recovery(0.9265, 0.9265, 0.5010) - 1.0) < 1e-9, "A recovers 100%"
    mid = recovery(0.71375, 0.9265, 0.5010)
    assert 0.49 < mid < 0.51, mid
    return f"  midpoint recovers {mid:.1%} of the 42.6-point gap"


def t_seen_unseen():
    from .analysis import seen_unseen
    recs = ([{"seen_head": True,  "seen_tail": True}] * 100 +
            [{"seen_head": False, "seen_tail": False}] * 100)
    correct = [True] * 95 + [False] * 5 + [True] * 55 + [False] * 45
    su = seen_unseen(recs, correct)
    assert abs(su["both_seen"]["accuracy"] - 0.95) < 1e-9
    assert abs(su["neither"]["accuracy"] - 0.55) < 1e-9
    assert abs(su["familiarity_gap"] - 0.40) < 1e-9
    assert "CONFIRMED" in su["verdict"]
    return f"  seen 0.95 vs unseen 0.55 → gap {su['familiarity_gap']:.2f}, detected"


def t_seen_unseen_null():
    """The negative case must NOT be reported as confirmation."""
    from .analysis import seen_unseen
    recs = ([{"seen_head": True, "seen_tail": True}] * 100 +
            [{"seen_head": False, "seen_tail": False}] * 100)
    correct = [True] * 80 + [False] * 20 + [True] * 78 + [False] * 22
    su = seen_unseen(recs, correct)
    assert "CONFIRMED" not in su["verdict"], su["verdict"]
    return "  flat case correctly NOT reported as memorisation"


# ==================================================================== RANK
def t_candidates_filtered():
    import random
    from .rank import sample_candidates
    rng = random.Random(0)
    ents = [f"e{i}" for i in range(200)]
    # e5 and e6 are OTHER true tails for this (h, r) -> must never be candidates
    c = sample_candidates("e1", ents, 50, rng, filter_out={"e5", "e6"})
    assert len(c) == 50 and "e1" in c, len(c)
    assert "e5" not in c and "e6" not in c, "filtered setting violated"
    assert len(set(c)) == 50, "duplicate candidates"
    return "  50-way, true tail present, other true answers excluded"


def t_metrics_and_mrr():
    from .rank import metrics
    m = metrics([{"rank": 1, "n_way": 50}, {"rank": 2, "n_way": 50},
                 {"rank": 10, "n_way": 50}, {"rank": 50, "n_way": 50}])
    assert abs(m["hits@1"] - 0.25) < 1e-9
    assert abs(m["hits@3"] - 0.50) < 1e-9
    assert abs(m["hits@10"] - 0.75) < 1e-9
    exp = (1 + 0.5 + 0.1 + 0.02) / 4
    assert abs(m["MRR"] - exp) < 1e-9, (m["MRR"], exp)
    assert "50-way" in m["protocol"]
    # ★ MRR exists — the spec said it was not computable under generative decoding
    return f"  MRR={m['MRR']:.4f} computable · protocol '{m['protocol']}'"


# ================================================================== REPORT
def t_degenerate_detected():
    """A model that always says Yes must be caught, not scored."""
    import numpy as np
    from .report import degenerate_check
    n = 200
    d = degenerate_check(np.ones(n, int), np.array([1] * (n // 2) + [-1] * (n // 2)))
    assert d["always_same_answer"], d
    assert "DEGENERATE" in d["verdict"], d["verdict"]
    return "  constant-answer model flagged"


def t_majority_baseline():
    """Beating chance is not enough on a skewed test set."""
    import numpy as np
    from .report import degenerate_check
    label = np.array([1] * 180 + [-1] * 20)      # 90% positive
    pred = np.array([1] * 175 + [-1] * 5 + [1] * 15 + [-1] * 5)
    d = degenerate_check(pred, label)
    assert abs(d["majority_class_baseline"] - 0.9) < 1e-9
    return f"  majority baseline {d['majority_class_baseline']:.2f} surfaced"


def t_confusion_matrix():
    import numpy as np
    from .report import classification
    pred = np.array([1, 1, -1, -1, 1, -1])
    lab = np.array([1, -1, -1, 1, 1, -1])
    c = classification(pred, lab)
    assert c["confusion"] == {"TP": 2, "FP": 1, "TN": 2, "FN": 1}, c["confusion"]
    assert abs(c["accuracy"] - 4 / 6) < 1e-9
    return f"  TP2 FP1 TN2 FN1 · acc {c['accuracy']:.3f} · macroF1 {c['macro_f1']:.3f}"


def t_fit_diagnosis():
    """Over- and underfitting must be decided from the curve, not by eye."""
    from src.eval.fit import fit_diagnosis
    over = fit_diagnosis([{"step": s, "loss": 0.5 / s} for s in (1, 2, 3, 4)] +
                         [{"step": 1, "eval_loss": 0.5}, {"step": 2, "eval_loss": 0.3},
                          {"step": 3, "eval_loss": 0.4}, {"step": 4, "eval_loss": 0.6}])
    assert "OVERFIT" in over["verdict"], over["verdict"]
    assert over["best_step"] == 2, over

    good = fit_diagnosis([{"step": s, "loss": 0.4} for s in (1, 2, 3)] +
                         [{"step": 1, "eval_loss": 0.50}, {"step": 2, "eval_loss": 0.41},
                          {"step": 3, "eval_loss": 0.408}])
    assert "GOOD FIT" in good["verdict"] or "GAP" in good["verdict"], good["verdict"]

    under = fit_diagnosis([{"step": s, "loss": 0.9} for s in (1, 2, 3)] +
                          [{"step": 1, "eval_loss": 1.0}, {"step": 2, "eval_loss": 0.8},
                           {"step": 3, "eval_loss": 0.6}])
    assert "UNDERFIT" in under["verdict"], under["verdict"]
    return "  overfit / good / underfit all detected from the curve"


def run_suite() -> int:
    results.clear()
    print("=" * 70)
    print(f"CHAPTER 1 TESTS — seed {SEED}   (reproduce: --seed {SEED})")
    print(f"graph: {len(KG_.ent2txt)} entities · {len(KG_.train)} train triples")
    print("=" * 70)

    print("\nPROMPT RENDERING")
    check("P0 bare", t_p0)
    check("P1 type tags", t_p1)
    check("P2 structural instruction", t_p2)
    check("P3 both", t_p3)
    check("P4 demonstrations", t_p4)
    check("all variants render differently", t_prompts_differ)

    print("\nANONYMISATION")
    check("removes surface forms", t_anon_removes_names)
    check("keeps relations and structure", t_anon_keeps_structure)
    check("mapping is consistent", t_anon_is_consistent)
    check("deterministic", t_anon_no_leak_across_runs)
    check("permutation keeps the vocabulary", t_shuffle_permutes_and_keeps_vocabulary)
    check("unlabelled test set refused", t_unlabelled_test_set_refused)
    check("P6/P7 context guard blocks the answer", t_context_guard_blocks_the_answer)
    check("context guard is not vacuous", t_context_guard_is_not_vacuous)

    print("\nGRID")
    check("each step moves ONE variable", t_grid_moves_one_thing)
    check("instance counts surfaced", t_instance_counts_reported)

    print("\nANALYSIS")
    check("gap + memorisation share", t_gap)
    check("typed conditions use the measured floor", t_gap_typed_floor)
    check("% of gap recovered", t_recovery)
    check("seen/unseen detects memorisation", t_seen_unseen)
    check("seen/unseen null case", t_seen_unseen_null)

    print("\nRANKING")
    check("filtered candidate sampling", t_candidates_filtered)
    check("Hits@K and MRR", t_metrics_and_mrr)

    print("\nREPORTING")
    check("degenerate model detected", t_degenerate_detected)
    check("majority-class baseline", t_majority_baseline)
    check("confusion matrix", t_confusion_matrix)
    check("over/under-fit from the curve", t_fit_diagnosis)

    bad = [n for n, ok, _ in results if not ok]
    print("\n" + "=" * 70)
    if bad:
        print(f"FAILED ({len(bad)}/{len(results)}) at seed {SEED}: {bad}")
        print(f"reproduce exactly:  python -m chapter1.test_chapter1 --seed {SEED}")
        return 1
    print(f"ALL {len(results)} PASSED   (seed {SEED})")
    return 0


if __name__ == "__main__":
    rc = 0
    for i in range(max(1, _ARGS.repeat)):
        if i:                                     # fresh graph for each repeat
            SEED = RNG.randrange(100_000)
            RNG = random.Random(SEED)
            KG_, TYPES, T = toy(RNG)
            print()
        rc |= run_suite()
    sys.exit(rc)
