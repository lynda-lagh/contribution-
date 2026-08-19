"""
★★ PREFLIGHT — refuse to start a run that cannot produce a valid number.

    python -m chapter1.preflight --dataset YAGO3-10 --require-semantic

Every check here corresponds to a failure that has ALREADY happened in this
project and that was silent at the time:

    1  dataset files          missing test.tsv -> load_kg raises 40 min in
    2  test labels            unlabelled test -> every instance built NEGATIVE
    3  type source            no semantic types -> C silently becomes induced
    4  type leak              tag-only rule explains the typed result for free
    5  S actually permutes    rank.py ignored cond.shuffle -> S == A
    6  C differs from B       types rendered nothing -> C byte-identical to B
    7  arms are matched       C and G must see the SAME tag inventory
    8  writable checkpoints   training completes, save fails, work lost

Exit code is non-zero on the first failure, so a notebook cell cannot sail past
it. There are NO fallbacks in this module by design: its whole purpose is to
convert a silent downgrade into a stop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OK, BAD = "  ✓", "  ✗"


class Fail(Exception):
    """A failed check.

    ★ NOT SystemExit. SystemExit derives from BaseException, not Exception, so
      `except Exception` does not catch it: raising it inside a check killed the
      process on the FIRST failure, skipped every later check, recorded nothing
      in `fails` and never printed the summary. The report then looked like it
      had simply stopped early rather than failed.
    """


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--root", default="data")
    ap.add_argument("--require-semantic", action="store_true",
                    help="fail if exogenous types are unavailable")
    ap.add_argument("--max-other", type=float, default=0.5)
    ap.add_argument("--max-leak", type=float, default=0.55)
    ns = ap.parse_args()

    ds, root = ns.dataset, ns.root
    fails: list[str] = []

    def check(name: str, fn):
        try:
            msg = fn()
        except Exception as exc:                                # noqa: BLE001
            fails.append(name)
            # ★ keep the report READABLE. An nltk LookupError carries a 20-line
            #   banner; printed raw it buries every check after it and the
            #   summary scrolls off screen, which is how a failing preflight
            #   gets mistaken for a passing one.
            first = str(exc).strip().splitlines()
            head = first[0] if first else type(exc).__name__
            print(f"{BAD} {name}\n      {type(exc).__name__}: {head[:160]}")
            for extra in first[1:4]:
                if extra.strip():
                    print(f"      {extra.strip()[:160]}")
            return None
        print(f"{OK} {name}   {msg or ''}")
        return msg

    print("=" * 74)
    print(f"PREFLIGHT — {ds}" + ("   [--require-semantic]" if ns.require_semantic else ""))
    print("=" * 74)

    from src.data.loaders import anonymise, load_kg, shuffle_surface_forms
    from .conditions import CONDITIONS, TYPE_TAG_FLOOR

    # 1 ------------------------------------------------------------------
    kg = check("dataset files present", lambda: (
        (lambda k: f"{len(k.ent2txt):,} entities, {len(k.rel2txt)} relations, "
                   f"{len(k.train):,} train, {len(k.test):,} test")(load_kg(ds, root))))
    if kg is None:
        sys.exit(f"\n{len(fails)} check(s) failed — cannot continue")
    kg = load_kg(ds, root)

    # 2 ------------------------------------------------------------------
    def labels():
        n = sum(1 for t in kg.test if t.label is not None)
        if n == 0:
            raise Fail(
                "test set has NO ±1 labels. Every test instance would be built "
                "as a NEGATIVE.\n      python -m scripts.make_test_negatives "
                f"--dataset {ds} --strategy type_consistent")
        pos = sum(1 for t in kg.test if t.label == 1) / n
        if abs(pos - 0.5) > 0.02:
            raise Fail(f"test set is {pos:.1%} positive, not balanced — "
                       f"accuracy is then NOT balanced accuracy")
        return f"{n:,} labelled, {pos:.1%} positive (balanced)"
    check("test set is labelled and balanced", labels)

    # 3 ------------------------------------------------------------------
    def types():
        from src.routing.semantic_types import coverage, provider_for, semantic_types
        if provider_for(ds) is None:
            raise Fail(f"no exogenous type source registered for {ds}")
        t = semantic_types(kg, ds, root=root)
        r = coverage(t)
        if r["other_rate"] > ns.max_other:
            raise Fail(
                f"OTHER={r['other_rate']:.1%} exceeds {ns.max_other:.0%} — "
                f"the typed conditions would be near-vacuous")
        if r["n_distinct"] < 2:
            raise Fail("fewer than 2 distinct types")
        return (f"{r['n_distinct']} types, OTHER {r['other_rate']:.1%}, "
                f"largest {r['largest_share']:.1%}")
    if ns.require_semantic:
        check("EXOGENOUS semantic types available", types)
    else:
        print("  - semantic types not required (induced fallback permitted)")

    # 4 ------------------------------------------------------------------
    def leak():
        f = TYPE_TAG_FLOOR.get(ds)
        if f is None:
            raise Fail(
                f"no measured type-tag floor for {ds}. Run "
                f"`python -m chapter1.check_type_leak --dataset {ds}` and add it "
                f"to TYPE_TAG_FLOOR — an unmeasured floor is added silently to "
                f"every typed result.")
        if f >= ns.max_leak:
            raise Fail(
                f"tag-only rule scores {f:.3f} >= {ns.max_leak} — a one-line "
                f"heuristic would explain most of any typed result.\n"
                f"      python -m scripts.make_test_negatives --dataset {ds} "
                f"--strategy type_consistent --regenerate")
        return f"tag-only floor {f:.3f} < {ns.max_leak}"
    check("type-tag leak is under control", leak)

    # 5 ------------------------------------------------------------------
    def permutes():
        s = shuffle_surface_forms(kg, seed=42)
        if set(s.ent2txt) != set(kg.ent2txt):
            raise Fail("permutation changed the entity IDs")
        if sorted(s.ent2txt.values()) != sorted(kg.ent2txt.values()):
            raise Fail("permutation did not preserve the vocabulary")
        moved = sum(s.ent2txt[e] != kg.ent2txt[e] for e in kg.ent2txt)
        if moved < 0.99 * len(kg.ent2txt):
            raise Fail(f"only {moved:,}/{len(kg.ent2txt):,} names moved")
        import inspect

        from . import rank
        if "shuffle" not in inspect.getsource(rank.main):
            raise Fail(
                "chapter1/rank.py does not apply cond.shuffle — condition S "
                "would be ranked on the REAL graph (train/test mismatch)")
        return f"{moved:,} names moved, vocabulary intact, rank.py applies it"
    check("condition S really permutes", permutes)

    # 6 + 7 --------------------------------------------------------------
    def arms_differ():
        from src.routing.semantic_types import semantic_types
        from .conditions import PROMPTS
        from .data import render
        t = kg.test[0]
        try:
            ty = semantic_types(kg, ds, root=root)
        except Exception:                                       # noqa: BLE001
            if ns.require_semantic:
                raise
            from src.routing.types import entity_types
            ty = entity_types(kg, method="induced")
        bare = render(t, kg, PROMPTS["P0"], None, None)
        typed = render(t, kg, PROMPTS["P1"], ty, None)
        if bare == typed:
            raise Fail("typed and untyped prompts are BYTE-IDENTICAL — "
                       "condition C would silently equal B")
        # C (anon) and G (real) must carry the SAME tag inventory
        try:
            ty_anon = semantic_types(anonymise(kg), ds, root=root)
        except Exception:                                       # noqa: BLE001
            from src.routing.types import entity_types
            ty_anon = entity_types(anonymise(kg), method="induced")
        if ty != ty_anon:
            n = sum(1 for k in ty if ty.get(k) != ty_anon.get(k))
            raise Fail(
                f"types are NOT invariant under anonymisation ({n:,} differ) — "
                f"C and G would not be a matched pair")
        return "C != B, and C/G share one tag inventory"
    check("typed arms differ from untyped, and match each other", arms_differ)

    # 8 ------------------------------------------------------------------
    def writable():
        from src.utils.config import load_config
        d = Path(load_config("configs/base.yaml")["output"]["adapter_dir"])
        d.mkdir(parents=True, exist_ok=True)
        p = d / ".preflight"
        p.write_text("ok", encoding="utf-8")
        p.unlink()
        return f"{d.resolve()} is writable"
    check("checkpoint directory is writable", writable)

    print("=" * 74)
    if fails:
        sys.exit(f"✗ {len(fails)} FAILED: {', '.join(fails)}\n"
                 f"  Fix these before spending GPU time. Nothing here has a "
                 f"fallback on purpose.")
    print("✓ ALL CHECKS PASSED — safe to build and train")


if __name__ == "__main__":
    main()
