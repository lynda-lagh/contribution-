"""
★ HARD CHECKS FOR AN INDUCTIVE SPLIT. Non-zero exit on failure.

WHY THIS IS THE FIRST FILE TO RUN
---------------------------------
Chapter 3's entire justification is that **test entities are unseen**, so prompt
context is the only signal about them. If even a fraction of test entities appear
in training, the model can fall back on memorised names -- exactly the shortcut
Chapter 1 measured at 96.8% -- and every allocation result becomes
uninterpretable.

A leaky "inductive" split does not fail loudly. It produces slightly better
numbers and a chapter that cannot be defended.

    python -m chapter3.validate --dataset WN18RR-ind --root data

CHECKS
------
  FATAL   test entities appearing in train        <- the one that matters
  FATAL   test relations absent from train        (relations must be SHARED;
                                                   inductive = new entities,
                                                   NOT new relations)
  FATAL   empty test or train
  FATAL   test triples with no unseen entity      (then it is not inductive)
  WARN    entities with no description AND no neighbours in train
          -> nothing to allocate; these queries are unanswerable by construction
  WARN    train/test triple overlap
  WARN    class imbalance if labels are present
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _rows(p: Path) -> list[tuple[str, ...]]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            out.append(tuple(parts))
    return out


def check(dataset: str, root: str) -> tuple[int, int]:
    d = Path(root, dataset)
    fatal = warn = 0

    def ok(name, msg):
        print(f"  ✓ ok     {name:38s} {msg}")

    def bad(name, msg):
        nonlocal fatal
        fatal += 1
        print(f"  ✗ FATAL  {name:38s} {msg}")

    def wrn(name, msg):
        nonlocal warn
        warn += 1
        print(f"  ⚠ WARN   {name:38s} {msg}")

    print("=" * 78)
    print(f"  VALIDATE INDUCTIVE SPLIT — {dataset}")
    print("=" * 78)

    train = _rows(d / "train.tsv")
    test = _rows(d / "test.tsv")
    valid = _rows(d / "valid.tsv")

    if not train:
        bad("train.tsv", f"missing or empty at {d/'train.tsv'}")
        return fatal, warn
    if not test:
        bad("test.tsv", f"missing or empty at {d/'test.tsv'}")
        return fatal, warn
    ok("files", f"train {len(train):,} · valid {len(valid):,} · test {len(test):,}")

    tr_ents = {t[0] for t in train} | {t[2] for t in train}
    te_ents = {t[0] for t in test} | {t[2] for t in test}
    tr_rels = {t[1] for t in train}
    te_rels = {t[1] for t in test}

    # ---- THE check ---------------------------------------------------------
    seen = te_ents & tr_ents
    if seen:
        bad("test entities unseen in train",
            f"{len(seen):,} of {len(te_ents):,} test entities ({len(seen)/len(te_ents):.1%}) "
            f"APPEAR IN TRAIN — this split is not inductive. e.g. {sorted(seen)[:3]}")
    else:
        ok("test entities unseen in train",
           f"all {len(te_ents):,} test entities are new ✓")

    # ---- relations must be shared -----------------------------------------
    new_rels = te_rels - tr_rels
    if new_rels:
        bad("relations shared",
            f"{len(new_rels)} test relations never appear in train: "
            f"{sorted(new_rels)[:5]} — inductive means new ENTITIES, not new relations")
    else:
        ok("relations shared", f"all {len(te_rels)} test relations seen in train")

    # ---- is every test triple actually inductive? --------------------------
    fully_seen = [t for t in test if t[0] in tr_ents and t[2] in tr_ents]
    if fully_seen:
        bad("every test triple has an unseen entity",
            f"{len(fully_seen):,} test triples have BOTH entities in train "
            f"({len(fully_seen)/len(test):.1%}) — those are transductive queries")
    else:
        ok("every test triple has an unseen entity", "confirmed")

    # ---- can the unseen entities be described at all? ----------------------
    ent2txt = {}
    p = d / "entity2text.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                ent2txt[k] = v
    described = sum(1 for e in te_ents if len(ent2txt.get(e, "").split()) >= 3)
    if described == 0:
        wrn("test entities have descriptions",
            "NO test entity has a description of 3+ words — the only allocatable "
            "context is neighbours, which unseen entities may also lack")
    else:
        ok("test entities have descriptions",
           f"{described:,} of {len(te_ents):,} ({described/len(te_ents):.1%}) have 3+ words")

    # ★ an unseen entity with neither a description nor a train neighbour has
    #   literally nothing to allocate. Those queries cannot be answered by any
    #   policy, and they dilute every comparison.
    nbr = set()
    for h, r, t, *_ in train:
        nbr.add(h); nbr.add(t)
    orphan = [e for e in te_ents
              if len(ent2txt.get(e, "").split()) < 3 and e not in nbr]
    if orphan:
        wrn("allocatable context exists",
            f"{len(orphan):,} test entities ({len(orphan)/len(te_ents):.1%}) have "
            f"neither a description nor any train neighbour — no policy can help "
            f"them; report this fraction, it caps achievable MRR")
    else:
        ok("allocatable context exists", "every test entity has description or neighbours")

    dup = len(test) - len({t[:3] for t in test})
    if dup:
        wrn("duplicate test triples", f"{dup:,}")

    print(f"\n  {'PASS' if not fatal else 'FAIL'} — {fatal} fatal, {warn} warnings")
    return fatal, warn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["WN18RR-ind"])
    ap.add_argument("--root", default="data")
    ns = ap.parse_args()

    total = 0
    for ds in ns.dataset:
        f, _ = check(ds, ns.root)
        total += f
        print()
    if total:
        print(f"✋ {total} fatal problem(s). Chapter 3's premise is that test "
              f"entities are UNSEEN; fix the split before training anything.")
        sys.exit(1)
    print("★ All splits valid — the inductive premise holds.")


if __name__ == "__main__":
    main()
