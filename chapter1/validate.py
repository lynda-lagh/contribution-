"""
★ HARD VALIDATION — run after ANY modification to a dataset.

    python -m chapter1.validate --dataset YAGO3-10
    python -m chapter1.validate --dataset WN11 FB13 YAGO3-10

Exits non-zero on failure, so it can gate a notebook cell.

`profile_data` DESCRIBES a dataset so you can design an experiment on it.
This module DECIDES whether the dataset is well-formed enough to use at all --
especially after `make_test_negatives` has rewritten `test.tsv`.

Checks, in order of how badly each one would corrupt a result:

  ✗ FATAL     test labels absent or one-class      -> accuracy measures nothing
  ✗ FATAL     train/test triple overlap            -> leakage; memorisation is free
  ✗ FATAL     unknown entity or relation ids       -> silent fallback to raw id text
  ⚠ WARN      duplicate triples, self-loops        -> inflates counts
  ⚠ WARN      class imbalance                      -> accuracy misleads
  ⚠ WARN      entities with no description         -> enrichment has nothing to use
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

FATAL, WARN, OK = "✗ FATAL", "⚠ WARN ", "✓ ok   "


class Report:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level, name, detail=""):
        self.rows.append((level, name, detail))

    @property
    def fatal(self) -> int:
        return sum(1 for l, _, _ in self.rows if l == FATAL)

    @property
    def warn(self) -> int:
        return sum(1 for l, _, _ in self.rows if l == WARN)

    def show(self):
        print(f"\n{'=' * 78}\n  VALIDATE  {self.dataset}\n{'=' * 78}")
        for level, name, detail in self.rows:
            print(f"  {level}  {name:<38} {detail}")
        v = ("PASS" if not self.fatal else "FAIL")
        print(f"\n  {v} — {self.fatal} fatal, {self.warn} warnings")


def _read_map(p: Path) -> tuple[dict, list[str]]:
    m, problems = {}, []
    for i, line in enumerate(p.open(encoding="utf-8"), 1):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            problems.append(f"line {i}: fewer than 2 columns")
            continue
        if parts[0] in m:
            problems.append(f"line {i}: duplicate key {parts[0]!r}")
        m[parts[0]] = parts[1]
    return m, problems


def _read_triples(p: Path) -> tuple[list[tuple], list[str]]:
    out, problems = [], []
    for i, line in enumerate(p.open(encoding="utf-8"), 1):
        c = line.rstrip("\n").split("\t")
        if len(c) < 3:
            problems.append(f"line {i}: fewer than 3 columns")
            continue
        lab = None
        if len(c) > 3 and c[3].strip():
            try:
                lab = int(c[3])
            except ValueError:
                problems.append(f"line {i}: label {c[3]!r} is not an integer")
        out.append((c[0], c[1], c[2], lab))
    return out, problems


def validate(dataset: str, root: str = "data") -> Report:
    r = Report(dataset)
    d = Path(root, dataset)

    # ---- files -----------------------------------------------------------
    need = ["entity2text.txt", "relation2text.txt", "train.tsv", "test.tsv"]
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        r.add(FATAL, "required files", f"missing {missing}")
        return r
    r.add(OK, "required files", f"all 4 present in {d}")

    ent, ep = _read_map(d / "entity2text.txt")
    rel, rp = _read_map(d / "relation2text.txt")
    train, tp = _read_triples(d / "train.tsv")
    test, sp = _read_triples(d / "test.tsv")

    for name, probs in (("entity2text", ep), ("relation2text", rp),
                        ("train.tsv", tp), ("test.tsv", sp)):
        if probs:
            r.add(WARN, f"{name} parse", f"{len(probs)} malformed: {probs[0]}")
    r.add(OK, "sizes", f"{len(ent):,} entities · {len(rel)} relations · "
                       f"{len(train):,} train · {len(test):,} test")

    # ---- ★ test labels ---------------------------------------------------
    labs = Counter(t[3] for t in test)
    pos, neg, none = labs.get(1, 0), labs.get(-1, 0), labs.get(None, 0)
    if none:
        r.add(FATAL, "test labels",
              f"{none:,} rows unlabelled -> every gold answer becomes 'No'. "
              f"Run scripts/make_test_negatives.py")
    elif pos == 0 or neg == 0:
        r.add(FATAL, "test labels", f"one class only (+{pos:,} / −{neg:,})")
    else:
        bal = pos / (pos + neg)
        lvl = OK if 0.4 <= bal <= 0.6 else WARN
        r.add(lvl, "test labels",
              f"+{pos:,} / −{neg:,} · balance {bal:.1%} · "
              f"majority baseline {max(bal, 1-bal):.3f}")

    # ---- ★ leakage -------------------------------------------------------
    tr = {(a, b, c) for a, b, c, _ in train}
    te_pos = {(a, b, c) for a, b, c, l in test if l != -1}
    overlap = tr & te_pos
    if overlap:
        r.add(FATAL, "train/test overlap",
              f"{len(overlap):,} positive test triples also in train "
              f"({len(overlap)/max(1,len(te_pos)):.1%}) -> memorisation is free "
              f"and the seen/unseen split is meaningless")
    else:
        r.add(OK, "train/test overlap", "none — no direct leakage")

    # ---- ★ generated negatives that are actually true --------------------
    te_neg = {(a, b, c) for a, b, c, l in test if l == -1}
    bad_neg = tr & te_neg
    if bad_neg:
        # Severity scales with the FRACTION, not the count. WN11 ships 7 such rows
        # out of 10,544 (0.07%) -- a defect in the benchmark itself, worth a
        # footnote but not a reason to refuse to train. A large fraction means the
        # negative generator is broken and every "negative" is suspect.
        frac = len(bad_neg) / max(1, len(te_neg))
        lvl = FATAL if frac > 0.01 else WARN
        r.add(lvl, "negatives that are true",
              f"{len(bad_neg):,} of {len(te_neg):,} negatives ({frac:.2%}) also "
              f"appear in train — true facts labelled false. "
              + ("Generator is broken." if lvl is FATAL else
                 "Benchmark-level noise; report it as a known label defect."))
    elif te_neg:
        r.add(OK, "negatives vs train", f"{len(te_neg):,} negatives, none in train")

    # ---- id coverage -----------------------------------------------------
    used_e = {x for a, b, c, _ in train + test for x in (a, c)}
    used_r = {b for _, b, _, _ in train + test}
    unk_e, unk_r = used_e - set(ent), used_r - set(rel)
    if unk_e:
        r.add(FATAL, "unknown entity ids",
              f"{len(unk_e):,} ids absent from entity2text (e.g. {list(unk_e)[:2]}) "
              f"-> prompts silently fall back to the raw id")
    else:
        r.add(OK, "entity id coverage", f"all {len(used_e):,} used ids resolve")
    if unk_r:
        r.add(FATAL, "unknown relation ids", f"{sorted(unk_r)[:3]}")
    else:
        r.add(OK, "relation id coverage", f"all {len(used_r)} used relations resolve")

    # ---- hygiene ---------------------------------------------------------
    dup_tr = len(train) - len(tr)
    if dup_tr:
        r.add(WARN, "duplicate train triples", f"{dup_tr:,}")
    self_loops = sum(1 for a, _, c, _ in train + test if a == c)
    if self_loops:
        r.add(WARN, "self-loops", f"{self_loops:,} triples where head == tail")

    empty_desc = sum(1 for v in ent.values() if not v.strip())
    if empty_desc:
        r.add(WARN, "empty descriptions", f"{empty_desc:,} entities")

    # ---- backup present after modification --------------------------------
    if (d / "test.original.tsv").exists():
        orig, _ = _read_triples(d / "test.original.tsv")
        r.add(OK, "original test preserved",
              f"test.original.tsv · {len(orig):,} rows (was unlabelled)")
        if pos and len(orig) != pos:
            r.add(WARN, "positive count changed",
                  f"{len(orig):,} original vs {pos:,} positives kept "
                  f"(--limit was used?)")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["YAGO3-10"])
    ap.add_argument("--root", default="data")
    ns = ap.parse_args()

    bad = 0
    for d in ns.dataset:
        rep = validate(d, ns.root)
        rep.show()
        bad += rep.fatal
    if bad:
        print(f"\n★ {bad} fatal problem(s). Do not train on this.")
        sys.exit(1)
    print("\n★ All datasets valid.")


if __name__ == "__main__":
    main()
