"""
★★ THE FROZEN CANDIDATE SETS — the fix that makes "matched cost" true by
   CONSTRUCTION rather than by luck.

THE BUG THIS FILE EXISTS TO PREVENT
-----------------------------------
The first version of `evaluate.py` sampled negatives inline:

    rng = random.Random(ns.seed)
    for q in queries:
        cands = sample_candidates(q["tail"], entities, known, ..., rng)

That is reproducible ONLY while the seed, the query order and the query COUNT
are all identical across cells. The rng state advances once per query, so the
candidate set for query #200 depends on every draw made for queries #1..#199.

Change `--limit`, skip a malformed query, reorder the file, or evaluate S0 on
300 queries and S4 on 250, and the two policies are silently ranking against
DIFFERENT negatives. The MRR difference then contains a sampling difference,
and the chapter's one claim — *same budget, same candidates, only the
allocation differs* — is false without anything visibly going wrong.

★ THE FIX: sample once, write to disk, key by query, and have every cell read
  the same file. Then "identical candidates" is a property of the artefact, not
  a property of a lucky execution order.

    python -m chapter3.candidates --dataset WN18RR-ind --direction tail
    python -m chapter3.candidates --dataset WN18RR-ind --direction both
    python -m chapter3.candidates --dataset WN18RR-ind --verify

FILTERED, AND FILTERED PROPERLY
-------------------------------
A sampled negative is discarded if the triple it forms is TRUE anywhere in
train ∪ valid ∪ test. The previous version omitted `valid`, so genuine facts
living in the validation split were being scored as negatives.

Following CATS, which RealKGC also adopts:
> *"we rank each answer entity against 50 randomly sampled negative entities"*

⚠️ 50-way Hits@1 is NOT full-ranking Hits@1. Every caption must say "50-way".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DIRECTIONS = ("tail", "head")


def qid(head: str, relation: str, tail: str, direction: str) -> str:
    """Stable identity for one ranking query, independent of file order."""
    return f"{direction}|{head}|{relation}|{tail}"


def _sample(gold: str, entities: list[str], true_set: set, fixed: str,
            relation: str, direction: str, n: int, rng) -> list[str]:
    """Gold + (n-1) negatives, filtered against every known true triple."""
    out, guard, seen = [gold], 0, {gold}
    cap = n * 200
    while len(out) < n and guard < cap:
        guard += 1
        e = rng.choice(entities)
        if e in seen:
            continue
        # the candidate triple, in the direction being ranked
        trip = ((fixed, relation, e) if direction == "tail"
                else (e, relation, fixed))
        if trip in true_set:
            continue                      # a true alternative is not a negative
        seen.add(e)
        out.append(e)
    rng.shuffle(out)
    return out


def build(kg, direction: str, n_way: int, limit: int, seed: int) -> dict:
    """
    One candidate set per test query. Deterministic in (dataset, direction,
    n_way, seed) and NOTHING ELSE — in particular not in `limit`, because each
    query gets its own seeded rng derived from its own id.

    ★ That per-query derivation is the whole point: query #200's candidates no
      longer depend on the 199 draws before it, so a cell evaluated on a subset
      sees byte-identical candidates to a cell evaluated on the full set.
    """
    entities = list(kg.ent2txt)
    true_set = kg.all_true()
    rows = {}
    for t in kg.test[:limit]:
        if t.label is not None and t.label < 0:
            continue                       # negative-labelled rows are not queries
        k = qid(t.head, t.relation, t.tail, direction)
        # per-query rng: stable regardless of how many queries precede it
        h = hashlib.sha256(f"{seed}|{k}".encode()).hexdigest()
        rng = random.Random(int(h[:16], 16))
        gold = t.tail if direction == "tail" else t.head
        fixed = t.head if direction == "tail" else t.tail
        cands = _sample(gold, entities, true_set, fixed, t.relation,
                        direction, n_way, rng)
        rows[k] = {"head": t.head, "relation": t.relation, "tail": t.tail,
                   "direction": direction, "gold": gold, "fixed": fixed,
                   "candidates": cands}
    return rows


def path_for(root: str, dataset: str, direction: str, n_way: int, seed: int) -> Path:
    return Path(root, dataset, f"candidates_{direction}_{n_way}way_s{seed}.json")


def load(root: str, dataset: str, direction: str, n_way: int, seed: int) -> dict:
    p = path_for(root, dataset, direction, n_way, seed)
    if not p.exists():
        raise SystemExit(
            f"✋ {p} not built.\n"
            f"   Candidate sets must be FROZEN before any cell is evaluated, or\n"
            f"   policies rank against different negatives and the comparison is\n"
            f"   not matched.\n\n"
            f"   python -m chapter3.candidates --dataset {dataset} "
            f"--direction {direction} --n-way {n_way} --seed {seed}")
    return json.loads(p.read_text(encoding="utf-8"))


def fingerprint(rows: dict) -> str:
    """A short hash of the whole candidate file, printed by every eval run."""
    h = hashlib.sha256()
    for k in sorted(rows):
        h.update(k.encode())
        h.update("".join(rows[k]["candidates"]).encode())
    return h.hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="WN18RR-ind")
    ap.add_argument("--root", default="data")
    ap.add_argument("--direction", nargs="+", default=["tail"],
                    choices=[*DIRECTIONS, "both"])
    ap.add_argument("--n-way", type=int, default=50)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true",
                    help="re-derive and confirm the file on disk is reproducible")
    ns = ap.parse_args()

    from src.data.loaders import load_kg

    kg = load_kg(ns.dataset, ns.root)
    dirs = DIRECTIONS if "both" in ns.direction else ns.direction

    print(f"[cand] {ns.dataset}: {len(kg.ent2txt):,} entities · "
          f"{len(kg.train):,} train · {len(kg.valid):,} valid · "
          f"{len(kg.test):,} test")
    if not kg.valid:
        print("  ⚠️ no valid.tsv — the filtered protocol uses train ∪ test only.")
        print("     Say so in the paper; do not call it 'filtered' without qualification.")
    else:
        print(f"  ✓ filtering against train ∪ valid ∪ test "
              f"({len(kg.all_true()):,} true triples)")

    for d in dirs:
        rows = build(kg, d, ns.n_way, ns.limit, ns.seed)
        p = path_for(ns.root, ns.dataset, d, ns.n_way, ns.seed)

        if ns.verify and p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            same = fingerprint(old) == fingerprint(rows)
            print(f"  {d:5s} verify: {'✓ reproducible' if same else '✗ MISMATCH'} "
                  f"({fingerprint(old)} vs {fingerprint(rows)})")
            if not same:
                raise SystemExit(
                    "the candidate file on disk cannot be re-derived from its seed. "
                    "Results computed against it are not reproducible.")
            continue

        p.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        # how many queries fell short of n_way — a graph too small to sample from
        short = sum(1 for r in rows.values() if len(r["candidates"]) < ns.n_way)
        print(f"  {d:5s} {len(rows):,} queries × {ns.n_way}-way   "
              f"fingerprint {fingerprint(rows)}   -> {p.name}")
        if short:
            print(f"        ⚠️ {short} queries have fewer than {ns.n_way} candidates "
                  f"(filtering removed too many). Their ranks are not comparable "
                  f"to the rest — evaluate.py reports them separately.")

    print("\n  ★ Every (policy, budget) cell now ranks against these EXACT sets.")
    print("    Pass the same --n-way and --seed to chapter3.evaluate, and it will")
    print("    print this fingerprint so a mismatch is impossible to miss.")


if __name__ == "__main__":
    main()
