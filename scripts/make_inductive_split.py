"""
★ Build a fully inductive split from any dataset already on disk.

    python -m scripts.make_inductive_split --source WN11 --out WN18RR-ind
    python -m scripts.make_inductive_split --source WN11 --out WN11-ind --frac 0.20

WHY THIS EXISTS
---------------
CATS ships **code only**; its splits live in a Google Drive folder that cannot be
fetched from a notebook without manual steps (see scripts/fetch_cats_splits.py).
This module builds an equivalent split from a graph you already have, so the
pipeline is never blocked on an external download.

⚠️ WHAT YOU GIVE UP, AND WHY IT DOES NOT MATTER HERE
   A split built locally is NOT byte-identical to CATS's, so absolute numbers are
   not comparable to their table. Chapter 3's claim is *internal* --- S0 versus R
   versus the policies, at a matched token budget, on one split --- so it needs an
   inductive split, not CATS's specific one. Say this in the paper rather than
   implying comparability you do not have.

THE CONSTRUCTION (GraIL / CATS family)
--------------------------------------
    1. sample a fraction of entities and mark them UNSEEN
    2. train.tsv = triples whose head AND tail are both seen
    3. test.tsv  = triples touching at least one unseen entity
                   -> these are both the QUERIES and the inference graph
                      that makes an unseen entity answerable at all
    4. valid.tsv = a held-out slice of (2), used only for filtered ranking

★ The guarantee that matters: no entity appearing as a test query head occurs
  anywhere in train.tsv. That is checked here and again by chapter3.validate,
  which exits non-zero.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def read_triples(p: Path) -> list[tuple[str, str, str]]:
    out = []
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                out.append((parts[0], parts[1], parts[2]))
    return out


def read_map(p: Path) -> dict[str, str]:
    out = {}
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="existing dataset dir name, e.g. WN11")
    ap.add_argument("--out", required=True, help="new dataset dir name, e.g. WN18RR-ind")
    ap.add_argument("--root", default="data")
    ap.add_argument("--frac", type=float, default=0.20,
                    help="fraction of entities held out as UNSEEN")
    ap.add_argument("--valid-frac", type=float, default=0.15,
                    help="share of the unseen-touching triples reserved for filtering")
    ap.add_argument("--min-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ns = ap.parse_args()

    src = Path(ns.root, ns.source)
    dst = Path(ns.root, ns.out)
    if not (src / "train.tsv").exists():
        raise SystemExit(f"✋ {src}/train.tsv not found. Available: "
                         f"{[p.name for p in Path(ns.root).glob('*') if p.is_dir()]}")

    ent2txt = read_map(src / "entity2text.txt")
    rel2txt = read_map(src / "relation2text.txt")
    triples = read_triples(src / "train.tsv")
    # positive test triples of the source count as facts too
    for h, r, t in read_triples(src / "test.tsv"):
        triples.append((h, r, t))
    triples = list(dict.fromkeys(triples))          # dedupe, keep order

    ents = sorted({e for h, _, t in triples for e in (h, t)})
    rng = random.Random(ns.seed)

    print(f"[split] source {ns.source}: {len(ents):,} entities · "
          f"{len(triples):,} triples · {len(rel2txt)} relations")

    # ---- 1 · choose the unseen entities ------------------------------------
    n_unseen = max(1, int(len(ents) * ns.frac))
    unseen = set(rng.sample(ents, n_unseen))

    # ---- 2/3 · partition -----------------------------------------------------
    train, touching = [], []
    for h, r, t in triples:
        (touching if (h in unseen or t in unseen) else train).append((h, r, t))

    # entities that actually survive in train
    seen_in_train = {e for h, _, t in train for e in (h, t)}

    # ★ a query is only valid if its HEAD is genuinely unseen in train
    queries = [(h, r, t) for h, r, t in touching
               if h in unseen and h not in seen_in_train]
    support = [x for x in touching if x not in set(queries)]

    if len(queries) < ns.min_test:
        raise SystemExit(
            f"✋ only {len(queries)} valid queries (need {ns.min_test}). "
            f"Raise --frac (currently {ns.frac}).")

    rng.shuffle(queries)
    n_valid = int(len(queries) * ns.valid_frac)
    valid_q, test_q = queries[:n_valid], queries[n_valid:]

    # test.tsv carries the queries AND the support facts, because an unseen
    # entity is only answerable if its local graph is observable at test time
    test_rows = test_q + support

    # ---- 4 · write ----------------------------------------------------------
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "train.tsv").write_text(
        "\n".join("\t".join(x) for x in train) + "\n", encoding="utf-8")
    (dst / "test.tsv").write_text(
        "\n".join("\t".join(x) + "\t1" for x in test_rows) + "\n", encoding="utf-8")
    (dst / "valid.tsv").write_text(
        "\n".join("\t".join(x) + "\t1" for x in valid_q) + "\n", encoding="utf-8")

    used = {e for h, _, t in triples for e in (h, t)}
    (dst / "entity2text.txt").write_text(
        "\n".join(f"{e}\t{ent2txt.get(e, e)}" for e in sorted(used)) + "\n",
        encoding="utf-8")
    (dst / "relation2text.txt").write_text(
        "\n".join(f"{r}\t{rel2txt.get(r, r.strip('_').replace('_', ' '))}"
                  for r in sorted({r for _, r, _ in triples})) + "\n",
        encoding="utf-8")

    # ---- 5 · verify the premise --------------------------------------------
    q_heads = {h for h, _, _ in test_q}
    leaked = q_heads & seen_in_train
    print(f"\n[split] train      {len(train):,} triples · "
          f"{len(seen_in_train):,} entities")
    print(f"[split] test       {len(test_rows):,} rows "
          f"({len(test_q):,} queries + {len(support):,} support facts)")
    print(f"[split] valid      {len(valid_q):,} triples (filtering only)")
    print(f"[split] unseen     {len(unseen):,} entities held out "
          f"({ns.frac:.0%} of {len(ents):,})")
    print(f"\n[split] {'✅' if not leaked else '✋'} query heads seen in train: "
          f"{len(leaked)}  (must be 0)")
    if leaked:
        raise SystemExit("the split is not inductive — this is a bug, not a warning")

    # how many queries have any observable context at all
    nbr = defaultdict(int)
    for h, r, t in test_rows:
        nbr[h] += 1
        nbr[t] += 1
    with_ctx = sum(1 for h, _, _ in test_q if nbr[h] > 1)
    print(f"[split] queries with an observable neighbour: {with_ctx:,}/{len(test_q):,} "
          f"({with_ctx/max(1,len(test_q)):.0%})")
    if with_ctx / max(1, len(test_q)) < 0.5:
        print("  ⚠️ under half the queries have context. Chapter 3 allocates context,")
        print("     so a low figure here caps what any policy can do. Raise --frac.")

    print(f"\nwrote {dst}")
    print(f"  next: python -m chapter3.validate --dataset {ns.out}")


if __name__ == "__main__":
    main()
