"""
★ Convert a CATS dataset folder into this project's format.

    python -m scripts.convert_cats --src /kaggle/working/CATS-data/datasets/WN18RR-subset-inductive --out data/WN18RR-ind

WHAT CATS SHIPS, AND HOW IT MAPS
--------------------------------
    train_full.txt        the training graph                 -> train.tsv
    test.txt              the ranking queries                -> test.tsv
    inductive_graph.txt   facts about UNSEEN entities        -> valid.tsv  ★
    entity2text.txt       id \\t description                  -> copied
    relation2text.txt     id \\t surface form                 -> copied
    ranking_tail.txt      ★ 50 candidates per test query     -> candidates_*.json
    ranking_head.txt      ★ same, head direction             -> candidates_*.json

★ WHY `inductive_graph.txt` BECOMES valid.tsv
   An unseen entity is only answerable if something about it is observable at
   test time; that file is the observable part. Mapping it to valid.tsv gives it
   two correct roles at once: it is filtered against during ranking (those
   triples are TRUE, so they are not negatives), and `GraphIndex` reads it as
   the support graph from which neighbour blocks are built. Discarding it would
   leave every test entity with no context, which is the exact bug that made an
   earlier version of this pipeline report a false null result.

★★ WHY WE USE CATS'S RANKING FILES INSTEAD OF SAMPLING OUR OWN
   `ranking_tail.txt` holds exactly 50 candidate triples per query. Adopting
   them makes our numbers comparable to CATS and to RealKGC, which states it
   uses "the specific dataset versions and splits as processed in CATS".
   Sampling our own 50 would be defensible but would forfeit that comparison for
   nothing.

⚠️ THE ONE THING TO WATCH: n. WN18RR-subset-inductive has 188 test queries. At
   that size the smallest detectable paired difference is roughly 0.04 MRR, so
   small contrasts will not be resolvable. NELL-995-subset-inductive has 476 and
   is the better choice if statistical power matters more than the WordNet
   structure S5 needs. The converter prints this so the choice is explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def read_triples(p: Path) -> list[tuple[str, str, str]]:
    out = []
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                out.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return out


def write_triples(p: Path, rows, label: str | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for h, r, t in rows:
            f.write(f"{h}\t{r}\t{t}" + (f"\t{label}" if label else "") + "\n")


def blocks_of(rows, n: int):
    for i in range(0, len(rows) - n + 1, n):
        yield rows[i:i + n]


def convert_ranking(rank_file: Path, queries, direction: str, n_way: int):
    """
    CATS stores `n_way` consecutive candidate triples per query. Returns rows in
    the frozen-candidates format used by chapter3.candidates, plus a report.

    The grouping is verified rather than assumed: for the tail direction, every
    line in a block must share (head, relation) with the query, and the gold
    tail must be present. A block that fails is reported and skipped.
    """
    rank = read_triples(rank_file)
    if not rank:
        return {}, {"status": "missing"}

    n_blocks = len(rank) // n_way
    rep = Counter()
    out = {}

    for i, blk in enumerate(blocks_of(rank, n_way)):
        if i >= len(queries):
            break
        qh, qr, qt = queries[i]
        gold = qt if direction == "tail" else qh
        fixed = qh if direction == "tail" else qt

        cands = [t if direction == "tail" else h for h, r, t in blk]
        # verify the block really belongs to this query
        if direction == "tail":
            consistent = all(h == qh and r == qr for h, r, _ in blk)
        else:
            consistent = all(t == qt and r == qr for _, r, t in blk)

        if not consistent:
            rep["block_mismatch"] += 1
            continue
        if gold not in cands:
            rep["gold_missing"] += 1
            continue

        rep["ok"] += 1
        key = f"{direction}|{qh}|{qr}|{qt}"
        out[key] = {"head": qh, "relation": qr, "tail": qt,
                    "direction": direction, "gold": gold, "fixed": fixed,
                    "candidates": cands}

    rep["n_blocks"] = n_blocks
    rep["n_queries"] = len(queries)
    return out, dict(rep)


def fingerprint(rows: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(rows):
        h.update(k.encode())
        h.update("".join(rows[k]["candidates"]).encode())
    return h.hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="a CATS dataset directory")
    ap.add_argument("--out", required=True, help="e.g. data/WN18RR-ind")
    ap.add_argument("--train", default="train_full.txt",
                    help="train_full.txt | train_2000.txt | train_1000.txt")
    ap.add_argument("--n-way", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42,
                    help="only used to name the candidate file; sets are CATS's")
    ns = ap.parse_args()

    src, out = Path(ns.src), Path(ns.out)
    if not src.exists():
        raise SystemExit(f"✋ {src} not found")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  CONVERT  {src.name}  ->  {out}")
    print("=" * 78)

    # ---- core files ---------------------------------------------------------
    train = read_triples(src / ns.train)
    test = read_triples(src / "test.txt")
    ind = read_triples(src / "inductive_graph.txt")
    valid_native = read_triples(src / "valid.txt")

    if not train:
        raise SystemExit(f"✋ {src/ns.train} is empty or missing")

    # ★ the inference graph is the support; if this dataset has none, fall back
    #   to its own valid split so that filtering still has something to work with
    support = ind if ind else valid_native
    support_src = "inductive_graph.txt" if ind else "valid.txt"

    write_triples(out / "train.tsv", train)
    write_triples(out / "test.tsv", test, label="1")
    write_triples(out / "valid.tsv", support, label="1")

    for name in ("entity2text.txt", "relation2text.txt"):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)

    ents = {e for h, _, t in train + test + support for e in (h, t)}
    tr_ents = {e for h, _, t in train for e in (h, t)}
    te_ents = {e for h, _, t in test for e in (h, t)}

    print(f"  train.tsv   {len(train):>7,}  from {ns.train}")
    print(f"  test.tsv    {len(test):>7,}  the ranking queries")
    print(f"  valid.tsv   {len(support):>7,}  from {support_src}  ★ the support graph")
    print(f"  entities    {len(ents):>7,}   relations {len({r for _,r,_ in train}):>3}")

    # ---- the inductive premise ---------------------------------------------
    overlap = te_ents & tr_ents
    print(f"\n  INDUCTIVE PREMISE")
    print(f"    test entities            {len(te_ents):>6,}")
    print(f"    of which seen in train   {len(overlap):>6,}  "
          f"({len(overlap)/max(1,len(te_ents)):.1%})")
    print(f"    {'✅ fully inductive' if not overlap else '⚠️ NOT fully disjoint — this is CATS-as-shipped; report it'}")

    # ---- ranking candidates -------------------------------------------------
    print(f"\n  CANDIDATE SETS (CATS's own, {ns.n_way}-way)")
    for direction, fname in (("tail", "ranking_tail.txt"), ("head", "ranking_head.txt")):
        rows, rep = convert_ranking(src / fname, test, direction, ns.n_way)
        if rep.get("status") == "missing":
            print(f"    {direction:5s} {fname} absent — chapter3.candidates will sample instead")
            continue
        dest = out / f"candidates_{direction}_{ns.n_way}way_s{ns.seed}.json"
        dest.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"    {direction:5s} {rep.get('ok',0):>5,}/{rep.get('n_queries',0):,} queries  "
              f"fingerprint {fingerprint(rows)}  -> {dest.name}")
        for bad, label in (("block_mismatch", "block did not match its query"),
                           ("gold_missing", "gold not among the candidates")):
            if rep.get(bad):
                print(f"          ⚠️ {rep[bad]:,} skipped: {label}")

    # ---- statistical power, stated up front --------------------------------
    n = len(test)
    mde = 2.8 * (0.30 / max(1, n) ** 0.5)      # rough: SD(RR) ~ 0.30
    print(f"\n  STATISTICAL POWER")
    print(f"    {n:,} queries -> smallest detectable PAIRED difference ~{mde:.3f} MRR")
    if n < 300:
        print(f"    ⚠️ small. Contrasts under ~{mde:.2f} MRR will not resolve.")
        print(f"       NELL-995-subset-inductive has 476 queries if power matters more")
        print(f"       than the WordNet hierarchy that S5 needs.")

    print(f"\n  next:")
    print(f"    python -m chapter3.validate --dataset {out.name}")
    print(f"    python -m chapter3.profile_specificity --dataset {out.name} --root data")


if __name__ == "__main__":
    main()
