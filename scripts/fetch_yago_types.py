"""
Build `data/YAGO3-10/entity2type.txt` — the one type source that is NOT free.

    python -m scripts.fetch_yago_types --dataset YAGO3-10

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
WN11 and NELL-995 carry their type inside the identifier, so the type comes for
free and is exogenous by construction. YAGO3-10 does not: its identifiers are
Wikipedia titles (`Steve_Jobs`, `F.C._Alverca`). Any type must be fetched from
outside the archive.

The tempting shortcut — read the type off the relation's domain/range, so that
anything appearing as head of `wasBornIn` becomes a Person — puts us straight
back into the endogenous trap. It derives the tag from the edges under test,
which is the whole defect this pipeline is trying to escape. So we do not do it.

ROUTE: Wikidata P31 ("instance of") via the Wikipedia title.
  ~123k entities, batched. Expect 15-25 minutes and a partial hit rate; a title
  that no longer resolves gets OTHER, which `coverage()` will report honestly.

If the endpoint is unreachable, the correct move is to run YAGO3-10 with induced
types and SAY SO in the caption — not to invent a type source.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.wikidata.org/w/api.php"
UA = "kgc-adaptation-thesis/1.0 (academic; entity typing for a thesis chapter)"


def _get(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def batch_types(titles: list[str]) -> dict[str, str]:
    """Wikipedia title -> the QID of its first P31 ('instance of') value.

    ★ BUG FIXED. `props` used to be "claims|labels", but the code then read
      `ent["sitelinks"]["enwiki"]["title"]` to key the result. wbgetentities
      only returns sitelinks when they are ASKED for, so that lookup was always
      None, every entity was skipped, and the script would have written an
      empty entity2type.txt while reporting 0.0% coverage — a silent no-op that
      looks like "Wikidata just doesn't have these entities".
    """
    q = urllib.parse.urlencode({
        "action": "wbgetentities", "sites": "enwiki",
        "titles": "|".join(titles), "props": "claims|sitelinks",
        "sitefilter": "enwiki", "format": "json"})
    data = _get(f"{API}?{q}")
    out: dict[str, str] = {}
    for ent in (data.get("entities") or {}).values():
        title = ((ent.get("sitelinks") or {}).get("enwiki") or {}).get("title")
        claims = (ent.get("claims") or {}).get("P31") or []
        if not (title and claims):
            continue
        qid = (claims[0].get("mainsnak", {}).get("datavalue", {})
               .get("value", {}).get("id"))
        if qid:
            out[title] = qid                      # resolved to a label below
    return out


def label_of(qids: list[str]) -> dict[str, str]:
    q = urllib.parse.urlencode({
        "action": "wbgetentities", "ids": "|".join(qids),
        "props": "labels", "languages": "en", "format": "json"})
    data = _get(f"{API}?{q}")
    return {k: (v.get("labels", {}).get("en", {}).get("value") or k)
            for k, v in (data.get("entities") or {}).items()}


def _from_yago(ns) -> None:
    """
    Stream YAGO's own type file and keep only this dataset's entities.

    ★ WHY THIS BEATS THE WIKIDATA ROUTE.
      YAGO joins Wikipedia (instances) to WordNet (classes), so every YAGO
      entity already has a WordNet class. YAGO3-10 is a SUBSET of YAGO3, so its
      entities are typed by construction -- coverage should be near total,
      where Wikidata depends on each Wikipedia title still resolving. The label
      is also the same kind of object WN11 gets, which makes the two graphs
      directly comparable.

      The file is large, so it is streamed line by line and never held in
      memory. Lines look like:

          <Alastair_Sim>  rdf:type  <wordnet_actor_109765278>
    """
    import gzip
    from collections import Counter, defaultdict

    from src.routing.semantic_types import parse_yago_line

    src = Path(ns.from_yago)
    if not src.exists():
        raise SystemExit(f"{src} not found. Download yagoSimpleTypes.tsv from "
                         f"the YAGO3 downloads page.")
    ent_file = Path(ns.root, ns.dataset, "entity2text.txt")
    if not ent_file.exists():
        raise SystemExit(f"{ent_file} not found — fetch the dataset first")

    wanted = {ln.split("\t")[0] for ln in
              ent_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
    print(f"{len(wanted):,} entities wanted · streaming {src}")

    # ★ COLLECT ALL classes per entity, then choose. An entity commonly has
    #   several (`actor`, `person`, `1979_films`); taking the first line seen
    #   makes the label depend on file order, which is not reproducible across
    #   dumps. Held in memory: ~123k entities x a few classes is tens of MB.
    cand: dict[str, set[tuple[str, str]]] = defaultdict(set)
    seen = matched = 0
    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            seen += 1
            row = parse_yago_line(line)
            if row and row[0] in wanted:
                cand[row[0]].add(row[1])
                matched += 1
            if seen % 5_000_000 == 0:
                print(f"  {seen:,} lines · {len(cand):,}/{len(wanted):,} entities hit")
    print(f"  {seen:,} lines read · {matched:,} usable rows")

    # ── choosing ONE class per entity ───────────────────────────────────────
    #   1. tier      WordNet > schema.org > wikicat heuristic
    #   2. frequency the MORE COMMON class wins, so `person` beats `actor`.
    #      That is deliberate: it keeps the tag vocabulary small and comparable
    #      to WN11's 23 supersenses, and a coarse category (person / place /
    #      film) is the granularity CATS and Knit mean by "type constraint".
    #      Flip the sign of `-freq` if you want the most SPECIFIC class instead
    #      — you will get thousands of sparse tags.
    #   3. alphabetical, purely to make ties reproducible across dumps.
    TIER = {"wordnet": 0, "schema": 1, "wikicat": 2}
    freq = Counter(c for s in cand.values() for _, c in s)
    found: dict[str, str] = {}
    tiers: Counter = Counter()
    for e, s in cand.items():
        kind, cls = min(s, key=lambda kc: (TIER.get(kc[0], 9), -freq[kc[1]], kc[1]))
        found[e] = cls
        tiers[kind] += 1

    out = Path(ns.root, ns.dataset, "entity2type.txt")
    out.write_text("".join(f"{e}\t{t}\n" for e, t in sorted(found.items())),
                   encoding="utf-8")
    cov = len(found) / max(1, len(wanted))
    print(f"\n-> {out}   {len(found):,}/{len(wanted):,} typed ({cov:.1%})")
    print(f"   source of the label: " + " · ".join(
        f"{k} {v:,} ({v / max(1, len(found)):.0%})" for k, v in tiers.most_common()))
    print(f"   {len(set(found.values())):,} distinct types · most common: " +
          ", ".join(f"{c}({n:,})" for c, n in Counter(found.values()).most_common(8)))
    if tiers.get("wikicat"):
        print("   ⚠️ wikicat labels are a HEAD-NOUN heuristic over Wikipedia "
              "category names — still exogenous, but noisier than WordNet. "
              "Report the split above.")
    if cov < ns.min_coverage:
        out.unlink(missing_ok=True)
        raise SystemExit(
            f"✋ coverage {cov:.1%} < {ns.min_coverage:.0%}. entity2type.txt "
            f"deleted.\n   Check the identifiers match: YAGO3-10 uses "
            f"`Alastair_Sim`, the dump uses `<Alastair_Sim>`.")
    print(f"✓ coverage {cov:.1%} — conditions C and G may use semantic types")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--root", default="data")
    ap.add_argument("--batch", type=int, default=50, help="wbgetentities caps at 50")
    ap.add_argument("--limit", type=int, default=0, help="0 = every entity")
    ap.add_argument("--min-coverage", type=float, default=0.60,
                    help="refuse to write the file below this hit rate")
    ap.add_argument("--from-yago", default=None, metavar="yagoSimpleTypes.tsv",
                    help="★ PREFERRED. Read YAGO's own WordNet classes instead "
                         "of querying Wikidata. Near-total coverage, because "
                         "YAGO3-10's entities are YAGO entities by construction.")
    ns = ap.parse_args()

    if ns.from_yago:
        _from_yago(ns)
        return

    src = Path(ns.root, ns.dataset, "entity2text.txt")
    if not src.exists():
        raise SystemExit(f"{src} not found — fetch the dataset first")

    ids = [ln.split("\t")[0] for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if ns.limit:
        ids = ids[: ns.limit]
    print(f"{len(ids):,} entities -> Wikidata P31, batches of {ns.batch}")

    qid_by_title: dict[str, str] = {}
    for i in range(0, len(ids), ns.batch):
        chunk = [e.replace("_", " ") for e in ids[i: i + ns.batch]]
        try:
            qid_by_title |= batch_types(chunk)
        except Exception as exc:                              # noqa: BLE001
            print(f"  batch {i // ns.batch}: {type(exc).__name__} — skipped ({exc})")
        if i and (i // ns.batch) % 20 == 0:
            print(f"  {i:,}/{len(ids):,}  hits={len(qid_by_title):,}")
        time.sleep(0.1)                                        # be polite

    qids = sorted({v for v in qid_by_title.values()})
    labels: dict[str, str] = {}
    for i in range(0, len(qids), 50):
        try:
            labels |= label_of(qids[i: i + 50])
        except Exception:                                      # noqa: BLE001
            pass

    out = Path(ns.root, ns.dataset, "entity2type.txt")
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for e in ids:
            qid = qid_by_title.get(e.replace("_", " "))
            if qid:
                f.write(f"{e}\t{labels.get(qid, qid)}\n")
                n += 1
    cov = n / max(1, len(ids))
    print(f"\n-> {out}   {n:,}/{len(ids):,} typed ({cov:.1%})")
    # ★ HARD GATE, not a warning. A partially-typed graph makes conditions C
    #   and G measure "does the model do better on the entities Wikidata
    #   happens to know about", which is not the question. Exit non-zero so a
    #   notebook cell cannot sail past it.
    if cov < ns.min_coverage:
        out.unlink(missing_ok=True)
        raise SystemExit(
            f"\n✋ coverage {cov:.1%} < --min-coverage {ns.min_coverage:.0%}. "
            f"entity2type.txt deleted.\n"
            f"   A typed condition built on this would be near-vacuous.\n"
            f"   Either raise coverage (check the titles resolve on enwiki), or\n"
            f"   run YAGO3-10 with INDUCED types and say so in the caption --\n"
            f"   but then do NOT pass --require-semantic.")
    print(f"✓ coverage {cov:.1%} — conditions C and G may use semantic types")


if __name__ == "__main__":
    main()
