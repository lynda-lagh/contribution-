"""
★ PROFILE A DATASET BEFORE YOU DESIGN AN EXPERIMENT ON IT.

    python -m chapter1.profile_data --dataset WN11
    python -m chapter1.profile_data --dataset WN11 FB13 --compare

WHY THIS EXISTS
---------------
Two bugs reached training because nobody looked at the data first:

  1. `entity_types(method="auto")` reads a POS marker out of the identifier
     (`stool_NN_2`). WN11 ids are `__east_indian_1` -- no marker. 100% OTHER,
     so condition C rendered prompts identical to condition B.
  2. YAGO3-10's test.tsv has no ±1 label column, so every gold answer became
     "No" and accuracy measured the class balance.

Both are visible in five seconds of profiling and invisible for hours of
training. This module answers, per dataset:

    what do the identifiers look like?      -> can we extract types?
    what do the descriptions look like?     -> is there text to enrich?
    are there test labels?                  -> is classification even possible?
    how concentrated are the types?         -> does a type tag carry information?
    how ambiguous are the surface forms?    -> is there anything to disambiguate?
    what does N training triples cover?     -> how big is the seen/unseen split?

and ends with a VERDICT: which conditions are viable here, and which are not.
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from collections import Counter, defaultdict

from src.data.loaders import KG, anonymise, load_kg
from src.routing.types import entity_types


# =============================================================================
def id_shape(kg: KG, n: int = 6) -> dict:
    """
    What do the identifiers look like, and do they carry a type?

    This is the check that would have caught the WN11 bug.
    """
    ids = list(kg.ent2txt)
    sample = ids[:n]
    pos_marker = sum(1 for e in ids if re.search(r"_(NN|VB|JJ|RB)_", e))
    freebase = sum(1 for e in ids if e.startswith("/m/") or e.startswith("/g/"))
    numeric = sum(1 for e in ids if e.strip("_").isdigit())
    wordnetish = sum(1 for e in ids if re.match(r"^__.+_\d+$", e))
    n_ids = len(ids) or 1
    return {
        "n_entities": len(ids),
        "sample": sample,
        "pos_marker_rate": pos_marker / n_ids,
        "freebase_mid_rate": freebase / n_ids,
        "numeric_rate": numeric / n_ids,
        "wordnet_style_rate": wordnetish / n_ids,
        "carries_type_in_id": pos_marker / n_ids > 0.5,
        "verdict": (
            "ids carry a POS marker -> `pos` type extraction will work"
            if pos_marker / n_ids > 0.5 else
            "⚠️ ids carry NO type marker -> `pos` returns 100% OTHER. "
            "Induced types (relation positions) are the fallback."),
    }


def description_quality(kg: KG) -> dict:
    """Is there text worth enriching, or are labels bare identifiers?"""
    lens, bare = [], 0
    for e, txt in kg.ent2txt.items():
        t = (txt or "").strip()
        lens.append(len(t.split()))
        if not t or t.replace("_", " ").strip() == e.replace("_", " ").strip():
            bare += 1
    n = len(lens) or 1
    return {
        "median_words": st.median(lens) if lens else 0,
        "mean_words": sum(lens) / n,
        "max_words": max(lens) if lens else 0,
        "bare_label_rate": bare / n,
        "has_real_descriptions": (st.median(lens) if lens else 0) >= 4,
        "verdict": (
            "descriptions are real sentences -> entity-description enrichment is meaningful"
            if (st.median(lens) if lens else 0) >= 4 else
            "⚠️ labels are short names, not descriptions -> 'entity description' "
            "enrichment has almost nothing to work with (cf. ColKGC: rewriting "
            "existing descriptions gives ~0 gain)"),
    }


def relation_stats(kg: KG) -> dict:
    c = Counter(t.relation for t in kg.train)
    v = sorted(c.values(), reverse=True)
    total = sum(v) or 1
    return {
        "n_relations": len(c),
        "max_freq": v[0] if v else 0,
        "min_freq": v[-1] if v else 0,
        "edge_imbalance_ratio": (v[0] / v[-1]) if v and v[-1] else None,
        "top1_share": v[0] / total if v else 0,
        "top3_share": sum(v[:3]) / total if v else 0,
        "verdict": (
            f"⚠️ heavily imbalanced (EIR {v[0]/v[-1]:.1f}, top relation = "
            f"{v[0]/total:.1%} of triples) -> report PER-RELATION accuracy, "
            f"a headline mean can be one relation solved"
            if v and v[-1] and v[0] / v[-1] > 10 else
            "reasonably balanced across relations"),
    }


def test_labels(kg: KG) -> dict:
    """★ Can this dataset support triple classification at all?"""
    labs = Counter(t.label for t in kg.test)
    n = len(kg.test) or 1
    pos, neg, none = labs.get(1, 0), labs.get(-1, 0), labs.get(None, 0)
    ok = pos > 0 and neg > 0
    return {
        "n_test": len(kg.test), "positive": pos, "negative": neg, "unlabelled": none,
        "balance": pos / (pos + neg) if (pos + neg) else None,
        "supports_classification": ok,
        "supports_ranking": True,          # ranking never needs ±1 labels
        "verdict": (
            f"±1 labels present ({pos:,} pos / {neg:,} neg) -> triple "
            f"classification AND ranking both work"
            if ok else
            f"⚠️ NO ±1 labels ({none:,} unlabelled) -> triple classification is "
            f"IMPOSSIBLE (every gold answer becomes 'No'). Ranking still works: "
            f"link prediction needs only the true tail."),
    }


def type_options(kg: KG) -> dict:
    """
    Try every extraction method and report not just coverage but CONCENTRATION.

    ⚠️ Coverage is not enough. 22 distinct types sounds healthy, but if one type
    covers 27% of entities the tag carries far less information than the count
    suggests. Entropy is the honest measure.
    """
    import math
    out = {}
    for m in ("pos", "induced"):
        try:
            t = entity_types(kg, method=m)
        except Exception as e:
            out[m] = {"error": f"{type(e).__name__}: {e}"}
            continue
        vals = [v for v in t.values() if v not in (None, "OTHER")]
        n = len(t) or 1
        c = Counter(vals)
        tot = sum(c.values()) or 1
        H = -sum((k / tot) * math.log2(k / tot) for k in c.values()) if c else 0.0
        out[m] = {
            "distinct": len(c),
            "other_rate": 1 - len(vals) / n,
            "top_type": c.most_common(1)[0] if c else None,
            "top_share": (c.most_common(1)[0][1] / tot) if c else 0,
            "entropy_bits": H,
            "max_entropy_bits": math.log2(len(c)) if c else 0,
            "usable": (1 - len(vals) / n) <= 0.5 and len(c) >= 2,
        }
    best = next((m for m in ("pos", "induced")
                 if out.get(m, {}).get("usable")), None)
    out["recommended"] = best
    out["verdict"] = (
        f"use method='{best}'" if best else
        "⚠️ NO usable type source -> type conditions (C, D, E, G) cannot differ "
        "from their baselines on this dataset")
    return out


def ambiguity(kg: KG) -> dict:
    """Do different entities share a surface form? (MKGL: 14 entities named 'call')"""
    by = defaultdict(list)
    for e, txt in kg.ent2txt.items():
        key = (txt or "").split(",")[0].strip().lower()
        if key:
            by[key].append(e)
    dup = {k: v for k, v in by.items() if len(v) > 1}
    n = len(kg.ent2txt) or 1
    worst = max(dup.items(), key=lambda kv: len(kv[1])) if dup else None
    return {
        "ambiguous_surface_forms": len(dup),
        "entities_affected": sum(len(v) for v in dup.values()),
        "rate": sum(len(v) for v in dup.values()) / n,
        "worst": {"name": worst[0], "n": len(worst[1])} if worst else None,
        "verdict": ("ambiguity is negligible -> nothing for a disambiguation "
                    "feature to route on"
                    if sum(len(v) for v in dup.values()) / n < 0.02 else
                    "meaningful ambiguity -> disambiguation features have signal"),
    }


def degree(kg: KG) -> dict:
    d = Counter()
    for t in kg.train:
        d[t.head] += 1
        d[t.tail] += 1
    v = sorted(d.values())
    if not v:
        return {}
    return {"median": st.median(v), "mean": sum(v) / len(v),
            "p90": v[int(0.9 * len(v)) - 1], "max": v[-1],
            "entities_with_degree_1": sum(1 for x in v if x == 1) / len(v)}


def coverage_at(kg: KG, sizes=(1000, 10_000, 25_000, 50_000), seed: int = 42) -> dict:
    """
    ★ How much of the entity set does N training triples touch?

    This determines the seen/unseen split -- the free second instrument. If
    coverage is ~100% there is no unseen bucket and that instrument is unavailable.
    """
    from src.data.sampling import sample_triples
    out = {}
    n_ent = len(kg.ent2txt) or 1
    for n in sizes:
        if n > len(kg.train):
            continue
        pos = sample_triples(kg.train, n, seed=seed, stratified=True, min_per_relation=10)
        seen = {e for t in pos for e in (t.head, t.tail)}
        test_both = sum(1 for t in kg.test if t.head in seen and t.tail in seen)
        out[n] = {"entities_seen": len(seen), "coverage": len(seen) / n_ent,
                  "test_both_seen": test_both / max(1, len(kg.test))}
    return out


# =============================================================================
def profile(dataset: str, root: str = "data") -> dict:
    kg = load_kg(dataset, root)
    rep = {
        "dataset": dataset,
        "ids": id_shape(kg),
        "descriptions": description_quality(kg),
        "relations": relation_stats(kg),
        "test": test_labels(kg),
        "types": type_options(kg),
        "types_anon": type_options(anonymise(kg)),
        "ambiguity": ambiguity(kg),
        "degree": degree(kg),
        "coverage": coverage_at(kg),
    }
    return rep


def verdict(rep: dict) -> list[str]:
    """★ Which conditions are viable on this dataset — the actionable output."""
    out = []
    t = rep["test"]
    ty, tya = rep["types"], rep["types_anon"]

    out.append(f"A / B (names on/off)      "
               f"{'✅' if t['supports_classification'] else '❌ no ±1 test labels'}")

    if ty.get("recommended") and tya.get("recommended"):
        m, ma = ty["recommended"], tya["recommended"]
        share = ty[m]["top_share"]
        flag = "✅" if share < 0.35 else "⚠️ "
        out.append(f"C / D / E / G (types)     {flag} via method='{m}' "
                   f"({ty[m]['distinct']} types, top covers {share:.0%}, "
                   f"H={ty[m]['entropy_bits']:.1f} bits)")
        if m != ma:
            out.append(f"   ⚠️ anonymised graph needs '{ma}' — types must survive "
                       f"anonymisation or B→C is not a controlled step")
        if share >= 0.35:
            out.append(f"   ⚠️ one type covers {share:.0%} of entities — the tag "
                       f"carries less than {ty[m]['distinct']} types suggests")
    else:
        out.append("C / D / E / G (types)     ❌ no usable type source")

    cov = rep["coverage"].get(10_000)
    if cov:
        b = cov["test_both_seen"]
        out.append(f"seen/unseen instrument    "
                   f"{'✅' if 0.05 < b < 0.95 else '⚠️ '} at 10k triples, "
                   f"{b:.0%} of test triples have both entities seen")
    out.append(f"ranking / link prediction ✅ always available (needs no ±1 labels)")
    return out


def show(rep: dict) -> None:
    d = rep["dataset"]
    print(f"\n{'=' * 78}\n  {d}\n{'=' * 78}")

    i = rep["ids"]
    print(f"\n IDENTIFIERS   {i['n_entities']:,} entities")
    for s in i["sample"][:4]:
        print(f"    {s}")
    print(f"    POS marker {i['pos_marker_rate']:.1%} · freebase-mid "
          f"{i['freebase_mid_rate']:.1%} · wordnet-style {i['wordnet_style_rate']:.1%}")
    print(f"    → {i['verdict']}")

    de = rep["descriptions"]
    print(f"\n DESCRIPTIONS  median {de['median_words']:.0f} words · "
          f"max {de['max_words']} · bare labels {de['bare_label_rate']:.1%}")
    print(f"    → {de['verdict']}")

    r = rep["relations"]
    eir = f"{r['edge_imbalance_ratio']:.1f}" if r["edge_imbalance_ratio"] else "—"
    print(f"\n RELATIONS     {r['n_relations']} · EIR {eir} · "
          f"top1 {r['top1_share']:.1%} · top3 {r['top3_share']:.1%}")
    print(f"    → {r['verdict']}")

    t = rep["test"]
    print(f"\n TEST SET      {t['n_test']:,} · +{t['positive']:,} / "
          f"−{t['negative']:,} / unlabelled {t['unlabelled']:,}")
    print(f"    → {t['verdict']}")

    print("\n TYPE SOURCES")
    for g, key in (("raw", "types"), ("anon", "types_anon")):
        for m in ("pos", "induced"):
            v = rep[key].get(m, {})
            if "error" in v:
                print(f"    {g:5s} {m:8s} error: {v['error'][:44]}")
            else:
                print(f"    {g:5s} {m:8s} {v['distinct']:3d} types · "
                      f"OTHER {v['other_rate']:5.1%} · top {v['top_share']:5.1%} · "
                      f"H {v['entropy_bits']:.2f} bits "
                      f"{'✅' if v['usable'] else '❌'}")
    print(f"    → {rep['types']['verdict']}")

    a = rep["ambiguity"]
    w = f" (worst: '{a['worst']['name']}' ×{a['worst']['n']})" if a["worst"] else ""
    print(f"\n AMBIGUITY     {a['rate']:.2%} of entities share a surface form{w}")
    print(f"    → {a['verdict']}")

    g = rep["degree"]
    if g:
        print(f"\n DEGREE        median {g['median']:.0f} · p90 {g['p90']} · "
              f"max {g['max']} · degree-1 {g['entities_with_degree_1']:.1%}")

    print("\n COVERAGE BY TRAINING SIZE")
    print(f"    {'triples':>9s} {'entities seen':>14s} {'coverage':>9s} {'test both-seen':>15s}")
    for n, v in rep["coverage"].items():
        print(f"    {n:>9,d} {v['entities_seen']:>14,d} {v['coverage']:>9.1%} "
              f"{v['test_both_seen']:>15.1%}")

    print(f"\n{'─' * 78}\n VERDICT — which Chapter 1 conditions are viable here\n{'─' * 78}")
    for line in verdict(rep):
        print(f"  {line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["WN11"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--json", default=None, help="also write the raw report here")
    ns = ap.parse_args()

    reps = []
    for d in ns.dataset:
        try:
            rep = profile(d, ns.root)
        except FileNotFoundError as e:
            print(f"\n[{d}] not available: {e}")
            continue
        show(rep)
        reps.append(rep)

    if len(reps) > 1:
        print(f"\n{'=' * 78}\n  SIDE BY SIDE\n{'=' * 78}")
        print(f"  {'':22s}" + "".join(f"{r['dataset']:>14s}" for r in reps))
        rows = [
            ("entities", lambda r: f"{r['ids']['n_entities']:,}"),
            ("relations", lambda r: str(r["relations"]["n_relations"])),
            ("EIR", lambda r: f"{r['relations']['edge_imbalance_ratio']:.1f}"
                if r["relations"]["edge_imbalance_ratio"] else "—"),
            ("±1 test labels", lambda r: "yes" if r["test"]["supports_classification"] else "NO"),
            ("desc. median words", lambda r: f"{r['descriptions']['median_words']:.0f}"),
            ("type method", lambda r: str(r["types"]["recommended"])),
            ("distinct types", lambda r: str(r["types"].get(
                r["types"]["recommended"], {}).get("distinct", "—"))),
            ("top type share", lambda r: f"{r['types'].get(r['types']['recommended'], {}).get('top_share', 0):.0%}"),
            ("ambiguity", lambda r: f"{r['ambiguity']['rate']:.2%}"),
        ]
        for label, fn in rows:
            print(f"  {label:22s}" + "".join(f"{fn(r):>14s}" for r in reps))

    if ns.json and reps:
        import json
        from pathlib import Path
        Path(ns.json).write_text(json.dumps(reps, indent=2, default=str), encoding="utf-8")
        print(f"\nwritten -> {ns.json}")


if __name__ == "__main__":
    main()
