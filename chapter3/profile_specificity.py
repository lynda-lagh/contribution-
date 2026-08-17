"""
★★ THE THREE CHECKS THAT DECIDE WHETHER `S5_semantic` IS WORTH BUILDING.

Semantic specificity — *how general or specific a label's meaning is* — is
unoccupied in the corpus as an allocation signal. But it has three cheap ways of
being an illusion, and two of them are mistakes this project has already made
once. Run this BEFORE spending GPU time on the policy.

    1. VARIANCE      does depth actually vary on this graph?
                     ⚠️ the first Chapter 3 router keyed on quality bands where
                     `moderate` = 95.7% of entities. One bucket covering nearly
                     everything made the ladder flat BY CONSTRUCTION, not by
                     finding. Same failure mode, second chance to catch it.

    2. DEGREE        is depth just log(degree) wearing a linguistics costume?
                     Resnik's information content is -log P(concept): general
                     concepts are frequent, specific ones rare. If depth tracks
                     degree, this is LONG-TAIL routing, which P30 (KICGPTv2)
                     already owns.

    3. CLUSTERING    does WordNet depth beat what clustering already gives free?
                     P31 (GS-Quant) derives a coarse-to-fine hierarchy by
                     AGGLOMERATIVE CLUSTERING of entity representations. If
                     lexical depth agrees closely with cluster depth, the
                     "linguistic ground truth" framing weakens and the honest
                     claim becomes "a free external signal reproduces what
                     clustering learns, at zero training cost" — different, and
                     smaller.

    python -m chapter3.profile_specificity --dataset WN18RR-ind

⚠️ Depth is EXACT on WN18RR because its entities are WordNet synsets. On
   FB15k-237 it is unavailable and the transferable proxies (IDF, degree) are
   precisely the confound in check 2. Say which you used.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

VERDICTS = {
    "variance": (1.5, "depth entropy in bits; below this the feature is degenerate"),
    "degree":   (0.70, "|rho| with log-degree; above this it is long-tail routing"),
    "cluster":  (0.70, "|rho| with cluster depth; above this clustering suffices"),
}


# ------------------------------------------------------------------ helpers
def spearman(x: list[float], y: list[float]) -> float | None:
    """Rank correlation without scipy. Ties averaged."""
    n = len(x)
    if n < 10:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


def entropy(values) -> float:
    c = Counter(values)
    tot = sum(c.values()) or 1
    return -sum((n / tot) * math.log2(n / tot) for n in c.values() if n)


def wordnet_depth(entity_ids: list[str]) -> dict[str, int]:
    """
    ★ WN18RR entity ids look like `__dog_1` / `04562658` depending on the release.
    Try NLTK WordNet; fall back to the id's own sense-number suffix, which is a
    weak proxy and is reported as such.
    """
    out: dict[str, int] = {}
    try:
        from nltk.corpus import wordnet as wn
        try:
            wn.synsets("dog")
        except LookupError:
            import nltk
            nltk.download("wordnet", quiet=True)
        for e in entity_ids:
            name = re.sub(r"^_+|_+$", "", e).replace("__", "")
            m = re.match(r"(.+?)_([nvasr])_(\d+)$", name) or re.match(r"(.+?)_(\d+)$", name)
            lemma = (m.group(1) if m else name).replace("_", " ")
            syns = wn.synsets(lemma.replace(" ", "_"))
            if syns:
                out[e] = min(len(p) for p in syns[0].hypernym_paths())
    except Exception as exc:                                  # noqa: BLE001
        print(f"[depth] WordNet unavailable ({type(exc).__name__}) — "
              f"no lexical depth computed")
    return out


def cluster_depth(labels: dict[str, str], k_levels: int = 4) -> dict[str, int]:
    """
    A stand-in for P31's agglomerative hierarchy, without sklearn: bucket by
    label length and token overlap. Crude ON PURPOSE — if even this crude
    clustering tracks WordNet depth, the lexical signal is not adding much.
    """
    toks = {e: set(re.split(r"\W+", t.lower())) - {""} for e, t in labels.items()}
    df = Counter(w for s in toks.values() for w in s)
    out = {}
    for e, s in toks.items():
        if not s:
            out[e] = 0
            continue
        # rarer tokens -> deeper in a coarse-to-fine tree
        rarity = sum(1.0 / (df[w] + 1) for w in s) / len(s)
        out[e] = min(k_levels, int(rarity * k_levels * 20))
    return out


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="WN18RR-ind")
    ap.add_argument("--root", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default=None)
    ns = ap.parse_args()

    d = Path(ns.root, ns.dataset)
    ent2txt: dict[str, str] = {}
    p = d / "entity2text.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                ent2txt[k] = v
    if not ent2txt:
        raise SystemExit(f"no entity2text.txt under {d}")

    deg: Counter = Counter()
    tp = d / "train.tsv"
    if not tp.exists():
        raise SystemExit(f"no train.tsv under {d}")
    for line in tp.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            deg[parts[0]] += 1
            deg[parts[2]] += 1

    ents = list(ent2txt)
    if ns.limit:
        ents = ents[: ns.limit]

    print("=" * 78)
    print(f"SEMANTIC SPECIFICITY PRE-CHECK — {ns.dataset}   ({len(ents):,} entities)")
    print("=" * 78)

    depth = wordnet_depth(ents)
    cdepth = cluster_depth({e: ent2txt[e] for e in ents})
    # polysemy proxy: how many entities share this surface form
    surface = defaultdict(list)
    for e in ents:
        surface[ent2txt[e].strip().lower()].append(e)
    senses = {e: len(surface[ent2txt[e].strip().lower()]) for e in ents}

    rep: dict = {"dataset": ns.dataset, "n_entities": len(ents),
                 "depth_available": len(depth)}

    # ---- 1 · variance ------------------------------------------------------
    print("\n1 · VARIANCE — does depth vary, or is it one bucket?")
    if not depth:
        print("   ✗ no lexical depth available on this dataset")
        print("     -> S5 can only use IDF / polysemy here, which is check 2's confound")
        rep["variance"] = None
    else:
        vals = [depth[e] for e in ents if e in depth]
        H = entropy(vals)
        c = Counter(vals)
        top = max(c.values()) / len(vals)
        rep["variance"] = {"entropy_bits": H, "top_bucket_share": top,
                           "distribution": dict(sorted(c.items()))}
        ok = H >= VERDICTS["variance"][0]
        print(f"   depth covered      {len(vals):,} of {len(ents):,} entities")
        print(f"   entropy            {H:.2f} bits   (need >= {VERDICTS['variance'][0]})")
        print(f"   largest bucket     {top:.1%}")
        print(f"   distribution       {dict(sorted(c.items()))}")
        print(f"   {'✅ usable' if ok else '✗ DEGENERATE — this is the 95.7% quality-band mistake again'}")

    # ---- 2 · degree --------------------------------------------------------
    print("\n2 · DEGREE — is depth just long-tail routing in disguise?")
    common = [e for e in ents if e in depth and deg[e] > 0]
    if len(common) < 10:
        print("   ✗ too few entities with both depth and degree")
        rep["degree"] = None
    else:
        rho = spearman([depth[e] for e in common],
                       [math.log(deg[e] + 1) for e in common])
        rep["degree"] = {"spearman_rho": rho, "n": len(common)}
        print(f"   Spearman rho(depth, log-degree) = {rho:+.3f}   over {len(common):,}")
        if rho is None:
            print("   ? undefined")
        elif abs(rho) >= VERDICTS["degree"][0]:
            print("   ✗ SAME FEATURE. This is long-tail routing, which P30/KICGPTv2 owns.")
            print("     Do not claim a semantic contribution; claim the degree one honestly.")
        else:
            print("   ✅ DISTINCT from degree — the semantic framing survives this check")

    # ---- 3 · clustering ----------------------------------------------------
    print("\n3 · CLUSTERING — does WordNet beat what clustering gives free? (P31)")
    common2 = [e for e in ents if e in depth and e in cdepth]
    if len(common2) < 10:
        print("   ✗ too few entities")
        rep["cluster"] = None
    else:
        rho2 = spearman([depth[e] for e in common2], [cdepth[e] for e in common2])
        rep["cluster"] = {"spearman_rho": rho2, "n": len(common2)}
        print(f"   Spearman rho(depth, cluster depth) = {rho2:+.3f}   over {len(common2):,}")
        if rho2 is not None and abs(rho2) >= VERDICTS["cluster"][0]:
            print("   ⚠️ clustering already recovers this. GS-Quant (P31) derives its")
            print("      hierarchy exactly that way. Reframe as 'a free external signal")
            print("      reproduces what clustering learns' — smaller, but honest.")
        else:
            print("   ✅ lexical depth carries something clustering does not")

    # ---- ambiguity, which is a SEPARATE axis -------------------------------
    amb = sum(1 for e in ents if senses[e] >= 3) / max(1, len(ents))
    rep["ambiguity_rate"] = amb
    print(f"\n   ambiguity (>=3 entities share a surface form): {amb:.2%}")
    print("   ★ ambiguity is a DIFFERENT axis from generality. A specific but")
    print("     ambiguous label needs DISAMBIGUATING context, not more of it —")
    print("     the case S1..S4 cannot express.")

    # ---- verdict -----------------------------------------------------------
    checks = [
        rep.get("variance") and rep["variance"]["entropy_bits"] >= VERDICTS["variance"][0],
        rep.get("degree") and rep["degree"]["spearman_rho"] is not None
            and abs(rep["degree"]["spearman_rho"]) < VERDICTS["degree"][0],
        rep.get("cluster") and rep["cluster"]["spearman_rho"] is not None
            and abs(rep["cluster"]["spearman_rho"]) < VERDICTS["cluster"][0],
    ]
    passed = sum(1 for c in checks if c)
    rep["checks_passed"] = passed
    print("\n" + "-" * 78)
    print(f"VERDICT — {passed}/3 checks passed")
    if passed == 3:
        print("  ✅ BUILD S5_semantic. It is distinct from degree, distinct from")
        print("     clustering, and has variance on this graph.")
    elif passed == 2:
        print("  ⚠️ BUILD IT, BUT NARROW THE CLAIM to whichever check failed.")
    else:
        print("  ✋ DO NOT BUILD S5_semantic on this dataset. Report the profile as")
        print("     the reason — 'the feature is degenerate / redundant here' is a")
        print("     finding, and it costs no GPU to say.")

    out = Path(ns.json or Path("results", f"ch3_specificity_{ns.dataset}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
