"""
★ EVALUATION for one (policy, budget) cell — link prediction in BOTH directions,
  plus relation prediction with a real confusion matrix.

    python -m chapter3.evaluate --dataset WN18RR-ind --policy S0_uniform --budget 120
    python -m chapter3.evaluate --dataset WN18RR-ind --policy S0_uniform --budget 120 \
        --direction head
    python -m chapter3.evaluate --dataset WN18RR-ind --policy S0_uniform --budget 120 \
        --task relation                      # ★ the confusion matrix lives here

HOW IT RANKS
------------
For `(h, r, ?)` we score every candidate by `P("Yes" | prompt)` and sort. That
yields an ORDERING, so MRR and Hits@K are computable — which generative top-1
decoding cannot give (see chapter1/rank.py's note).

**50-way, filtered**, following CATS and RealKGC:
> *"we rank each answer tail entity against 50 randomly sampled negative entities"*

⚠️ 50-way Hits@1 is NOT full-ranking Hits@1. Every caption must say "50-way".

★ THREE CORRECTIONS TO THE FIRST VERSION
----------------------------------------
1. **Candidates are read from a frozen file**, not sampled inline. Inline
   sampling was reproducible only while seed, query order AND query count all
   matched across cells; change `--limit` and two policies silently ranked
   against different negatives. See candidates.py.

2. **Both directions.** `(h, r, ?)` and `(?, r, t)`. CATS and RealKGC report
   both; a one-directional table looks like the easy side was chosen.

3. **The F1 is real now.** The old `top1_classification` incremented `fp` and
   `fn` on the SAME relation for every miss, so `fp == fn`, so precision ==
   recall == F1 == per-relation Hits@1. It was Hits@1 wearing a hat, and there
   was no confusion matrix anywhere because a two-outcome event has nothing to
   confuse. Genuine multi-class classification needs a task whose classes are
   confusable — hence `--task relation`: predict `(h, ?, t)` over the relation
   vocabulary, where predicting `_hypernym` when the answer is `_hyponym` is a
   real, informative error.

★ Per-query rows are written to the results file so `report.py` can run the
  PAIRED bootstrap (stats.py). Without them only unpaired tests are possible,
  and at n=300 those cannot detect anything smaller than ~0.05 MRR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .candidates import fingerprint, load as load_candidates, qid


# ------------------------------------------------------------------ metrics
def ranking_metrics(ranks: list[int]) -> dict:
    if not ranks:
        return {}
    n = len(ranks)
    return {
        "n": n,
        "MRR": sum(1.0 / r for r in ranks) / n,
        "MR": sum(ranks) / n,
        "hits@1": sum(r <= 1 for r in ranks) / n,
        "hits@3": sum(r <= 3 for r in ranks) / n,
        "hits@10": sum(r <= 10 for r in ranks) / n,
    }


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def per_relation_hits1(rows: list[dict]) -> dict:
    """
    Per-relation Hits@1 for the LINK task — named for what it actually is.

    ⚠️ Do not call this F1. On a retrieval task the top-1 event has two
       outcomes ("gold retrieved" / "not"), so any per-relation precision,
       recall and F1 computed from it are all algebraically equal to Hits@1.
       The old code reported exactly that under three different names.

    It is still worth reporting: a model that nails one huge relation and fails
    the rest looks fine on aggregate MRR and bad here. With relation frequency
    as skewed as it is, that is the failure mode to watch for.
    """
    per = defaultdict(lambda: {"hit": 0, "n": 0})
    for r in rows:
        per[r["relation"]]["n"] += 1
        per[r["relation"]]["hit"] += int(r["rank"] == 1)
    out = {k: {"n": v["n"], "hits@1": v["hit"] / v["n"]} for k, v in per.items()}
    macro = sum(v["hits@1"] for v in out.values()) / len(out) if out else 0.0
    micro = sum(v["hit"] for v in per.values()) / sum(v["n"] for v in per.values()) \
        if per else 0.0
    return {"per_relation": out, "macro_hits@1": macro, "micro_hits@1": micro,
            "n_relations": len(out),
            "note": "per-relation precision/recall/F1 are all identical to "
                    "hits@1 on a retrieval task; see --task relation for real F1"}


def confusion(rows: list[dict], labels: list[str]) -> dict:
    """
    ★★ THE REAL CLASSIFICATION VIEW — relation prediction, `(h, ?, t)`.

    Here a mistake names a WRONG CLASS, so `fp` lands on the predicted relation
    and `fn` on the gold one. They differ, precision != recall, and macro-F1 is
    a genuine number rather than Hits@1 renamed.

    ★ And the matrix is the interesting part for Chapter 3: it shows WHICH
      relations a policy's context helps with. `_hypernym` confused with
      `_hyponym` is the inverse-relation failure — a directional error that more
      tokens will not fix but the RIGHT tokens might. That is an argument that
      allocation is doing something semantic rather than just adding context.
    """
    idx = {r: i for i, r in enumerate(labels)}
    M = [[0] * len(labels) for _ in labels]
    for r in rows:
        g, p = r.get("gold_label"), r.get("pred_label")
        if g in idx and p in idx:
            M[idx[g]][idx[p]] += 1

    per, f1s = {}, []
    TP = FP = FN = 0
    for i, lab in enumerate(labels):
        tp = M[i][i]
        fn = sum(M[i]) - tp                              # gold i, predicted else
        fp = sum(M[j][i] for j in range(len(labels))) - tp   # predicted i, gold else
        p, rc, f = prf(tp, fp, fn)
        support = sum(M[i])
        per[lab] = {"support": support, "tp": tp, "fp": fp, "fn": fn,
                    "precision": p, "recall": rc, "f1": f}
        if support:                       # macro over OBSERVED classes only
            f1s.append(f)
        TP += tp; FP += fp; FN += fn
    mp, mr, mf = prf(TP, FP, FN)

    # the single most common confusion — usually the most quotable line
    worst = None
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and M[i][j] and (worst is None or M[i][j] > worst[2]):
                worst = (labels[i], labels[j], M[i][j])

    return {"labels": labels, "matrix": M, "per_class": per,
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "micro_f1": mf, "micro_precision": mp, "micro_recall": mr,
            "accuracy": (sum(M[i][i] for i in range(len(labels)))
                         / max(1, sum(map(sum, M)))),
            "n_classes_observed": len(f1s),
            "top_confusion": ({"gold": worst[0], "predicted": worst[1],
                               "n": worst[2]} if worst else None)}


def calibration(scores: list[float], correct: list[bool], n_bins: int = 10) -> dict:
    if not scores:
        return {}
    ece, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        m = [j for j, s in enumerate(scores) if (lo < s <= hi) or (i == 0 and s == 0)]
        if not m:
            continue
        acc = sum(correct[j] for j in m) / len(m)
        conf = sum(scores[j] for j in m) / len(m)
        ece += (len(m) / len(scores)) * abs(acc - conf)
        bins.append({"lo": lo, "hi": hi, "n": len(m), "acc": acc, "conf": conf})
    brier = sum((s - float(c)) ** 2 for s, c in zip(scores, correct)) / len(scores)
    return {"ECE": ece, "Brier": brier, "bins": bins}


def relation_candidates(kg, q: dict, cap: int, seed: int) -> list[str]:
    """
    Candidate relations for `(h, ?, t)`, frozen per query the same way entity
    candidates are: a per-query rng derived from the query id, so the set does
    not depend on how many queries precede it.

    Uses the FULL vocabulary when it is small enough (WN18RR has 11, which makes
    an 11x11 matrix fully printable). Above `cap` it samples, always including
    the gold — and the matrix is then over the sampled classes, which must be
    stated in the caption.
    """
    rels = sorted(kg.rel2txt)
    if len(rels) <= cap:
        return rels
    h = hashlib.sha256(f"{seed}|rel|{q['head']}|{q['tail']}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    pool = [r for r in rels if r != q["relation"]]
    rng.shuffle(pool)
    return sorted([q["relation"], *pool[: cap - 1]])


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="WN18RR-ind")
    ap.add_argument("--root", default="data")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--policy", default="S0_uniform")
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--adapter", default=None,
                    help="the SHARED model. OMIT for the ★ untuned baseline")
    ap.add_argument("--direction", default="tail", choices=("tail", "head"))
    ap.add_argument("--task", default="link", choices=("link", "relation"),
                    help="link = rank entities · relation = rank relations "
                         "(★ the real confusion matrix)")
    ap.add_argument("--n-way", type=int, default=50)
    ap.add_argument("--rel-cap", type=int, default=50,
                    help="max relation classes for --task relation")
    ap.add_argument("--limit", type=int, default=300,
                    help="queries. 300 x 50 = 15k forward passes ~ 12 min on a T4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None,
                    help="suffix for the results filename, e.g. 'untuned'")
    ns = ap.parse_args()

    import torch

    from src.data.loaders import load_kg
    from src.data.prompts import ALPACA_NO_INPUT
    from src.infer.scoring import yes_no_probabilities
    from src.utils.config import load_config
    from src.utils.progress import track
    from chapter1.evaluate import _load

    cfg = load_config(ns.config)
    kg = load_kg(ns.dataset, ns.root)

    cell = Path(ns.root, ns.dataset, "built",
                f"{ns.policy}_B{ns.budget}_{ns.direction}")
    legacy = Path(ns.root, ns.dataset, "built", f"{ns.policy}_B{ns.budget}")
    if not cell.exists() and legacy.exists() and ns.direction == "tail":
        cell = legacy                      # tolerate files built before --direction
    qp = cell / "queries.json"
    if not qp.exists():
        raise SystemExit(
            f"{qp} not built.\n  python -m chapter3.data --dataset {ns.dataset} "
            f"--policy {ns.policy} --budget {ns.budget} --direction {ns.direction}")
    queries = json.loads(qp.read_text(encoding="utf-8"))[: ns.limit]

    # ★ frozen candidates — identical for every policy BY CONSTRUCTION
    if ns.task == "link":
        cand_file = load_candidates(ns.root, ns.dataset, ns.direction,
                                    ns.n_way, ns.seed)
        print(f"[eval] candidates: {len(cand_file):,} frozen sets · "
              f"fingerprint {fingerprint(cand_file)}")
    else:
        cand_file = None

    is_untuned = ns.adapter is None
    print(f"[eval] {ns.dataset} · {ns.policy} · B={ns.budget} · {ns.direction} · "
          f"task={ns.task}")
    print(f"[eval] adapter: {ns.adapter or '★ NONE — untuned baseline'}")
    print(f"[eval] {len(queries)} queries")

    model, tok = _load(cfg["model"]["name"], ns.adapter)

    rows, gold_scores, gold_correct = [], [], []
    n_short = 0
    for q in track(queries, f"ranking ({ns.task}/{ns.direction})",
                   total=len(queries), unit="query"):
        k = qid(q["head"], q["relation"], q["tail"], ns.direction)

        if ns.task == "link":
            rec = cand_file.get(k)
            if rec is None:
                continue                    # query not in the frozen set; skip
            cands = rec["candidates"]
            gold = rec["gold"]
            surface = [kg.ent2txt.get(c, c) for c in cands]
            if len(cands) < ns.n_way:
                n_short += 1
        else:
            cands = relation_candidates(kg, q, ns.rel_cap, ns.seed)
            gold = q["relation"]
            surface = [kg.rel2txt.get(c, c) for c in cands]

        # prefix/suffix straddle the slot being predicted, so the SAME machinery
        # serves tail, head and relation prediction
        pre = q.get("prefix", "")
        suf = q.get("suffix", "?")
        if ns.task == "relation":
            pre, suf = q.get("rel_prefix", pre), q.get("rel_suffix", suf)

        prompts = [ALPACA_NO_INPUT.format(instruction=pre + s + suf) for s in surface]
        probs = yes_no_probabilities(model, tok, prompts)
        scores = [py / (py + pn) if (py + pn) > 0 else 0.0 for py, pn in probs]

        order = sorted(range(len(cands)), key=lambda i: -scores[i])
        rank = next(i + 1 for i, j in enumerate(order) if cands[j] == gold)
        gi = cands.index(gold)

        rows.append({
            "qid": k,                                   # ★ for the paired bootstrap
            "head": q["head"], "relation": q["relation"], "tail": q["tail"],
            "gold_label": gold, "pred_label": cands[order[0]],
            "rank": rank, "gold_score": scores[gi],
            "top_score": max(scores), "score_spread": max(scores) - min(scores),
            "n_candidates": len(cands),
            "context_tokens": q["context_tokens"],
        })
        gold_scores.append(scores[gi])
        gold_correct.append(rank == 1)

    del model
    torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("no queries scored — the frozen candidate file and the "
                         "built queries do not overlap. Rebuild both with the "
                         "same --direction and --limit.")

    ranks = [r["rank"] for r in rows]
    toks = [r["context_tokens"] for r in rows]
    mean_tok = sum(toks) / max(1, len(toks))
    m = ranking_metrics(ranks)
    cal = calibration(gold_scores, gold_correct)

    per_rel = defaultdict(list)
    for r in rows:
        per_rel[r["relation"]].append(r["rank"])
    per_rel_m = {k: ranking_metrics(v) for k, v in
                 sorted(per_rel.items(), key=lambda kv: -len(kv[1]))}

    degenerate = (max(r["score_spread"] for r in rows) < 0.05) if rows else False

    # ★ single-cell CI, so even one row carries its own uncertainty
    from .stats import bootstrap_ci, min_detectable_effect
    rr = [1.0 / r for r in ranks]
    ci = bootstrap_ci(rr, seed=ns.seed)

    res = {
        "dataset": ns.dataset, "policy": ns.policy, "budget": ns.budget,
        "adapter": ns.adapter, "untuned": is_untuned,
        "direction": ns.direction, "task": ns.task,
        "n_way": ns.n_way if ns.task == "link" else None,
        "protocol": (f"{ns.n_way}-way, filtered (train ∪ valid ∪ test)"
                     if ns.task == "link" else
                     f"relation prediction over {len(kg.rel2txt)} relations"),
        "candidate_fingerprint": fingerprint(cand_file) if cand_file else None,
        "has_valid_split": bool(kg.valid),
        "ranking": m,
        "MRR_ci": ci,
        "min_detectable_effect_unpaired": min_detectable_effect(rr),
        "per_relation_hits1": per_relation_hits1(rows),
        "calibration": cal,
        "cost": {"mean_context_tokens": mean_tok,
                 "total_context_tokens": sum(toks),
                 "MRR_per_1k_tokens": (m["MRR"] / mean_tok * 1000) if mean_tok else None},
        "per_relation": per_rel_m,
        "degenerate": degenerate,
        "n_short_candidate_sets": n_short,
        "rows": rows,                     # ★ REQUIRED by the paired bootstrap
    }

    if ns.task == "relation":
        labels = sorted(kg.rel2txt) if len(kg.rel2txt) <= ns.rel_cap else \
            sorted({r["gold_label"] for r in rows} | {r["pred_label"] for r in rows})
        res["confusion"] = confusion(rows, labels)

    # ------------------------------------------------------------- printing
    print(f"\n{'='*76}\n{ns.policy}  B={ns.budget}  {ns.direction}  "
          f"{'[UNTUNED]' if is_untuned else ''}\n{'='*76}")
    print(f"  MRR        {m['MRR']:.4f}  [{ci['lo']:.4f}, {ci['hi']:.4f}]   "
          f"MR {m['MR']:.2f}")
    print(f"  Hits@1     {m['hits@1']:.4f}      @3   {m['hits@3']:.4f}   "
          f"@10 {m['hits@10']:.4f}")
    print(f"  ECE        {cal['ECE']:.4f}      Brier   {cal['Brier']:.4f}")
    print(f"  tokens     {mean_tok:.1f} mean   ★ MRR/1k tok "
          f"{res['cost']['MRR_per_1k_tokens']:.4f}")
    print(f"  ⚠️ smallest UNPAIRED difference detectable at n={len(rows)}: "
          f"{res['min_detectable_effect_unpaired']:.4f} MRR")
    print(f"     (report.py uses the PAIRED bootstrap, which does much better)")

    if ns.task == "relation":
        c = res["confusion"]
        print(f"\n  ★ RELATION PREDICTION — a real classification problem")
        print(f"  accuracy   {c['accuracy']:.4f}")
        print(f"  macro-F1   {c['macro_f1']:.4f}      micro-F1 {c['micro_f1']:.4f}   "
              f"({c['n_classes_observed']} classes observed)")
        if c["top_confusion"]:
            t = c["top_confusion"]
            print(f"  most confused: gold {kg.rel2txt.get(t['gold'], t['gold'])} "
                  f"-> predicted {kg.rel2txt.get(t['predicted'], t['predicted'])} "
                  f"({t['n']}x)")
        print(f"\n  {'relation':26s} {'sup':>5s} {'prec':>7s} {'rec':>7s} {'F1':>7s}")
        for lab, v in sorted(c["per_class"].items(),
                             key=lambda kv: -kv[1]["support"])[:12]:
            if not v["support"]:
                continue
            print(f"  {kg.rel2txt.get(lab, lab)[:26]:26s} {v['support']:>5d} "
                  f"{v['precision']:>7.3f} {v['recall']:>7.3f} {v['f1']:>7.3f}")
    else:
        h1 = res["per_relation_hits1"]
        print(f"\n  macro Hits@1 {h1['macro_hits@1']:.4f}   "
              f"micro {h1['micro_hits@1']:.4f}   ({h1['n_relations']} relations)")
        print(f"  ⚠️ per-relation P/R/F1 would all equal Hits@1 here — that is why")
        print(f"     the real F1 and the confusion matrix live in --task relation.")

    if degenerate:
        print("\n  ⚠️ DEGENERATE — all candidates score the same; ranking is arbitrary")
    if n_short:
        print(f"  ⚠️ {n_short} queries had fewer than {ns.n_way} candidates")

    print(f"\n  {'relation':28s} {'n':>5s} {'MRR':>7s} {'H@1':>7s}")
    for rel, mm in list(per_rel_m.items())[:8]:
        print(f"  {kg.rel2txt.get(rel, rel)[:28]:28s} {mm['n']:>5d} "
              f"{mm['MRR']:>7.4f} {mm['hits@1']:>7.4f}")
    if len(per_rel_m) > 8:
        print(f"  … {len(per_rel_m)-8} more relations in the json")

    out = Path(cfg["output"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tag = ns.tag or ("untuned" if is_untuned else "tuned")
    suffix = "" if ns.task == "link" else "_rel"
    dest = out / (f"ch3_{ns.dataset}_{ns.policy}_B{ns.budget}_"
                  f"{ns.direction}_{tag}{suffix}.json")
    dest.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
