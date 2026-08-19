"""
★★ THE SECOND INSTRUMENT — free, no GPU, runs on results you already have.

    python -m chapter1.seen_unseen --dataset WN11

WHY THIS IS THE MOST IMPORTANT CHEAP THING IN THE CHAPTER
---------------------------------------------------------
The whole memorisation claim currently rests on ONE instrument (anonymisation),
and that instrument has one obvious attack:

    "Replacing names with entity4471 destroys ALL information, so of course
     accuracy collapses. That tells us nothing."

This answers it from a completely different direction. **Names stay intact.**
We only ask whether accuracy depends on having SEEN the entity during training.

    both entities seen in training   -> high accuracy?
    neither seen                     -> collapse?

If yes, memorisation is confirmed **without touching the data at all**. Two
instruments agreeing from opposite directions is a different class of claim from
one instrument asserting.

IT COSTS NOTHING
----------------
We trained on 10,000 of WN11's 112,581 triples, so only ~35% of entities were
ever seen and only ~22% of test triples have both. That accidental inductive
split is sitting in the results already -- it just was never read.

★ It is also the ENRICHMENT argument, made measurable. Enrichment means new
entities; the "neither seen" bucket IS the enrichment case.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def seen_entities(dataset: str, root: str, n_triples: int, seed: int) -> set[str]:
    """
    Re-derive exactly which entities training touched.

    Deterministic: `sample_triples` with the same seed and the same stratified
    settings reproduces the identical sample, so this needs no stored state.
    """
    from src.data.loaders import load_kg
    from src.data.sampling import sample_triples
    from src.utils.config import set_all_seeds
    set_all_seeds(seed)

    kg = load_kg(dataset, root)
    pos = sample_triples(kg.train, n_triples, seed=seed,
                         stratified=True, min_per_relation=10)
    return {e for t in pos for e in (t.head, t.tail)}, kg


def buckets(records, seen):
    h = np.array([r.get("head") in seen for r in records])
    t = np.array([r.get("tail") in seen for r in records])
    return {"both_seen": h & t, "one_seen": h ^ t, "neither": (~h) & (~t)}


def analyse(conf, labels, records, seen, name, anon_gap=None):
    p = np.clip(np.asarray(conf, float), 0, 1)
    y = np.asarray(labels, int)
    pred = np.where(p >= 0.5, 1, -1)
    correct = pred == y
    conf_mag = np.maximum(p, 1 - p)          # certainty, direction-independent

    B = buckets(records, seen)
    out = {"condition": name, "n": int(len(y)), "buckets": {}}
    for k, m in B.items():
        n = int(m.sum())
        # ★★ RAW ACCURACY IS NOT COMPARABLE ACROSS THESE BUCKETS.
        #
        #    Measured on WN11 (2,000 items): the positive rate is
        #        both_seen 60.7%   one_seen 50.0%   neither 40.7%
        #    The buckets are NOT balanced, even though the test set as a whole
        #    is exactly 50/50. Frequent entities are both likelier to be sampled
        #    into training AND likelier to appear in a true triple.
        #
        #    Consequence: a model that always answers "Yes" scores 0.607 / 0.500
        #    / 0.407 and shows a +0.20 "familiarity gap" having learned nothing.
        #    A "No"-biased model shows the same gap with the sign flipped. Both
        #    are artefacts of the base rate.
        #
        #    BALANCED ACCURACY (mean of per-class recall) removes it: its chance
        #    level is 0.5 in every bucket regardless of skew. Read that column.
        if n:
            yb, cb = y[m], correct[m]
            recalls = [float(cb[yb == c].mean()) for c in (1, -1) if (yb == c).any()]
            bal = float(np.mean(recalls)) if recalls else None
            pos_rate = float((yb == 1).mean())
        else:
            bal, pos_rate = None, None
        out["buckets"][k] = {
            "n": n,
            "share": n / max(1, len(y)),
            "accuracy": float(correct[m].mean()) if n else None,
            "balanced_accuracy": bal,
            "positive_rate": pos_rate,
            "majority_baseline": (max(pos_rate, 1 - pos_rate)
                                  if pos_rate is not None else None),
            "mean_confidence": float(conf_mag[m].mean()) if n else None,
        }

    b, ne = out["buckets"]["both_seen"], out["buckets"]["neither"]
    if b["balanced_accuracy"] is not None and ne["balanced_accuracy"] is not None:
        # ★ the gap that matters is the BALANCED one -- see the note above
        out["familiarity_gap"] = b["balanced_accuracy"] - ne["balanced_accuracy"]
        out["familiarity_gap_raw"] = b["accuracy"] - ne["accuracy"]
        out["base_rate_artefact"] = b["majority_baseline"] - ne["majority_baseline"]
        out["confidence_gap"] = b["mean_confidence"] - ne["mean_confidence"]
        # ★ Is the model HONEST about not knowing? Confidence should fall as much
        # as accuracy does. If it does not, it is overconfident exactly where it
        # has no memorised answer -- the worst case for deployment, and the
        # strongest possible argument for abstention (Chapter 4).
        out["overconfident_on_unseen"] = (
            out["familiarity_gap"] - out["confidence_gap"] > 0.05)

        # ★★ THE JOINT READING. This instrument alone cannot interpret itself.
        #
        #   A flat familiarity curve does NOT weaken the memorisation claim --
        #   that was the old verdict text and it was wrong. What the two
        #   instruments say TOGETHER is what matters:
        #
        #     big anonymisation gap + flat familiarity
        #         -> the memorised knowledge came from PRETRAINING, not from the
        #            fine-tuning sample. The claim is LOCALISED, not weakened,
        #            and the obvious alternative explanation is ruled out.
        #     big anonymisation gap + steep familiarity
        #         -> training-set recall is also contributing.
        #     small anonymisation gap
        #         -> no memorisation claim to make either way.
        g = out["familiarity_gap"]
        flat = abs(g) < 0.05
        if anon_gap is None:
            out["verdict"] = (f"familiarity gap {g:+.4f} (balanced). No "
                              f"anonymisation gap supplied, so no joint reading.")
        elif anon_gap > 0.10 and flat:
            out["verdict"] = (
                f"★★ MEMORISATION IS PRETRAINED, NOT TRAINING-SET RECALL. "
                f"Anonymisation removes {anon_gap:+.4f}, yet the balanced "
                f"familiarity gap is only {g:+.4f} — withholding the entity from "
                f"fine-tuning changes nothing. Names carry the accuracy; the "
                f"knowledge behind them predates this training run. This LOCALISES "
                f"the claim rather than weakening it, and rules out the obvious "
                f"alternative explanation.")
        elif anon_gap > 0.10:
            out["verdict"] = (
                f"★ TWO SOURCES. Anonymisation removes {anon_gap:+.4f} and "
                f"familiarity is worth a further {g:+.4f} (balanced) — so "
                f"training-set recall contributes on top of pretrained knowledge.")
        else:
            out["verdict"] = (
                f"anonymisation gap is only {anon_gap:+.4f}, so there is no "
                f"memorisation effect for this instrument to localise "
                f"(familiarity gap {g:+.4f}).")
    return out


def show(a):
    print(f"\n{'─' * 78}\n  {a['condition']}   (n={a['n']:,})\n{'─' * 78}")
    print(f"  {'bucket':12s} {'n':>6s} {'%pos':>7s} {'raw acc':>9s} "
          f"{'BAL ACC':>9s} {'conf':>8s}")
    for k in ("both_seen", "one_seen", "neither"):
        b = a["buckets"][k]
        f = lambda v, w=9: (f"{v:.4f}".rjust(w) if v is not None else "—".rjust(w))
        print(f"  {k:12s} {b['n']:>6,d} {b['positive_rate']:>6.1%} "
              f"{f(b['accuracy'])} {f(b['balanced_accuracy'])} "
              f"{f(b['mean_confidence'], 8)}")
    if "familiarity_gap" in a:
        print(f"\n  ★ familiarity gap (BALANCED acc) {a['familiarity_gap']:+.4f}   "
              f"<- the one to report")
        print(f"    familiarity gap (raw acc)      {a['familiarity_gap_raw']:+.4f}")
        print(f"    of which base-rate artefact    {a['base_rate_artefact']:+.4f}   "
              f"(bucket skew alone, no model)")
        print(f"    familiarity gap (confidence)   {a['confidence_gap']:+.4f}")
        if a.get("overconfident_on_unseen"):
            print("  ⚠️ OVERCONFIDENT ON UNSEEN — confidence falls less than accuracy.")
            print("     The model does not know that it does not know. ★ This is the")
            print("     strongest argument for abstention in the whole thesis.")
        print(f"\n  {a['verdict']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WN11")
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ch1-json", default=None)
    ns = ap.parse_args()

    from src.utils.config import load_config
    cfg = load_config(ns.config)
    n_tr, seed = cfg["data"]["train_triples"], cfg["seed"]

    src = Path(ns.ch1_json or Path(ns.results, f"ch1_{ns.dataset}.json"))
    if not src.exists():
        raise SystemExit(f"{src} not found — this reads an EXISTING ch1 result")

    d = json.loads(src.read_text(encoding="utf-8"))
    seen, kg = seen_entities(ns.dataset, ns.data, n_tr, seed)
    print(f"[seen] training touched {len(seen):,} of {len(kg.ent2txt):,} entities "
          f"({len(seen)/len(kg.ent2txt):.1%}) at {n_tr:,} triples, seed {seed}")

    # ★ build_condition writes to data/{dataset}-{condition}/built/, so the old
    #   path data/{dataset}/built/ never existed and this script could not run.
    #   Condition A is the right test set: real names, matching the `tuned` arm.
    candidates = [Path(ns.data, f"{ns.dataset}-A", "built", "test_instructions.json"),
                  Path(ns.data, ns.dataset, "built", "test_instructions.json")]
    built = next((p for p in candidates if p.exists()), None)
    if built is None:
        raise SystemExit(
            "no built test set found. Looked in:\n  "
            + "\n  ".join(str(p) for p in candidates)
            + f"\n\nBuild it first:  python -m chapter1.data --condition A "
              f"--dataset {ns.dataset}")
    print(f"[seen] test records <- {built}")
    recs = json.loads(built.read_text(encoding="utf-8"))

    # older builds have no head/tail columns -- recover them from kg.test order,
    # which build_instructions preserves exactly
    if recs and "head" not in recs[0]:
        print("[seen] test_instructions.json has no head/tail — recovering from kg.test")
        for r, t in zip(recs, kg.test, strict=True):
            r["head"], r["tail"], r["relation"] = t.head, t.tail, t.relation

    # ★ the other instrument's number, so the verdict can be a JOINT reading
    _t = (d.get("tuned") or {}).get("logit", {}).get("accuracy")
    _a = (d.get("tuned_anon") or {}).get("logit", {}).get("accuracy")
    anon_gap = (_t - _a) if (_t is not None and _a is not None) else None
    if anon_gap is not None:
        print(f"[seen] anonymisation gap from the same file: {anon_gap:+.4f} "
              f"(tuned {_t:.4f} - anon {_a:.4f})")

    results = []
    for cond in ("untuned", "tuned", "tuned_anon"):
        blk = d.get(cond)
        if not blk:
            continue

        # ★★ DIRECTION. `confidences` is LogitParser.confidence =
        #    |p_yes - p_no| / (p_yes + p_no) -- a CERTAINTY MAGNITUDE. It says how
        #    sure the model was, NOT which way it answered. Thresholding it at 0.5
        #    as if it were P(Yes) silently produces a chance-level prediction and
        #    a plausible-looking familiarity gap. Use p_yes/p_no when present.
        if "p_yes" in blk and "p_no" in blk:
            py = np.asarray(blk["p_yes"], float)
            pn = np.asarray(blk["p_no"], float)
            tot = np.where(py + pn > 0, py + pn, 1.0)
            score = py / tot                       # directional, in [0,1]
        else:
            raise SystemExit(
                f"\n✋ '{cond}' logs only `confidences`, which is an UNDIRECTED\n"
                f"   certainty magnitude (|p_yes - p_no| / total). It cannot say\n"
                f"   whether the model answered Yes or No, so no familiarity split\n"
                f"   computed from it is meaningful.\n\n"
                f"   Re-run the evaluation with a build that logs `p_yes` and\n"
                f"   `p_no` per record, then re-run this script.\n\n"
                f"   ⚠️ Any previously produced seen/unseen table is INVALID.")

        n = min(len(score), len(recs))
        y = np.array([r["label"] for r in recs[:n]], int)

        # ---- GUARD: do these scores reproduce the accuracy stored beside them?
        reported = (blk.get("logit") or {}).get("accuracy")
        got = float((np.where(score[:n] >= 0.5, 1, -1) == y).mean())
        if reported is not None and abs(got - reported) > 0.02:
            raise SystemExit(
                f"\n✋ '{cond}': the logged scores do NOT reproduce the logged "
                f"accuracy.\n     reported {reported:.4f}   recomputed {got:.4f}\n"
                f"   The two were not produced by the same run, or the records are\n"
                f"   misaligned with the scores. Refusing to report a familiarity\n"
                f"   split that would look plausible and mean nothing.")

        a = analyse(score[:n], y, recs[:n], seen, cond, anon_gap)
        show(a)
        results.append(a)

    if not results:
        raise SystemExit("no conditions had usable per-record scores")

    out = Path(ns.results, f"ch1_seen_unseen_{ns.dataset}.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out}")
    print("\n★ Put this table beside the anonymisation table. Two independent")
    print("  instruments agreeing is much harder to dismiss than either alone.")


if __name__ == "__main__":
    main()
