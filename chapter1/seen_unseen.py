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


def analyse(conf, labels, records, seen, name):
    p = np.clip(np.asarray(conf, float), 0, 1)
    y = np.asarray(labels, int)
    pred = np.where(p > 0.5, 1, -1)
    correct = pred == y
    conf_mag = np.maximum(p, 1 - p)          # certainty, direction-independent

    B = buckets(records, seen)
    out = {"condition": name, "n": int(len(y)), "buckets": {}}
    for k, m in B.items():
        n = int(m.sum())
        out["buckets"][k] = {
            "n": n,
            "share": n / max(1, len(y)),
            "accuracy": float(correct[m].mean()) if n else None,
            "mean_confidence": float(conf_mag[m].mean()) if n else None,
        }

    b, ne = out["buckets"]["both_seen"], out["buckets"]["neither"]
    if b["accuracy"] is not None and ne["accuracy"] is not None:
        out["familiarity_gap"] = b["accuracy"] - ne["accuracy"]
        out["confidence_gap"] = b["mean_confidence"] - ne["mean_confidence"]
        # ★ Is the model HONEST about not knowing? Confidence should fall as much
        # as accuracy does. If it does not, it is overconfident exactly where it
        # has no memorised answer -- the worst case for deployment, and the
        # strongest possible argument for abstention (Chapter 4).
        out["overconfident_on_unseen"] = (
            out["familiarity_gap"] - out["confidence_gap"] > 0.05)
        out["verdict"] = (
            f"★ MEMORISATION CONFIRMED by a second instrument — accuracy tracks "
            f"familiarity ({b['accuracy']:.4f} seen vs {ne['accuracy']:.4f} unseen, "
            f"gap {out['familiarity_gap']:+.4f}) with names LEFT INTACT"
            if out["familiarity_gap"] > 0.10 else
            f"accuracy does NOT depend on familiarity (gap "
            f"{out['familiarity_gap']:+.4f}) — the anonymisation result must come "
            f"from something other than having seen the entity. Report this: it "
            f"WEAKENS the memorisation reading and is worth knowing.")
    return out


def show(a):
    print(f"\n{'─' * 74}\n  {a['condition']}   (n={a['n']:,})\n{'─' * 74}")
    print(f"  {'bucket':12s} {'n':>7s} {'share':>8s} {'accuracy':>10s} {'confidence':>11s}")
    for k in ("both_seen", "one_seen", "neither"):
        b = a["buckets"][k]
        acc = f"{b['accuracy']:.4f}" if b["accuracy"] is not None else "     —"
        cf = f"{b['mean_confidence']:.4f}" if b["mean_confidence"] is not None else "     —"
        print(f"  {k:12s} {b['n']:>7,d} {b['share']:>8.1%} {acc:>10s} {cf:>11s}")
    if "familiarity_gap" in a:
        print(f"\n  familiarity gap (accuracy)   {a['familiarity_gap']:+.4f}")
        print(f"  familiarity gap (confidence) {a['confidence_gap']:+.4f}")
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
        for r, t in zip(recs, kg.test):
            r["head"], r["tail"], r["relation"] = t.head, t.tail, t.relation

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

        a = analyse(score[:n], y, recs[:n], seen, cond)
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
