"""
★ END-TO-END SMOKE TEST — the whole Chapter 3 pipeline on a synthetic graph,
  with NO GPU, NO transformers and NO downloads. Runs in about two seconds.

    python -m chapter3.test_pipeline

WHAT IT ACTUALLY PROVES
-----------------------
`test_chapter3.py` tests the allocator in isolation. This tests the parts that
only break when the pieces are joined — which is where every bug in this project
so far has actually lived:

    · candidate sets are IDENTICAL across policies and INDEPENDENT of --limit
      ★ the failure mode that would silently unmatch the comparison
    · head and tail prompts put the candidate in the right slot
    · head-direction context describes the TAIL, not the answer
    · the paired bootstrap is calibrated and refuses misaligned inputs
    · the confusion matrix has fp != fn  (the old F1 could not)
    · report.py and qualitative.py parse real result files end to end

Every check prints its own verdict, so a failure names the stage.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append((name, detail))
    print(f"  {'✓' if cond else '✗'} {name:52s} {detail}")


def words(s: str) -> int:
    return len(s.split())


# --------------------------------------------------------------- a fake graph
def make_dataset(root: Path, name: str = "SYNTH-ind", seed: int = 3) -> Path:
    """
    An inductive-shaped graph: test entities never appear in train.
    Written in the real on-disk format so the real loader reads it.
    """
    rng = random.Random(seed)
    d = root / name
    d.mkdir(parents=True, exist_ok=True)

    rels = ["_hypernym", "_hyponym", "_part_of", "_member_of", "_similar_to"]
    train_e = [f"tr{i}" for i in range(60)]
    test_e = [f"te{i}" for i in range(25)]          # unseen — inductive

    (d / "relation2text.txt").write_text(
        "\n".join(f"{r}\t{r.strip('_').replace('_',' ')}" for r in rels),
        encoding="utf-8")
    (d / "entity2text.txt").write_text(
        "\n".join(f"{e}\t{'entity ' + e + ' a short description here'}"
                  if rng.random() < 0.4 else f"{e}\t{e}"
                  for e in train_e + test_e), encoding="utf-8")

    train = [(rng.choice(train_e), rng.choice(rels), rng.choice(train_e))
             for _ in range(400)]
    valid = [(rng.choice(train_e), rng.choice(rels), rng.choice(train_e))
             for _ in range(40)]
    test = [(rng.choice(test_e), rng.choice(rels), rng.choice(train_e))
            for _ in range(60)]

    (d / "train.tsv").write_text(
        "\n".join("\t".join(t) for t in train), encoding="utf-8")
    (d / "valid.tsv").write_text(
        "\n".join("\t".join(t) + "\t1" for t in valid), encoding="utf-8")
    (d / "test.tsv").write_text(
        "\n".join("\t".join(t) + "\t1" for t in test), encoding="utf-8")
    return d


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ch3_smoke_"))
    root = tmp / "data"
    try:
        make_dataset(root)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        from src.data.loaders import load_kg
        from chapter3 import candidates as C
        from chapter3.data import build_one
        from chapter3.policies import POLICIES
        from chapter3.sources import candidate_blocks
        from chapter3.evaluate import confusion, per_relation_hits1, ranking_metrics
        from chapter3 import stats as S

        kg = load_kg("SYNTH-ind", root)

        # ---------------------------------------------------------- LOADER
        print("\nLOADER AND THE FILTERED SET")
        check("valid.tsv is loaded", len(kg.valid) == 40, f"{len(kg.valid)} triples")
        check("all_true() spans train ∪ valid ∪ test",
              len(kg.all_true()) > len(kg.train),
              f"{len(kg.all_true())} true triples vs {len(kg.train)} train")
        tr = {e for t in kg.train for e in (t.head, t.tail)}
        te_heads = {t.head for t in kg.test}
        check("split is inductive (test heads unseen in train)",
              not (te_heads & tr), f"{len(te_heads)} test heads, 0 seen")

        # ------------------------------------------------------ CANDIDATES
        print("\nFROZEN CANDIDATES — the fix that matters most")
        full = C.build(kg, "tail", 20, 1000, 42)
        part = C.build(kg, "tail", 20, 10, 42)           # ★ a DIFFERENT --limit
        shared = set(full) & set(part)
        same = all(full[k]["candidates"] == part[k]["candidates"] for k in shared)
        check("candidates independent of --limit", same and len(shared) > 0,
              f"{len(shared)} shared queries, all byte-identical")

        again = C.build(kg, "tail", 20, 1000, 42)
        check("candidate build is deterministic",
              C.fingerprint(full) == C.fingerprint(again),
              f"fingerprint {C.fingerprint(full)}")

        diff_seed = C.build(kg, "tail", 20, 1000, 43)
        check("a different seed gives different candidates",
              C.fingerprint(full) != C.fingerprint(diff_seed),
              "otherwise the seed is being ignored")

        true_set = kg.all_true()
        leaks = [k for k, r in full.items()
                 for c in r["candidates"]
                 if c != r["gold"] and (r["fixed"], r["relation"], c) in true_set]
        check("no true triple is used as a negative", not leaks,
              f"{len(leaks)} leaks (filtered against train ∪ valid ∪ test)")

        gold_in = all(r["gold"] in r["candidates"] for r in full.values())
        check("gold is always among the candidates", gold_in,
              "or the rank is undefined")

        head = C.build(kg, "head", 20, 1000, 42)
        check("head direction filters on the head slot",
              all(r["fixed"] == r["tail"] and r["gold"] == r["head"]
                  for r in head.values()),
              "gold=head, fixed=tail")

        # ----------------------------------------------------------- PROMPTS
        print("\nPROMPT CONSTRUCTION — both directions")
        rel_desc = {r: f"the relation {r} connects a concept to a category"
                    for r in kg.rel2txt}
        # ⚠️ VARIED types on purpose. With one type for every entity the relation
        #    type-entropy is 0 everywhere, S2 degenerates onto S0 and S4 onto S1 —
        #    and the fixture would then "prove" that policies collide when in fact
        #    only the fixture is degenerate. The degenerate case is tested below,
        #    deliberately, as a check that the guard FIRES.
        trng = random.Random(17)
        types = {e: trng.choice(["Concept", "Animal", "Place", "Work", "Person"])
                 for e in kg.ent2txt}
        qs = kg.test[:20]

        rec_t, qr_t, al_t = build_one(kg, qs, POLICIES["S4_instance"], 120,
                                      rel_desc, types, words, direction="tail")
        rec_h, qr_h, al_h = build_one(kg, qs, POLICIES["S4_instance"], 120,
                                      rel_desc, types, words, direction="head")

        q0, t0 = qr_t[0], qs[0]
        tail_prompt = q0["prefix"] + kg.ent2txt.get(t0.tail, t0.tail) + q0["suffix"]
        check("tail prompt places the candidate last",
              tail_prompt.rstrip("?").endswith(kg.ent2txt.get(t0.tail, t0.tail)),
              tail_prompt[-46:].replace("\n", " "))

        q1 = qr_h[0]
        head_prompt = q1["prefix"] + kg.ent2txt.get(t0.head, t0.head) + q1["suffix"]
        check("head prompt places the candidate mid-sentence",
              kg.ent2txt.get(t0.tail, t0.tail) in q1["suffix"],
              q1["suffix"][:46].replace("\n", " "))

        check("head-direction context describes the TAIL, not the answer",
              all(a["anchor"] == q["tail"] for a, q in zip(al_h, qr_h)),
              "★ describing the head would be describing the gold")
        check("tail-direction context describes the HEAD",
              all(a["anchor"] == q["head"] for a, q in zip(al_t, qr_t)), "")

        rp = q0["rel_prefix"] + kg.rel2txt[t0.relation] + q0["rel_suffix"]
        check("relation-prediction slot is between head and tail",
              kg.ent2txt.get(t0.head, t0.head) in q0["rel_prefix"]
              and kg.ent2txt.get(t0.tail, t0.tail) in q0["rel_suffix"],
              rp[-46:].replace("\n", " "))

        # budget really enforced, both directions
        over = [q for q in (*qr_t, *qr_h) if q["context_tokens"] > 120]
        check("budget enforced in both directions", not over,
              f"max spend {max(q['context_tokens'] for q in (*qr_t, *qr_h))}/120")

        # policies still differ once direction is in play
        def prompts_for(pid, t, budget=40, ix=None):
            _, qq, _ = build_one(kg, qs, POLICIES[pid], budget, rel_desc, t,
                                 words, direction="tail", index=ix)
            return tuple(q["prefix"] + q["suffix"] for q in qq)

        ladder = ("S0_uniform", "R_random", "S1_property", "S2_type",
                  "S4_instance", "S5_semantic")
        pr = {pid: prompts_for(pid, types) for pid in ladder}
        dupes = {}
        for k, v in pr.items():
            dupes.setdefault(v, []).append(k)
        collided = [ks for ks in dupes.values() if len(ks) > 1]
        check("policies produce distinct prompts on a varied graph",
              len(set(pr.values())) == len(pr),
              f"{len(set(pr.values()))}/{len(pr)} distinct"
              + (f"  collisions={collided}" if collided else ""))

        # ------------------------------------------- ★ INDUCTIVE NEIGHBOURS
        print("\n★ INDUCTIVE NEIGHBOURS — the bug that would have faked a null result")
        from chapter3.sources import GraphIndex, assert_no_leak
        idx = GraphIndex(kg, use_inference_graph=True)
        train_only = GraphIndex(kg, use_inference_graph=False)

        n_with = sum(1 for t in kg.test
                     if idx.neighbours_of(t.head, (t.head, t.relation, t.tail), 5))
        n_without = sum(1 for t in kg.test
                        if train_only.neighbours_of(t.head, None, 5))
        check("train-only index gives unseen entities NO neighbours",
              n_without == 0, f"{n_without}/{len(kg.test)} — the old behaviour")
        check("★ inference graph restores them",
              n_with > 0.5 * len(kg.test),
              f"{n_with}/{len(kg.test)} queries now have neighbours")

        blocks = [candidate_blocks(kg, t.head, t.relation, rel_desc, types,
                                   words, index=idx,
                                   exclude=(t.head, t.relation, t.tail))
                  for t in kg.test[:40]]
        check("neighbours block is emitted for inductive queries",
              sum(any(b.kind == "neighbours" for b in bl) for bl in blocks) > 20,
              f"{sum(any(b.kind=='neighbours' for b in bl) for bl in blocks)}/40")

        # ⚠️ THE LEAK — the reason the fix needs a guard
        leaked = 0
        for t, bl in zip(kg.test[:40], blocks):
            try:
                assert_no_leak(bl, kg, t.head, t.relation, t.tail)
            except AssertionError:
                leaked += 1
        check("★ gold answer never appears in the context", leaked == 0,
              f"{leaked}/40 leaks with the query triple excluded")

        # and prove the guard is not vacuous: it must catch a real leak
        bad = candidate_blocks(kg, kg.test[0].head, kg.test[0].relation, rel_desc,
                               types, words, index=idx, exclude=None)
        caught = False
        try:
            assert_no_leak(bad, kg, kg.test[0].head, kg.test[0].relation,
                           kg.test[0].tail)
        except AssertionError:
            caught = True
        check("★ leak guard is not vacuous (catches an unexcluded query)", caught,
              "exclude=None must be detected, or the guard proves nothing")

        # the index must be built once, not per call
        import time
        t0 = time.time()
        for t in kg.test:
            candidate_blocks(kg, t.head, t.relation, rel_desc, types, words,
                             index=idx, exclude=(t.head, t.relation, t.tail))
        shared = time.time() - t0
        t0 = time.time()
        for t in kg.test[:15]:
            candidate_blocks(kg, t.head, t.relation, rel_desc, types, words)
        rebuilt = (time.time() - t0) / 15 * len(kg.test)
        check("shared index is faster than rebuilding per call",
              shared < rebuilt,
              f"{shared*1000:.0f}ms vs {rebuilt*1000:.0f}ms projected "
              f"({rebuilt/max(shared,1e-9):.0f}x on a 400-triple graph)")

        # ★★ AND THE REGRESSION TEST FOR THE BUG ITSELF.
        #    Rebuild WITHOUT the inference graph — the old behaviour. Unseen
        #    entities then have no neighbours, S2's only decision (neighbours vs
        #    type tag) has nothing to act on, and it produces byte-identical
        #    prompts to S0. That collapse is what would have been reported as
        #    "specificity does not pay", so it must stay pinned by a test.
        pf = {pid: prompts_for(pid, types, ix=train_only)
              for pid in ("S0_uniform", "S2_type", "S4_instance", "S1_property")}
        check("★★ REGRESSION: without the inference graph the ladder collapses",
              len(set(pf.values())) < len(pf),
              f"{len(set(pf.values()))}/{len(pf)} distinct — the bug, pinned")
        check("★★ ...and WITH it the same policies stay distinct",
              len({prompts_for(p, types, ix=idx) for p in
                   ("S0_uniform", "S2_type", "S4_instance", "S1_property")}) == 4,
              "4/4 distinct — the fix, pinned")

        # ------------------------------------------------------------- STATS
        print("\nSTATISTICS")
        rng = random.Random(11)
        base = [rng.random() for _ in range(400)]
        a = [x + 0.10 + rng.gauss(0, 0.1) for x in base]
        b = [x + rng.gauss(0, 0.1) for x in base]
        d = S.paired_bootstrap(a, b)
        check("paired bootstrap detects a real effect",
              d["significant"] and d["diff"] > 0, S.fmt_diff(d))

        null = S.paired_bootstrap([x + rng.gauss(0, .1) for x in base],
                                  [x + rng.gauss(0, .1) for x in base])
        check("null case is reported as unmeasurable, not as a small gain",
              "INDISTINGUISHABLE" in S.verdict(null), S.verdict(null)[:52])

        try:
            S.align([{"qid": "a", "rank": 1}], [{"qid": "z", "rank": 1}])
            ok = False
        except ValueError:
            ok = True
        check("align() refuses cells with no shared queries", ok,
              "silently comparing different test sets is the danger")

        try:
            S.align([{"qid": f"q{i}", "rank": 1} for i in range(100)],
                    [{"qid": f"q{i}", "rank": 1} for i in range(50)])
            ok = False
        except ValueError:
            ok = True
        check("align() refuses a <90% overlap", ok, "mismatched --limit is caught")

        # ------------------------------------------------------- CONFUSION
        print("\nCONFUSION MATRIX — the fix to the fake F1")
        labels = sorted(kg.rel2txt)
        rows = []
        rng2 = random.Random(5)
        for i in range(200):
            g = rng2.choice(labels)
            p = g if rng2.random() < 0.55 else rng2.choice(labels)
            rows.append({"gold_label": g, "pred_label": p, "relation": g,
                         "rank": 1 if g == p else rng2.randint(2, 5)})
        c = confusion(rows, labels)
        asym = [l for l, v in c["per_class"].items() if v["fp"] != v["fn"]]
        check("fp != fn for some class (the old code could not)", bool(asym),
              f"{len(asym)}/{len(labels)} classes asymmetric")
        check("precision differs from recall somewhere",
              any(abs(v["precision"] - v["recall"]) > 1e-9
                  for v in c["per_class"].values()),
              "★ proof this is a real classification metric now")
        check("matrix totals match the row count",
              sum(map(sum, c["matrix"])) == len(rows), f"{len(rows)} rows")
        check("accuracy equals the diagonal share",
              abs(c["accuracy"] - sum(r["gold_label"] == r["pred_label"]
                                      for r in rows) / len(rows)) < 1e-9,
              f"{c['accuracy']:.3f}")
        check("top_confusion is off-diagonal",
              c["top_confusion"] is None
              or c["top_confusion"]["gold"] != c["top_confusion"]["predicted"], "")

        h1 = per_relation_hits1([{"relation": r["relation"], "rank": r["rank"]}
                                 for r in rows])
        check("link-task view is labelled hits@1, not F1",
              "macro_hits@1" in h1 and "note" in h1,
              "the name now matches the arithmetic")

        # ------------------------------------------- REPORT + QUALITATIVE
        print("\nREPORT AND QUALITATIVE, END TO END")
        results = tmp / "results"
        results.mkdir()
        built = root / "SYNTH-ind" / "built"

        def emit(pid, budget, direction, tag, boost):
            recs, qq, aa = build_one(kg, qs, POLICIES[pid], budget, rel_desc,
                                     types, words, direction=direction)
            dd = built / f"{pid}_B{budget}_{direction}"
            dd.mkdir(parents=True, exist_ok=True)
            (dd / "queries.json").write_text(json.dumps(qq), encoding="utf-8")
            (dd / "allocations.json").write_text(json.dumps(aa), encoding="utf-8")
            (dd / "train_instructions.json").write_text(json.dumps(recs),
                                                        encoding="utf-8")
            rr = random.Random(hash(pid) & 0xFFFF)
            rws = []
            for q in qq:
                base_rank = rr.randint(1, 20)
                rank = max(1, base_rank - boost)
                rws.append({
                    "qid": C.qid(q["head"], q["relation"], q["tail"], direction),
                    "head": q["head"], "relation": q["relation"], "tail": q["tail"],
                    "gold_label": q["tail"], "pred_label": q["tail"],
                    "rank": rank, "gold_score": 1.0 / rank, "top_score": 0.9,
                    "score_spread": 0.4, "n_candidates": 20,
                    "context_tokens": q["context_tokens"]})
            ranks = [r["rank"] for r in rws]
            mt = sum(r["context_tokens"] for r in rws) / len(rws)
            m = ranking_metrics(ranks)
            res = {
                "dataset": "SYNTH-ind", "policy": pid, "budget": budget,
                "direction": direction, "task": "link", "n_way": 20,
                "adapter": None if tag == "untuned" else "x",
                "untuned": tag == "untuned",
                "protocol": "20-way, filtered", "has_valid_split": True,
                "candidate_fingerprint": "deadbeef1234",
                "ranking": m,
                "MRR_ci": S.bootstrap_ci([1 / r for r in ranks]),
                "per_relation_hits1": per_relation_hits1(rws),
                "calibration": {"ECE": 0.1, "Brier": 0.2},
                "cost": {"mean_context_tokens": mt,
                         "total_context_tokens": mt * len(rws),
                         "MRR_per_1k_tokens": m["MRR"] / mt * 1000 if mt else 0},
                "per_relation": {}, "degenerate": False,
                "n_short_candidate_sets": 0, "rows": rws}
            (results / f"ch3_SYNTH-ind_{pid}_B{budget}_{direction}_{tag}.json"
             ).write_text(json.dumps(res), encoding="utf-8")

        # ORACLE is retired: it kept no blocks and reproduced the B=0 floor.
        # The ceiling is now report.policy_selection_oracle, computed post hoc.
        for pid, boost in (("S0_uniform", 0), ("R_random", 0),
                           ("S1_property", 3), ("S4_instance", 6)):
            emit(pid, 120, "tail", "tuned", boost)
            emit(pid, 120, "tail", "untuned", max(0, boost - 3))
            emit(pid, 60, "tail", "tuned", boost // 2)
            emit(pid, 120, "head", "tuned", boost)

        cfgp = tmp / "cfg.yaml"
        cfgp.write_text(f"seed: 42\nmodel:\n  name: dummy\n"
                        f"output:\n  results_dir: {results}\n", encoding="utf-8")

        env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        import os
        env = {**os.environ, **env}

        r = subprocess.run(
            [sys.executable, "-m", "chapter3.report", "--dataset", "SYNTH-ind",
             "--config", str(cfgp), "--results", str(results),
             "--compare-untuned", "--both-directions", "--n-boot", "300"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
            env=env)
        out = r.stdout
        check("report.py runs", r.returncode == 0,
              (r.stderr.strip().splitlines() or [""])[-1][:56])
        check("report prints confidence intervals", "[" in out and "]" in out, "")
        check("report runs the paired bootstrap",
              "paired" in out and ("p<" in out or "p=" in out), "")
        check("report reaches a headline verdict", "➤" in out, "")
        check("report shows the post-hoc ceiling",
              "CEILING — best policy PER QUERY" in out, "")
        check("★ ceiling is above every single policy",
              "over best single policy +" in out,
              "a routing bound that is below its own policies is a bug")
        check("report compares tuned vs untuned", "WITHOUT FINE-TUNING" in out, "")
        check("report compares both directions", "BOTH DIRECTIONS" in out, "")
        check("report flags the missing confusion matrix",
              "no real F1" in out or "RELATION PREDICTION" in out, "")
        check("★ significant improvement is detected",
              "ALLOCATION PAYS" in out or "★" in out, "")

        r2 = subprocess.run(
            [sys.executable, "-m", "chapter3.qualitative", "--dataset", "SYNTH-ind",
             "--root", str(root), "--config", str(cfgp), "--results", str(results),
             "--a", "S0_uniform", "--b", "S4_instance", "--budget", "120",
             "--n", "2", "--latex"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
            env=env)
        o2 = r2.stdout
        check("qualitative.py runs", r2.returncode == 0,
              (r2.stderr.strip().splitlines() or [""])[-1][:56])
        check("qualitative shows both allocations",
              "S0_uniform" in o2 and "S4_instance" in o2, "")
        check("qualitative shows the rank change", "rank" in o2, "")
        check("qualitative states the selection rule", "selected by rule" in o2, "")
        check("qualitative emits LaTeX", "\\begin{tabular}" in o2, "")

        r3 = subprocess.run(
            [sys.executable, "-m", "chapter3.qualitative", "--dataset", "SYNTH-ind",
             "--root", str(root), "--config", str(cfgp), "--results", str(results),
             "--a", "S0_uniform", "--b", "S4_instance", "--budget", "120",
             "--only-disagreements", "--worst", "--n", "2"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
            env=env)
        check("qualitative --worst / --only-disagreements run",
              r3.returncode == 0, "regressions must be reportable too")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 74)
    if FAIL:
        print(f"{len(FAIL)} FAILED of {len(PASS)+len(FAIL)}")
        for n, d in FAIL:
            print(f"   ✗ {n}  {d}")
        sys.exit(1)
    print(f"ALL {len(PASS)} PIPELINE CHECKS PASSED")


if __name__ == "__main__":
    main()
