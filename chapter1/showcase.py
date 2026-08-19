"""
★★ WHAT THE MODEL ACTUALLY DOES — random samples, side by side, saved.

    python -m chapter1.showcase --dataset YAGO3-10              # ranking
    python -m chapter1.showcase --dataset WN11 --task classify  # yes/no
    python -m chapter1.showcase --dataset WN11 --n 12 --seed 7

WHY THIS EXISTS
---------------
Every number this chapter reports is an aggregate. "94.0 % of above-chance skill
is entity surface form" is a claim about 500 queries at once, and a reader has
no way to check what it looks like on any single one. KG-LLM's Table VI is
qualitative -- worked examples of the model's own output -- and it is one of the
most-read parts of that paper. We had none.

This module reads what the runs already saved and prints the same query answered
by every arm, so the decomposition can be SEEN rather than asserted:

    Alastair Sim -- died in -- ?          gold: London
      A  real names   ->  London                    rank  1   CORRECT
      S  permuted     ->  Littlefield, Texas        rank 19   wrong, still a place
      B  anonymised   ->  Bradley Wright-Phillips   rank 48   wrong, a person

Binding lost at S, readability lost at B, in one line.

NO GPU. Everything comes from results/*.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ARMS = ["A", "S", "B", "C", "G"]
LABEL = {"A": "real names", "S": "permuted", "B": "anonymised",
         "C": "anon + types", "G": "real + types"}

# ── plain-English names, for the --story report ─────────────────────────────
# Nobody outside the field reads "condition B, MRR 0.1336". These are the same
# arms said in words a supervisor, a jury member or a colleague can follow.
PLAIN = {
    "A": "with the real names",
    "S": "with the names swapped around",
    "B": "with the names hidden",
    "C": "with names hidden, categories shown",
    "G": "with real names AND categories",
}
WHY = {
    "A": "this is the normal setup everyone publishes",
    "S": "every name is still a real word — only WHO owns it changed",
    "B": "names replaced by codes like entity4471",
    "C": "no names, but we tell it what kind of thing each one is",
    "G": "names plus categories, to see if categories add anything",
}


def _load(res: Path, pat: str) -> dict:
    """Newest file matching `pat`, or {} — a missing arm is not an error."""
    hits = sorted(res.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return json.loads(hits[0].read_text(encoding="utf-8")) if hits else {}


# =============================================================================
#  RANKING
# =============================================================================
def ranking_samples(res: Path, dataset: str, n: int, seed: int,
                    tags: tuple[str, ...] = ("fixed", "P0")) -> list[dict]:
    # ★ EXACT names only. A `ch1rank-{ds}-{arm}-*.json` wildcard used to be the
    #   last resort, sorted by mtime -- so a freshly written `-untuned` run
    #   would win and be compared, unlabelled, against tuned arms. Two arms
    #   that differ in whether the model was tuned are not a decomposition.
    #   `-fixed` first because it supersedes `-P0` for the S arm.
    arms, used, mets = {}, {}, {}
    for a in ARMS:
        for tag in tags:
            p = res / f"ch1rank-{dataset}-{a}-{tag}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("ranks"):
                arms[a] = {(r["head"], r["relation"], r["tail"]): r for r in d["ranks"]}
                used[a] = p.name
                mets[a] = d.get("metrics", {})
                break
    if not arms:
        raise SystemExit(
            f"no ch1rank-{dataset}-<arm>-{{{','.join(tags)}}}.json in {res}.\n"
            f"  run chapter1.rank first. (Files with any other tag -- -untuned,\n"
            f"  custom --tag -- are ignored on purpose: they are not comparable\n"
            f"  to the tuned arms.)")
    print("reading:  " + "   ".join(f"{a}<-{f}" for a, f in used.items()))

    shared = set.intersection(*(set(v) for v in arms.values()))
    if not shared:
        raise SystemExit("the arms share no queries — were they run with the same --limit?")
    # ★ each arm's query count matters: an intersection much smaller than the
    #   smallest arm means the runs are not aligned and the side-by-side is a
    #   comparison of different questions.
    sizes = {a: len(v) for a, v in arms.items()}
    if len(shared) < 0.9 * min(sizes.values()):
        print(f"  ⚠️ only {len(shared)} shared queries against arm sizes {sizes} — "
              f"the arms were not run over the same query set")
    keys = random.Random(seed).sample(sorted(shared), min(n, len(shared)))

    out = []
    for k in keys:
        any_row = arms[next(iter(arms))][k]
        row = {"head": any_row.get("head_text", k[0]),
               "relation": any_row.get("relation_text", k[1]),
               "gold": any_row.get("gold_text", k[2]), "arms": {}}
        for a, m in arms.items():
            r = m[k]
            row["arms"][a] = {
                "rank": r["rank"],
                "top1": r.get("top5", [{}])[0].get("text", r.get("top1")),
                "score_top": r.get("score_top"), "score_gold": r.get("score_true"),
                "top5": [(c.get("text"), round(c.get("score", 0), 4))
                         for c in r.get("top5", [])],
            }
        out.append(row)
    return out, mets


# =============================================================================
#  PLAIN-ENGLISH REPORT  — for people who do not read MRR
# =============================================================================
def _pretty(s: str) -> str:
    """`England_national_football_team` -> `England national football team`."""
    return str(s).replace("_", " ").strip()


def _question(head: str, rel: str) -> str:
    """Turn a triple into an English sentence with a blank at the end."""
    r = _pretty(rel)
    # a few relations read better as a real question; everything else falls
    # back to the fill-in-the-blank form, which is always grammatical enough
    for key, tmpl in (
        ("born", 'Where was {h} born?'),
        ("died", 'Where did {h} die?'),
        ("located", 'Where is {h} located?'),
        ("capital", 'What is the capital of {h}?'),
        ("language", 'What language does {h} use?'),
        ("graduated", 'Where did {h} study?'),
        ("plays for", 'Which team does {h} play for?'),
        ("affiliated", 'Which organisation is {h} part of?'),
        ("acted in", 'What did {h} act in?'),
        ("directed", 'What did {h} direct?'),
        ("created", 'What did {h} create?'),
        ("won", 'What prize did {h} win?'),
    ):
        if key in r.lower():
            return tmpl.format(h=_pretty(head))
    return f'"{_pretty(head)} {r} ______"  — fill in the blank'


def _verdict(rank: int, n_way: int) -> str:
    if rank == 1:
        return "RIGHT — its very first answer"
    if rank <= 3:
        return f"close — the right answer was its #{rank} choice"
    if rank <= 10:
        return f"wrong — right answer buried at #{rank} of {n_way}"
    return f"WRONG — right answer was #{rank} of {n_way}, near the bottom"


def story(rows: list[dict], metrics: dict, n_way: int = 50) -> list[str]:
    """The whole result, said in sentences. Returns markdown lines."""
    out: list[str] = [
        "# What the model actually answered", "",
        "We hid one piece of a true fact and asked the model to fill it in.",
        "It had to pick from **50 possible answers**. Only one was correct.", "",
        "We ran the same questions three times, changing only one thing each "
        "time: **whether the model could read the real names**.", "",
    ]
    for a in ARMS:
        if any(a in r["arms"] for r in rows):
            out.append(f"- **{PLAIN[a]}** — {WHY[a]}")
    out += ["", "---", ""]

    for i, row in enumerate(rows, 1):
        out += [f"### Question {i}", "",
                f"> {_question(row['head'], row['relation'])}", "",
                f"**The true answer is _{_pretty(row['gold'])}_.**", "",
                "| the model was given | it answered | result |",
                "|---|---|---|"]
        for a, r in row["arms"].items():
            out.append(f"| {PLAIN[a]} | *{_pretty(r['top1'])}* | "
                       f"{_verdict(r['rank'], r.get('n_way', n_way))} |")
        out.append("")

    if metrics:
        out += ["---", "", "## The same thing, over all the questions", ""]
        out += ["| the model was given | got it right first try | "
                "where the right answer usually landed |", "|---|---|---|"]
        for a, m in metrics.items():
            h1 = m.get("hits@1")
            mr = m.get("mean_rank")
            out.append(
                f"| {PLAIN.get(a, a)} | **{h1 * 100:.0f} out of 100** | "
                f"about #{mr:.1f} out of {m.get('n_way', n_way)} |")
        out += ["",
                f"For comparison, **guessing at random** gets it right about "
                f"**2 out of 100**, and the right answer lands around "
                f"#{(n_way + 1) / 2:.0f}.", ""]

    out += ["---", "", "## What this means", "",
            "The model looks very strong **as long as it can read the names**.",
            "",
            "When the names are swapped — same real words, just attached to the "
            "wrong things — it collapses. When the names are hidden completely, "
            "it collapses further.", "",
            "So most of what looked like *knowledge about the world* was really "
            "**recognising a name**. That is fine if you only ever ask about "
            "things it has already seen. It is a problem the moment you ask "
            "about something new — which is the whole point of completing a "
            "knowledge graph.", ""]
    return out


def print_ranking(rows: list[dict], show_top5: bool) -> None:
    for i, row in enumerate(rows, 1):
        print(f"\n{'─' * 78}\n[{i}]  {row['head']}  --  {row['relation']}  --  ?"
              f"\n     gold: {row['gold']}")
        for a, r in row["arms"].items():
            hit = "CORRECT" if r["rank"] == 1 else f"rank {r['rank']:>2}"
            print(f"  {a}  {LABEL[a]:<14} -> {str(r['top1'])[:38]:<40} {hit}")
            if show_top5 and r["top5"]:
                for j, (t, s) in enumerate(r["top5"], 1):
                    mark = "  <- gold" if t == row["gold"] else ""
                    print(f"        {j}. {str(t)[:44]:<46} {s:.4f}{mark}")


# =============================================================================
#  TRIPLE CLASSIFICATION
# =============================================================================
def classify_samples(res: Path, dataset: str, n: int, seed: int) -> list[dict]:
    out = []
    for a in ARMS:
        d = _load(res, f"ch1-{dataset}-{a}-eval.json")
        for side in ("real", "anon"):
            s = d.get(f"samples_{side}")
            if not s:
                continue
            for r in random.Random(seed).sample(s, min(n, len(s))):
                tot = (r["p_yes"] + r["p_no"]) or 1.0
                out.append({"condition": a, "test_set": side, **r,
                            "p_yes_norm": r["p_yes"] / tot})
    if not out:
        raise SystemExit(
            f"no samples_* blocks in results/ch1-{dataset}-*-eval.json.\n"
            f"  Those are written by the CURRENT chapter1/evaluate.py — re-run\n"
            f"  the evaluation to capture them (no retraining needed).")
    return out


def print_classify(rows: list[dict]) -> None:
    for i, r in enumerate(rows, 1):
        verdict = "Yes" if r["predicted"] == 1 else "No"
        truth = "Yes" if r["label"] == 1 else "No"
        ok = "CORRECT" if r["correct"] else "WRONG"
        q = r["prompt"].replace("\n", " ")
        q = q[q.rfind("Is this true:"):] if "Is this true:" in q else q
        print(f"\n[{i}] {r['condition']} on the {r['test_set']} test set")
        print(f"    {q[:96]}")
        print(f"    model answers {verdict!r} (P(Yes)={r['p_yes_norm']:.3f})"
              f"   truth {truth!r}   {ok}")


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="YAGO3-10")
    ap.add_argument("--results", default="results")
    ap.add_argument("--task", default="rank", choices=("rank", "classify", "both"))
    ap.add_argument("--n", type=int, default=8, help="how many random samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top5", action="store_true",
                    help="also print the full 5-best list with scores")
    ap.add_argument("--tags", nargs="+", default=["fixed", "P0"],
                    help="which run tags to read, in priority order. Default "
                         "reads the tuned P0 arms. Use e.g. --tags P6trained to "
                         "compare a trained-on-P6 run. ✋ Tags are NOT mixed "
                         "across arms: every arm must supply the same tag or "
                         "the side-by-side compares different systems.")
    ap.add_argument("--story", action="store_true",
                    help="★ write a PLAIN-ENGLISH report — sentences, not "
                         "metrics. For sharing with people who do not read MRR.")
    ap.add_argument("--out", default=None)
    ns = ap.parse_args()

    res = Path(ns.results)
    saved: dict = {"dataset": ns.dataset, "n": ns.n, "seed": ns.seed}

    if ns.task in ("rank", "both"):
        rows, mets = ranking_samples(res, ns.dataset, ns.n, ns.seed,
                                     tuple(ns.tags))
        print("=" * 78)
        print(f"LINK PREDICTION — {ns.n} random queries, every arm, {ns.dataset}")
        print("=" * 78)
        print_ranking(rows, ns.top5)
        saved["ranking"] = rows
        saved["metrics"] = mets
        if ns.story:
            lines = story(rows, mets)
            md = Path(res / f"ch1_story_{ns.dataset}"
                      + ("" if ns.tags == ["fixed", "P0"]
                         else "_" + "-".join(ns.tags)) + ".md")
            md.write_text("\n".join(lines), encoding="utf-8")
            print("\n" + "\n".join(lines))
            print(f"\nplain-English report -> {md}")

    if ns.task in ("classify", "both"):
        rows = classify_samples(res, ns.dataset, ns.n, ns.seed)
        print("\n" + "=" * 78)
        print(f"TRIPLE CLASSIFICATION — random prompts and the model's answer")
        print("=" * 78)
        print_classify(rows)
        saved["classification"] = rows

    stem = f"ch1_showcase_{ns.dataset}"
    if ns.tags != ["fixed", "P0"]:
        stem += "_" + "-".join(ns.tags)
    out = Path(ns.out or res / f"{stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n\nsaved -> {out}")

    # a LaTeX-ready table, because this is what goes in the paper
    if saved.get("ranking"):
        tex = out.with_suffix(".tex")
        lines = [r"\begin{tabular}{@{}llll@{}}", r"\toprule",
                 r"\textbf{Query} & \textbf{Arm} & \textbf{Model's answer} & "
                 r"\textbf{Rank} \\", r"\midrule"]
        for row in saved["ranking"][:6]:
            q = f"{row['head']} -- {row['relation']} -- ?"
            for j, (a, r) in enumerate(row["arms"].items()):
                lines.append(f"{q if j == 0 else ''} & {a} & "
                             f"{str(r['top1'])[:28]} & {r['rank']} \\\\")
            lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}"]
        tex.write_text("\n".join(lines).replace("_", r"\_"), encoding="utf-8")
        print(f"saved -> {tex}   (paste into the paper)")


if __name__ == "__main__":
    main()
