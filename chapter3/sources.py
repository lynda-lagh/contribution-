"""
★ THE ENRICHMENT SOURCES, SEPARATED — and the relation descriptions, generated.

WHAT WAS WRONG BEFORE
---------------------
`src/data/prompts.py::build_enrichment_extras` produced relation descriptions as

    f"the relation '{txt}' links a subject to an object"

The same sentence for every relation, one word swapped. It carries nothing the
relation name does not already carry, and the numbers show it: L0 -> L1 is the
level that adds this block, and train loss moved **0.050475 -> 0.050223**.

Its own docstring says *"replace with LLM-generated descriptions when running the
full Chapter 3 pipeline"*. This is that replacement.

★★ AND THE GATE IS THE POINT.
   The skeleton calls the generated-content quality gate **N9**, *"the thesis's
   central missing piece"* — no paper in 188 scores LLM-generated enrichment
   before it enters the pipeline. Relations are few (WN18RR: 11) so this is the
   one place the gate can be applied exhaustively rather than by sampling.

    python -m chapter3.sources --dataset WN18RR-ind --generate
    python -m chapter3.sources --dataset WN18RR-ind --gate      # check only
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

# the string the old pipeline emitted — anything matching this failed to generate
TEMPLATE = re.compile(r"links a subject to an object", re.I)

PROMPT = """You are documenting a knowledge graph schema.

Relation: {rel}
Example facts using it:
{examples}

Write ONE sentence (max 25 words) defining this relation. State what kind of
entity is on each side. Do not repeat the relation name as the whole definition.
Definition:"""


# ---------------------------------------------------------------- generation
def relation_examples(kg, rel: str, k: int = 4) -> list[str]:
    out = []
    for t in kg.train:
        if t.relation == rel:
            h = kg.ent2txt.get(t.head, t.head)
            tl = kg.ent2txt.get(t.tail, t.tail)
            out.append(f"  {h} -> {tl}")
            if len(out) >= k:
                break
    return out


def generate(kg, model_name: str, max_new_tokens: int = 48) -> dict[str, str]:
    """One call per relation. WN18RR has 11; this is cheap."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation="sdpa",          # fp16 + eager returns NaN on Qwen2.5
        device_map="cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    out = {}
    for i, (rel, txt) in enumerate(sorted(kg.rel2txt.items()), 1):
        ex = "\n".join(relation_examples(kg, rel)) or "  (none found)"
        msg = PROMPT.format(rel=txt, examples=ex)
        ids = tok(msg, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        text = " ".join(text.strip().split())
        text = text.split("\n")[0].strip().strip('"')
        out[rel] = text
        print(f"  [{i:>3}/{len(kg.rel2txt)}] {txt:32s} {text[:70]}")
    return out


# ---------------------------------------------------------------- ★ the gate
def gate(descriptions: dict[str, str], rel2txt: dict[str, str]) -> dict:
    """
    ★★ N9 IN MINIATURE — score generated content BEFORE it enters the pipeline.

    Five checks, each of which the old template would fail:

      template     is it the placeholder string?
      distinct     do two relations share a description? (the template's defect)
      length       6..40 words — a 3-word definition says nothing, a 60-word one
                   eats the budget this chapter is about
      informative  does it say more than the relation name already does?
      typed        does it name what is on each side? (CATS's latent type
                   constraint, stated in words)

    A description failing `template` or `distinct` is REJECTED, not warned about:
    including it would repeat exactly the bug this file exists to fix.
    """
    counts = Counter(d.strip().lower() for d in descriptions.values())
    rows, rejected = {}, []

    TYPE_WORDS = ("person", "place", "location", "organi", "film", "work",
                  "group", "country", "city", "event", "entity", "concept",
                  "word", "noun", "verb", "category", "member", "part")

    for rel, desc in descriptions.items():
        name = rel2txt.get(rel, rel).lower()
        d = desc.strip()
        w = d.split()
        name_words = {x for x in re.split(r"\W+", name) if len(x) > 2}
        extra = {x.lower() for x in re.split(r"\W+", d) if len(x) > 2} - name_words

        checks = {
            "template": not bool(TEMPLATE.search(d)),
            "distinct": counts[d.lower()] == 1,
            "length": 6 <= len(w) <= 40,
            "informative": len(extra) >= 4,
            "typed": any(t in d.lower() for t in TYPE_WORDS),
        }
        fatal = not (checks["template"] and checks["distinct"])
        rows[rel] = {"text": d, "n_words": len(w), "checks": checks,
                     "passed": all(checks.values()), "rejected": fatal}
        if fatal:
            rejected.append(rel)

    n = len(rows) or 1
    report = {
        "n": len(rows),
        "n_passed": sum(1 for r in rows.values() if r["passed"]),
        "n_rejected": len(rejected),
        "rejected": rejected,
        "rates": {k: sum(1 for r in rows.values() if r["checks"][k]) / n
                  for k in ("template", "distinct", "length", "informative", "typed")},
        "per_relation": rows,
    }
    return report


def show_gate(rep: dict, rel2txt: dict[str, str]) -> None:
    print(f"\n{'='*78}\n★ QUALITY GATE ON GENERATED RELATION DESCRIPTIONS (N9)\n{'='*78}")
    print(f"{'relation':30s} {'w':>3s}  tmpl dist len info type   verdict")
    for rel, r in sorted(rep["per_relation"].items()):
        c = r["checks"]
        m = lambda b: " ✓ " if b else " ✗ "
        verdict = ("REJECTED" if r["rejected"] else
                   "pass" if r["passed"] else "weak")
        print(f"{rel2txt.get(rel, rel)[:30]:30s} {r['n_words']:>3d} "
              f"{m(c['template'])}{m(c['distinct'])}{m(c['length'])}"
              f"{m(c['informative'])}{m(c['typed'])}  {verdict}")
    print(f"\n  passed {rep['n_passed']}/{rep['n']}   rejected {rep['n_rejected']}")
    for k, v in rep["rates"].items():
        print(f"    {k:12s} {v:6.1%}")
    if rep["n_rejected"]:
        print("\n  ✋ Rejected descriptions must NOT enter the pipeline — that is the")
        print("     bug this file exists to fix. Regenerate them or write them by hand.")
    print("\n  ★ Report this table. No paper in the corpus scores its generated")
    print("    enrichment before using it; that gap is N9 in the skeleton.")


# ------------------------------------------------------- blocks for the budget
class GraphIndex:
    """
    ★★ BUILT ONCE. Two bugs live in the code this replaces, and both are fatal
       in different ways.

    ⚠️ BUG 1 — THE INDUCTIVE SETTING HAD NO NEIGHBOURS AT ALL.
       The old `candidate_blocks` collected neighbours from `kg.train` only.
       In the inductive setting test entities are UNSEEN BY DEFINITION, so every
       test entity had exactly zero neighbours and the `neighbours` block was
       never emitted for any query.

       That block is the one S1, S2, S4 and S5 all discriminate on. With it
       missing, most of the ladder had nothing left to decide and collapsed onto
       the baseline — measured: S2 produced byte-identical prompts to S0.
       The chapter would have reported "specificity does not pay" when what it
       actually measured was "there was nothing to allocate".

       ★ THE FIX: an unseen entity's neighbours come from the INFERENCE GRAPH —
       the other facts observable about it at test time. This is what CATS
       supplies as σ neighbouring facts and RealKGC as structure blocks. An
       inductive model is given the new entity's local graph; that is the whole
       premise of inductive KGC.

    ⚠️ BUG 2 — THE 468x PERFORMANCE BUG, REINTRODUCED.
       The old version rebuilt the neighbour, domain and range maps from the
       ENTIRE training graph on every call. Chapter 1 hit exactly this in
       `make_negatives` and it turned a 1.5-hour run into a 9-hour one.
       At 2,000 queries x 8 policies x 5 budgets x 2 directions = 160,000 calls
       over WN18RR's 86k training triples, prompt-building alone would take
       hours of pure CPU before a single forward pass.

    ⚠️ AND THE LEAK GUARD THAT COMES WITH THE FIX.
       Once test-graph facts are usable as context, the query triple itself
       becomes reachable — showing `(h, r, t)` while asking `(h, r, ?)` hands
       over the answer. `neighbours_of` takes the query triple and excludes it,
       in BOTH orientations. `assert_no_leak` re-checks it independently.
    """

    def __init__(self, kg, use_inference_graph: bool = True):
        self.kg = kg
        self.use_inference_graph = use_inference_graph
        self.nbrs: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self.by_relation: dict[str, list] = defaultdict(list)
        self._range: dict[str, list[str]] = defaultdict(list)

        # the observable graph: training facts always, plus the inference graph
        # in the inductive setting (that is what makes the setting solvable)
        support = list(kg.train)
        n_sup = 0
        if use_inference_graph:
            seen = {e for t in kg.train for e in (t.head, t.tail)}
            # ★ valid FIRST. When the dataset is CATS-derived, valid.tsv holds
            #   `inductive_graph.txt` — the facts about unseen entities that make
            #   the setting solvable at all. Omitting it leaves every test entity
            #   with no observable context, which is the exact failure that once
            #   made this pipeline report a false null result.
            for t in (*kg.valid, *kg.test):
                if t.label is not None and t.label < 0:
                    continue                       # false triples are not evidence
                if t.head not in seen or t.tail not in seen:
                    support.append(t)              # a fact about an unseen entity
                    n_sup += 1
        self.n_support = n_sup

        for t in support:
            # (relation, other_entity, direction) — direction keeps the phrasing right
            self.nbrs[t.head].append((t.relation, t.tail, "out"))
            self.nbrs[t.tail].append((t.relation, t.head, "in"))
            self._range[t.relation].append(t.tail)
        for t in kg.train:                         # demonstrations from TRAIN only
            self.by_relation[t.relation].append(t)

        self._ent_cache: dict[str, float] = {}

    def type_entropy(self, relation: str, types: dict[str, str]) -> float:
        """Entropy of the relation's range over entity types. S2 allocates on it."""
        if relation in self._ent_cache:
            return self._ent_cache[relation]
        c = Counter(types.get(e, "OTHER") for e in self._range.get(relation, ()))
        tot = sum(c.values()) or 1
        ent = -sum((n / tot) * math.log2(n / tot) for n in c.values() if n)
        self._ent_cache[relation] = ent
        return ent

    def neighbours_of(self, entity: str, exclude: tuple | None, k: int):
        """
        Up to `k` neighbouring facts, with the query triple removed.

        `exclude` is the (head, relation, tail) being asked about. It is checked
        in both orientations because the index stores each edge twice.
        """
        out = []
        for r, other, d in self.nbrs.get(entity, ()):
            if exclude is not None:
                trip = (entity, r, other) if d == "out" else (other, r, entity)
                if trip == exclude:
                    continue                       # ★ never show the answer
            out.append((r, other, d))
            if len(out) >= k:
                break
        return out


def candidate_blocks(kg, head: str, relation: str, rel_desc: dict[str, str],
                     types: dict[str, str], count, n_neighbours: int = 5,
                     n_demos: int = 3, index: GraphIndex | None = None,
                     exclude: tuple | None = None):
    """
    Everything that COULD occupy budget for one query, with the features
    `policies.py` keys on attached to each block.

    ⚠️ Every block carries `tokens` measured with the REAL counter. The whole
       method depends on the budget being real; an estimated cost here would
       reintroduce exactly the flaw that made the first run uninterpretable.

    `index` should be a GraphIndex built ONCE by the caller. Building one per
    call is the performance bug documented on GraphIndex.
    `exclude` is the query triple, withheld from the neighbour block.
    """
    from .budget import Block

    if index is None:
        index = GraphIndex(kg)                     # correct, but slow if repeated

    ent = index.type_entropy(relation, types)
    desc = kg.ent2txt.get(head, "")
    meta = {
        "has_description": len(desc.split()) >= 3,
        "label_words": len(desc.split()),
        "type_entropy": ent,
        "degree": len(index.nbrs.get(head, ())),   # S5's profiler checks this
    }

    out = []
    if types.get(head):
        txt = f"[{types[head]}]"
        out.append(Block("type_tag", head, txt, count(txt), dict(meta)))
    if desc:
        out.append(Block("entity_description", head, desc, count(desc), dict(meta)))
    if rel_desc.get(relation):
        txt = rel_desc[relation]
        out.append(Block("relation_description", relation, txt, count(txt), dict(meta)))

    nb = index.neighbours_of(head, exclude, n_neighbours)
    if nb:
        parts = []
        for r, e, d in nb:
            rt = kg.rel2txt.get(r, r)
            et = kg.ent2txt.get(e, e)
            parts.append(f"{rt} {et}" if d == "out" else f"is {rt} of {et}")
        txt = "; ".join(parts)
        out.append(Block("neighbours", head, txt, count(txt), dict(meta)))

    # ★ DEMONSTRATIONS — added after P30 (KICGPTv2), whose Knowledge Prompt
    #   supplies other triples as in-context examples. CATS uses k=3 supporting
    #   triples sharing the query relation; RealKGC shows triples sharing r_q.
    #
    #   This is the most expensive block by a wide margin, and plausibly the most
    #   valuable for exactly the long-tail elements that have nothing else. That
    #   is what makes the budget decision interesting rather than obvious: at
    #   B=30 one demonstration can consume the entire budget.
    #
    #   ⚠️ Drawn from TRAIN only. A demonstration taken from the test graph could
    #      be the query triple itself.
    demos = index.by_relation.get(relation, [])[:n_demos]
    if demos:
        txt = "; ".join(f"{kg.ent2txt.get(t.head, t.head)} {kg.rel2txt.get(relation, relation)} "
                        f"{kg.ent2txt.get(t.tail, t.tail)}" for t in demos)
        txt = f"Other triples using this relation: {txt}."
        out.append(Block("demonstrations", f"{head}|{relation}", txt,
                         count(txt), dict(meta)))
    return out


def assert_no_leak(blocks, kg, head: str, relation: str, gold: str) -> None:
    """
    ★ An INDEPENDENT check that the gold answer is not in the context.

    `neighbours_of` already excludes the query triple, but the guard that
    matters is the one that does not trust the code it is guarding. Called by
    data.py on a sample of queries; raises rather than warns, because a leak
    makes every number in the chapter meaningless.
    """
    gold_txt = kg.ent2txt.get(gold, gold)
    rel_txt = kg.rel2txt.get(relation, relation)
    for b in blocks:
        if b.kind != "neighbours":
            continue
        for seg in b.text.split(";"):
            s = seg.strip()
            if s.startswith(f"{rel_txt} ") and s[len(rel_txt) + 1:].strip() == gold_txt:
                raise AssertionError(
                    f"★✋ CONTEXT LEAK: the neighbour block for {head} contains "
                    f"'{rel_txt} {gold_txt}', which IS the answer to "
                    f"({head}, {relation}, ?). Every result computed with this "
                    f"context is invalid.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="WN18RR-ind")
    ap.add_argument("--root", default="data")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--generate", action="store_true", help="run the LLM (needs GPU)")
    ap.add_argument("--gate", action="store_true", help="score existing descriptions")
    ap.add_argument("--out", default=None)
    ns = ap.parse_args()

    from src.data.loaders import load_kg
    from src.utils.config import load_config

    cfg = load_config(ns.config)
    kg = load_kg(ns.dataset, ns.root)
    dest = Path(ns.out or Path(ns.root, ns.dataset, "relation_descriptions.json"))

    print(f"[src] {ns.dataset}: {len(kg.rel2txt)} relations")

    if ns.generate:
        desc = generate(kg, cfg["model"]["name"])
        dest.write_text(json.dumps(desc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {dest}")
    elif dest.exists():
        desc = json.loads(dest.read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"{dest} not found — run with --generate first")

    rep = gate(desc, kg.rel2txt)
    show_gate(rep, kg.rel2txt)
    gp = dest.with_name(dest.stem + "_gate.json")
    gp.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {gp}")
    if rep["n_rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
