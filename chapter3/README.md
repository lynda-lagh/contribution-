The hierarchy source is the decisive detail. Let me get the rest.Read in full. **The idea survives, and one detail is what saves it.**

> *"injects hierarchical knowledge **(derived via clustering)** into the codebook"*
> *"we perform **agglomerative clustering** on entity semantic representations to generate a hierarchy tree"*

**GS-Quant's hierarchy is learned from embeddings. It is not WordNet, not a taxonomy, not any external resource.** And it structures **discrete representation codes**, never prompt content.

| | GS-Quant | your S5 |
|---|---|---|
| granularity from | agglomerative clustering of embeddings | WordNet depth, IDF, polysemy |
| structures | codes in a codebook | which blocks occupy a token budget |
| purpose | modality alignment | **cost** |
| level | representation | prompt |
| cost measured | ❌ | the whole point |

Shared intuition, different source, object and purpose. That's defensible — but it needs to be **one explicit sentence in the paper**, or a reviewer who knows GS-Quant says the idea is taken.

**It also hands you a good argument you don't have to make yourself.** Their motivating example: *for `(Christmas, month, ?)`, the model should identify the coarse category "month" before narrowing to "December"*. That's a published case for specificity as a real axis of KGC reasoning, from a paper that beats SOTA (+1.7% MRR on WN18RR, +1.6% on FB15k-237). Cite it rather than asserting the intuition.

**⚠️ And it gives you a competing explanation to rule out.** If clustering embeddings already recovers usable granularity, WordNet depth may add nothing beyond what the model's own representations encode. Cheap check: correlate WordNet hypernym depth against agglomerative-cluster depth on the same entities. If they agree closely, the "linguistic ground truth" framing weakens and the honest claim becomes *"a free external signal reproduces what clustering learns, at zero training cost"* — still worth something, but a different claim.

That now makes **three pre-checks** before building `S5_semantic`, all free and all answering whether the feature is real:

```
depth vs log-degree      -> is it long-tail in disguise?      (KICGPTv2 owns that)
depth distribution       -> does it have variance?            (the 95.7% band mistake)
depth vs cluster depth   -> does WordNet beat clustering?     (GS-Quant owns that)
```

Matrix is at **31 rows**, note written, skeleton's Stage-3 row updated to record that GS-Quant's codes are hierarchical and where that hierarchy comes from.

Want me to write that three-way profiler? It's an hour of CPU and it decides whether `S5_semantic` is worth building.# Chapter 3 — Where should the context budget go?

> Prior work asks **how much** context helps. We fix the amount and ask **where
> it should go.**

> ★★ **Updated after reading P28–P31.** The claim is now narrower and better
> supported: relation-level tailoring is already published (**P29**), context
> selection already exists (**P30**, KRST, APST, RealKGC), and coarse-to-fine
> granularity is already used on representation codes (**P31**). What remains
> unoccupied is the **budget constraint** — and **P28** supplies a mechanism
> saying withholding context can *raise* accuracy, not merely save tokens.

---

# 0 · The question, and why it is new

Seven papers in the corpus report a version of *more context is not better*
(P02, P08, P11, P12, P19, P20, P21). **MKGL** prices context at 91.4 against
811.2 average input tokens and is the only paper to publish a cost table.
**CATS** supplies δ=6 reasoning paths, σ=6 neighbouring facts and k=3 supporting
triples; **RealKGC** supplies structure, type and background-fact blocks;
**GS-KGC** supplies negatives and neighbours.

Every one of them gives **every element the same treatment**.

So the field has studied the *quantity* axis and left the *specificity* axis
untouched. That is this chapter:

> At a **fixed** context budget, does it matter **how specifically** the context
> is matched to the element being completed?

---

# 1 · Why inductive, and why that makes the chapter independent

| | transductive | **inductive** |
|---|---|---|
| test entities | seen during tuning | **unseen** |
| what the model can lean on | the entity name | **only what is in the prompt** |
| does allocation matter? | often not | **it is the whole input** |

Transductively a model can answer by recognising the entity, so context competes
with a shortcut that already works and a flat ladder is the expected outcome.
Inductively there is no learned representation of the test entity at all.

★ **This justification follows from the definition of the inductive setting and
needs nothing from Chapter 1.** Chapter 3 stands alone.

⚠️ **FB15k-237 is primary; WN18RR validates the pipeline.** P28 reports that
WN18RR *"has only 11 relations, so the model has seen almost all combinations"* —
long-tail effects are muted there, and our policies are motivated by thin
elements. WN18RR keeps its role because 11 relations make the description
generation and its quality gate exhaustively checkable.

**Splits: CATS's, which RealKGC also adopts** — *"the specific dataset versions
and splits as processed in CATS"*. Both of the closest papers become directly
comparable, and split construction is not something a reviewer can question.
CATS also ablates **Qwen2-1.5B**, so our model size is size-matched by their
choice rather than by our budget.

---

# 2 · ★★ The method: a budget that is actually enforced

The first Chapter 3 run could not answer its own question:

```
L0  17,866,755 tokens   no routing          }  quantity AND specificity
L3   2,977,580 tokens   per-quality-band    }  changed together
```

Two variables moved at once, so no result was attributable to either.

**Here every policy gets the same budget and differs only in priority order.**
Any MRR difference is specificity and nothing else.

⚠️ **Counted in tokens, never in blocks.** A typed-neighbour list is ~60 tokens
and a type tag ~8; "two blocks each" would let a policy smuggle in seven times
the context while appearing matched. `budget.py::allocate` enforces this, and
oversized blocks are **truncated on a word boundary** rather than dropped, so a
policy is never silently starved.

---

# 3 · The grid

## Policies — same budget, different priorities

| id | policy | decides on | level |
|---|---|---|---|
| **S0** | uniform | nothing — the baseline | — |
| **R** | **random** | nothing, shuffled | ★ **the control** |
| **S1** | entity property | does a description exist? | L1 |
| **S2** | semantic type | type implied by the relation | L2 |
| **S3** | label quality | how informative the label is | L3 |
| **S4** | instance | per (head, relation) query | L4 |
| **S5** | ★ **semantic specificity** | *how specific* the label's meaning is — depth, polysemy, IDF | L5 ⚠️ **gated** |
| **ORACLE** | oracle | the gold answer | ★ **the ceiling** |

**Budgets:** 0 · 30 · 60 · 120 · 240 tokens. `B=0` is the floor — `(h, r, ?)`
and nothing else.

## ★ The two rows that make the rest interpretable

**R (random)** is Chapter 1's condition S transplanted. Same budget, same action
mix, decisions shuffled. If `S4 ≈ R` the *decisions* add nothing and only the
budget matters — a clean negative result. Without R, `S4 ≈ S0` is ambiguous
between *specificity does not pay* and *our policy is bad*, and a reviewer will
say exactly that.

**ORACLE** allocates using the gold answer. Infeasible in deployment; its only
job is to bound what any policy could achieve. **Run it first:** if
`ORACLE ≈ S0` at a budget, nothing can win there and the twelve policy runs
would be measuring noise. One cheap run can save a week.

## Pre-registered interpretations

Written before any result exists — `policies.py::INTERPRETATION`.

| outcome | conclusion |
|---|---|
| `ORACLE ≈ S0` | ★★ stop. Uniform is already optimal at this budget |
| `S4 > R` and `S4 > S0` | ★★ **specificity pays at fixed cost** — the headline |
| `S4 ≈ R` | ★ the decisions add nothing; the specificity analogue of *more context is not better* |
| `S4 > S0` but `S4 ≈ R` | ⚠️ the gain is the action **mix**, not targeting. Report the mix, drop the specificity claim |
| `S2 ≈ S4` | specificity saturates at type level — tells the next person where to stop |
| curves converge as B grows | specificity matters **only under a tight budget** — precise and deployable |
| `S3 ≈ S1` | on this graph label *quality* and label *presence* are one feature. A finding about the graph |

---

# 3b · ★★ What the literature fixed, and what it forbids

Four papers read in full for this chapter. Notes: `1_corpus/notes/P28…P31`.

| paper | what it establishes | what it does to our claim |
|---|---|---|
| **P28 ToC** (TKDE 37(10) 2025) | unsupportive context is learned as *"a feature supporting the existence of the query relation"* | ★★ **raises the target**: withholding may *improve* MRR, not just save tokens |
| **P29 struOKGC** (WWW 28:8 2025) | relation-specific templates with `T_h` / `T_t` slots beat SOTA | ★ **the S2 rung is already published** — we measure where the ladder *stops* |
| **P30 KICGPTv2** (TKDE 38(7) 2026) | long-tail entities have *"scant structural information"*; Knowledge Prompt **selects** context | ⚠️ **forbids the word "select"**; supplies the premise and a 6th source |
| **P31 GS-Quant** | coarse-to-fine granularity, hierarchy **learned by clustering**, applied to **codes** | ⚠️ **bounds `S5_semantic`** — different source, object and cost model |

## The claim, as it must now be worded

> Prior work **selects** context by relevance and **tailors** it by relation,
> always unconditionally and unpriced. We hold the **token budget fixed** and ask
> whether the **granularity of the allocation rule** changes ranking quality **at
> matched cost**.

⚠️ *"We select context"* is unavailable — P30, KRST, APST and RealKGC's RSD all do.
The novelty is the **constraint**, not the selection.

## What is verifiably empty

Across 188 extracted PDFs:

| | hits |
|---|---|
| *"when to retrieve"* / *"whether to retrieve"* | **0** |
| per-query compute budget / input-adaptive computation | **0** |
| prompt or context **compression** | 6 (incidental) |
| accuracy-vs-cost **Pareto** curve | **1** |
| WordNet **information content** (Resnik / Lin / Jiang–Conrath) | **1** |

Only **P20/MKGL** prices context at all (91.4 vs 811.2 input tokens, plus a
GPU-hour table).

---

# 4 · Provenance — every design choice, its closest paper, and why we differ

## 4.1 Fixed budget

| | |
|---|---|
| **Closest** | **MKGL** (NeurIPS 2024) — 91.4 vs 811.2 input tokens, plus a GPU-hour cost table |
| **They did** | Priced their own method against baselines. Context quantity is an outcome, not a controlled variable |
| **Why us** | We make it the control. Holding it fixed is what isolates specificity |

## 4.2 The random control

| | |
|---|---|
| **Closest** | nothing in the corpus |
| **Why us** | ★ Chapter 1 became defensible because condition S could falsify it. An allocation study without a random baseline cannot distinguish a working policy from a lucky action mix |

## 4.3 Routing on description presence, not quality bands

| | |
|---|---|
| **We do** | Split on `has_description` |
| **The first run did** | Split on quality bands: **moderate 95.7%**, rich 3.9%, poor 0.4% |
| **Why us** | One bucket covering 96% of elements gives the policy nothing to decide — the ladder came out flat **by construction, not by finding**. `has_description_rate = 7.9%` is a genuine 8/92 split |

## 4.4 Type-aware allocation

| | |
|---|---|
| **Closest** | **CATS** — *"relations impose latent type constraints… `works in` typically connects a person and a location"* |
| **They did** | Supply the type to every query and report accuracy up |
| **Why us** | Where the relation's range is near-deterministic a tag is cheap and sufficient; where it is diffuse the tag is weak evidence and neighbours are worth more. We allocate on that entropy instead of always paying for the tag |

## 4.5 Relation descriptions

| | |
|---|---|
| **Closest** | **ColKGC**, **RelSemEnh** — generating relation descriptions pays; rewriting entity descriptions gives ~0 |
| ⚠️ **The first run did** | `f"the relation '{txt}' links a subject to an object"` — the same sentence 37 times, one word swapped |
| **Evidence it was empty** | L0 → L1 is the level that adds this block. Train loss moved **0.050475 → 0.050223** |
| **Why us** | Generate them with an LLM **and quality-gate the output**. That gate is N9 in the skeleton — *"the thesis's central missing piece"*, since no paper scores generated enrichment before it enters the pipeline |

## 4.6 Faithfulness of the allocation

| | |
|---|---|
| **Closest** | **Jacovi & Goldberg** (ACL 2020) on faithfulness; **P08/TraceVal** on traceable justification |
| **Why us** | Every allocation states a reason, and we measure what fraction of those reasons are *operative* — changing the named feature changes the decision. First-run result: L1/L2 **100% faithful**, L3/L4 **8.3%**. Sophistication bought worse explanations, which is the *explicable* half of the thesis title, measured |

---

# 5 · What "good results" means here

Be honest about the target. Allocation is a **cost** method; it usually does not
beat uniform enrichment on accuracy. The publishable claim is a Pareto point:

```
uniform-all     MRR 0.41   1.00x tokens
routed          MRR 0.40   0.35x tokens     <- the contribution
uniform-cheap   MRR 0.33   0.35x tokens     <- the row that proves it
```

The third row is what makes the second mean anything: **at the same budget**,
does allocating beat spending uniformly?

⚠️ Chasing *routed beats uniform-all on MRR* will probably fail and does not need
to succeed. Equal ranking quality at a third of the cost is the result.

---

# 6 · Files

| file | purpose | GPU |
|---|---|---|
| `budget.py` | ★ the method — token budget, truncation, greedy allocation, per-decision reasons | — |
| `policies.py` | ★ the grid: 8 policies, 5 budgets, pre-registered outcomes, sanity rules | — |
| `validate.py` | ★★ **hard checks on the inductive split, non-zero exit** | — |
| `sources.py` | the enrichment sources + **`GraphIndex`** (inference-graph neighbours, leak guard) + relation descriptions **quality gated = N9** | — |
| `profile_specificity.py` | ★★ **the three checks that gate `S5_semantic`** | — |
| `candidates.py` | ★★ **freeze the ranking negatives to disk** — matched comparison by construction | — |
| `stats.py` | ★★ **the paired bootstrap** — what a difference must be before it is a result | — |
| `data.py` | build prompts per (policy, budget, **direction**) | — |
| `evaluate.py` | link prediction both directions + **relation prediction w/ confusion matrix** | ✓ |
| `report.py` | seven views, every verdict backed by a paired test | — |
| `qualitative.py` | ★ **same query, two policies, side by side** + LaTeX | — |
| `test_chapter3.py` | 27 unit tests, a different random inductive graph each run | — |
| `test_pipeline.py` | ★★ **49 end-to-end checks on a synthetic graph, ~2 s, no GPU** | — |

Reused unchanged: `src/routing/faithfulness.py`, `src/routing/features.py`.

---

# 6b · ★★ Six corrections, and two bugs found while making them

Everything below changes what the chapter is *allowed to conclude*. Each is
pinned by a test in `test_pipeline.py`.

| # | was | now | why it matters |
|---|---|---|---|
| 1 | `top1_classification` incremented `fp` **and** `fn` on the same class | real confusion matrix via `--task relation` | with `fp == fn`, precision == recall == F1 == Hits@1. The "macro-F1" was Hits@1 renamed, and no matrix existed because a two-outcome event has nothing to confuse |
| 2 | verdicts declared at **±0.005 MRR** | paired bootstrap; CI must exclude 0 | SE at n=300 is ≈0.02, so **every** old verdict could flip on the seed |
| 3 | negatives sampled inline from a running RNG | **frozen to disk**, per-query seed | reproducible only while seed *and query count* matched; change `--limit` and two policies silently ranked against different negatives |
| 4 | filtered against train ∪ test | train ∪ **valid** ∪ test | true facts living in valid were being scored as negatives |
| 5 | tail direction only | tail **and** head | CATS and RealKGC report both |
| 6 | tuned only | **untuned row** | if allocation pays untuned, the claim is about context, not our training recipe |

## ★★ And the two bugs the fixes exposed

**A · Inductive queries had no neighbours at all.** `candidate_blocks` read
neighbours from `kg.train` only — but inductive test entities are unseen *by
definition*, so the block that S1, S2, S4 and S5 all discriminate on was never
emitted for any test query. Measured on the synthetic fixture: **0/60 queries had
a neighbours block**, and `S2_type` produced byte-identical prompts to
`S0_uniform`.

> The chapter would have reported *"specificity does not pay"* when what it
> actually measured was *"there was nothing to allocate"*.

The fix gives unseen entities their **inference graph** — the other facts
observable about them at test time, which is what CATS supplies as σ neighbouring
facts and what makes inductive KGC solvable at all. With it: **54/60**.

⚠️ That fix makes the gold answer reachable, so `neighbours_of` excludes the query
triple in both orientations and `assert_no_leak` re-checks it independently. The
test proves the guard is **not vacuous** by feeding it an unexcluded query.

**B · The 468× bug from Chapter 1, reintroduced.** The neighbour/domain/range
maps were rebuilt from the entire training graph on every call. At
2,000 queries × 8 policies × 5 budgets × 2 directions = **160,000 calls** over
WN18RR's 86k training triples, prompt-building alone would have cost hours of CPU
before a single forward pass. `GraphIndex` is now built once and passed down.

---

# 7 · Run order

```bash
# ── free, no GPU ───────────────────────────────────────────────────────────
python -m chapter3.test_pipeline                 # ★ 49 end-to-end checks, ~2 s
python -m chapter3.test_chapter3 --repeat 3      # 27 unit tests
python -m chapter3.stats --demo                  # ★ validate the estimator itself
python -m chapter3.validate --dataset WN18RR-ind # ✋ gates everything
python -m chapter3.profile_specificity --dataset WN18RR-ind   # gates S5

# ── ★ FREEZE THE NEGATIVES BEFORE ANY EVALUATION ───────────────────────────
python -m chapter3.candidates --dataset WN18RR-ind --direction both --n-way 50
python -m chapter3.data --dataset WN18RR-ind --all --budget 30 60 120 240 \
       --direction both --limit 300
python -m chapter3.candidates --dataset WN18RR-ind --direction both --verify

# ── the two anchors, BEFORE the twelve policy runs ─────────────────────────
#    B=0 floor  and  ORACLE ceiling
#    if ORACLE is NOT distinguishable from S0, stop and report that

# ── the main curve ─────────────────────────────────────────────────────────
#    4 budgets x {S0, R, S4}          = 12 runs
#    then S1, S2, S3 (+S5 if gated in) at ONE budget
#    then: untuned rows · head direction · --task relation

python -m chapter3.report --dataset WN18RR-ind --compare-untuned --both-directions
python -m chapter3.qualitative --dataset WN18RR-ind --budget 120 \
       --a S0_uniform --b S4_instance --only-disagreements --latex
```

⚠️ **`validate.py` gates everything.** Chapter 3's premise is that test entities
are unseen. A split with even a few percent leakage produces slightly better
numbers and an indefensible chapter — it does not fail loudly.

⚠️ **`chapter3.data` prints two guards. Read them.**
`N/200 sampled queries have a neighbours block` — if this is **0**, stop: bug A
above has recurred and the ladder cannot differ from the baseline.
`[guard] B=120 tail: 7/7 policies produce DISTINCT prompts ✓` — if two policies
are byte-identical, that cell measures nothing.

---

# 8 · What we do NOT claim

| | why not |
|---|---|
| "Allocation beats more context" | It is a cost method. The claim is matched quality at lower cost |
| "The router learned a policy" | The policies are **hand-written rules over measured features**. Learning the policy is future work |
| "This transfers to transductive KGC" | It should not — there the name is a shortcut. That is a prediction, not a result |
| "Faithful explanations mean correct decisions" | Faithfulness is whether the stated reason is *operative*, not whether it is *wise* |
| "Policy A and policy B perform equivalently" | ★ A CI containing 0 means the effect was **not measurable**, not that it is absent. `report.py` prints the minimum detectable effect so the bound can be stated honestly |
| "Hits@1 = 0.4" without qualification | ★ it is **50-way** Hits@1, not full-ranking. Every caption must say so |
| "macro-F1 on link prediction" | ★ that number is per-relation Hits@1. Real F1 exists only under `--task relation` |

---

# 9 · The positioning sentence

> Recent LLM-based knowledge graph completion systems supply substantial context
> and report that beyond a point more context does not help. That finding is
> about quantity. We ask whether, at a **fixed** context budget, it matters **how
> specifically** the context is matched to the element being completed. We define
> four levels of specificity, enforce an identical token budget across all of
> them, and evaluate on inductive link prediction, where unseen test entities
> make prompt context the only available signal. A random-allocation control
> separates the value of the decisions from the value of the budget, and an
> oracle bounds what any policy could achieve. Every allocation decision carries
> a stated reason, and we measure what fraction of those reasons are operative.
