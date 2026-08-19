# Chapter 3, standing on its own

The question you asked: *how do I make this an independent contribution with good
results, as link prediction, and should it be inductive?*

---

## 1 · ★★★ Inductive. And not as a preference — it is what makes the chapter possible

Selective enrichment routing only means something if enrichment means something.
Consider the two settings:

| | transductive | **inductive** |
|---|---|---|
| test entities | seen in training | **unseen** |
| what the model can lean on | the entity name, learned during tuning | nothing about the entity except **what you put in the prompt** |
| does enrichment matter? | often not | **it is the only signal there is** |
| is routing it a real question? | no — you are allocating something worthless | **yes** |

In the transductive setting a model can answer by recognising the entity. Every
description, type and neighbour you add is competing with a shortcut that already
works, so a flat enrichment ladder is the expected outcome and routing has nothing
to allocate.

In the inductive setting the test entity was never seen. There is no learned
representation to fall back on. **Context is not an enhancement; it is the entire
input.** That is what turns "which context is worth its tokens?" into a question
with an answer.

★ **And this justification needs nothing from Chapter 1.** It follows from the
definition of the inductive setting. Chapter 3 becomes independent the moment it
moves.

★ It also puts you in the same setting as **CATS** (AAAI 2025) and **RealKGC**
(KBS 2026), which gives you baselines, released splits, and a protocol —
50-way filtered ranking, which `chapter1/rank.py` already implements.

⚠️ Use **CATS's released splits** rather than building your own. RealKGC adopts
them explicitly ("the specific dataset versions and splits as processed in
CATS"), so both papers become directly comparable, and split construction stops
being a thing a reviewer can question.

---

## 2 · ★★ The structural problem: L0–L4 confounds two different axes

Right now the ladder moves two things at once:

```
L0  none              L1  entity/relation      L2  semantic type
L3  label quality     L4  instance
```

L1 and L2 change **what content** goes in the prompt. L3 and L4 change **how
finely the decision is made**. Those are independent, and mixing them means no
effect can be attributed. It is why the ladder looks flat: L3's extra granularity
is being measured on top of L2's extra content.

**Separate them.**

| | axis A — *what* | axis B — *how finely decided* |
|---|---|---|
| values | description · relation description · type · neighbours · exclusions | uniform · per-relation · per-entity |
| question | what is each context type worth per token? | does deciding per element beat one global policy? |

Axis A is a content ablation. Axis B is the routing claim. **Axis A must come
first**, because a router with no idea what each action is worth has no objective
function.

---

## 3 · ★★ The experiment you are missing, and it is the one that makes the chapter

**What is each enrichment type worth, per token?**

A leave-one-out ablation over the four content types `build_enrichment_extras`
already produces:

| condition | prompt contains | MRR | tokens/prompt | MRR per 1k tokens |
|---|---|---|---|---|
| bare | h, r, t only | | | |
| +description | | | | |
| +relation description | | | | |
| +type | | | | |
| +neighbours | | | | |
| +exclusions | | | | |
| all | | | | |
| all − neighbours | | | | |
| all − type | | | | |
| … | | | | |

That table is a contribution on its own, before any routing. **Nobody in your
corpus has it.** CATS supplies δ=6 reasoning paths, σ=6 neighbouring facts and
k=3 supporting triples; RealKGC supplies structure, type and background-fact
blocks; GS-KGC supplies negatives and neighbours. All of them report accuracy
with everything switched on. None reports what each block costs or buys.

★ The last column is the whole chapter in one number. If neighbours cost 60
tokens and buy 0.01 MRR while types cost 8 tokens and buy 0.03, the routing
policy writes itself — and you can defend it.

---

## 4 · ★★ Route on the feature that actually has variance

The current router keys on quality bands: **moderate 95.7%, rich 3.9%, poor
0.4%**. Ninety-six percent of elements land in one bucket, so nearly every
element gets the same action at every level. The ladder is flat by construction.

The same feature report contains a far better signal:

```
has_description_rate  7.9%
```

**8% of entities have a description; 92% do not.** That is a genuine split with
an obvious, defensible decision rule:

- entity **has** a description → it may not need neighbours
- entity **lacks** one → a name alone is nothing; give it typed neighbours

And in the inductive setting this is exactly the decision that matters, because
the unseen entity's description is the only thing standing between the model and
a bare identifier.

**Second candidate: relation.** 37 relations, EIR 19,672, wildly different
semantics. `hasGender` has two possible tails and needs no enrichment at all;
`isLocatedIn` plausibly needs neighbours. Thirty-seven real decisions instead of
one bucket covering 96% of the graph.

---

## 5 · What "good results" realistically looks like

Be honest with yourself about the target. Routing usually does **not** beat
uniform enrichment on accuracy — it is a *cost* method. The publishable claim is
a Pareto point:

> **Matched MRR at a fraction of the tokens**, with a per-decision explanation
> whose faithfulness is measured.

Something of this shape is a genuine result:

```
uniform-all     MRR 0.41   1.00x tokens
routed          MRR 0.40   0.35x tokens     <- the contribution
uniform-cheap   MRR 0.33   0.35x tokens     <- the control that proves it
```

The third row is what makes the second meaningful: at the **same budget**, does
routing beat spending uniformly? If routed ≈ uniform-cheap, the router adds
nothing and only the budget matters. If routed ≫ uniform-cheap, the decisions
are doing work.

⚠️ Chasing "routed beats uniform-all on MRR" will probably fail and does not need
to succeed. Equal accuracy at a third of the cost is the contribution.

---

## 6 · The control, borrowed from Chapter 1's condition S

**Random routing at a matched action mix.** Same skip rate, same distribution of
actions, assigned at random rather than by policy.

| | reading |
|---|---|
| random ≈ learned | the decisions add nothing; only the budget matters. Clean, reportable |
| learned > random | the policy is doing real work, and this quantifies how much |

One run. Without it, every routing result is ambiguous between *routing does not
help* and *this router is bad* — and a reviewer will raise exactly that.

---

## 7 · The cost axis, with a precedent to cite

**MKGL** (NeurIPS 2024) reports 91.4 against 811.2 average input tokens and gives
a cost table in GPU-hours — the only paper in your corpus that prices its
accuracy. Follow it: tokens per prompt, GPU-hours, peak VRAM, all measured.

You already have the VRAM story from the first run: **3.87 GB at L0 rising to
6.19 GB at L3/L4**, and runtime 2196 s → 5850 s. That is a 60% memory increase
and 2.7× the compute for context whose value has never been measured. Stated
that way it motivates the whole chapter.

---

## 8 · What to build, in order

| | step | cost |
|---|---|---|
| 1 | **Inductive split** — adopt CATS's, or hold out entities and verify no test entity appears in train | ~half a day, no GPU |
| 2 | **Content ablation (§3)** — 6–9 conditions, MRR + tokens each | the bulk of the compute |
| 3 | **Fixed policies** from the ablation: uniform-all, uniform-cheap, and 2–3 hand-written rules | 2–3 runs |
| 4 | **Routed policy** using has-description and relation (§4) | 1–2 runs |
| 5 | **Random-routing control** (§6) | 1 run |
| 6 | Faithfulness on the routed policy — you already have this machinery | free |

Steps 1–3 alone give you a publishable chapter: *what each context type is worth
per token in inductive KGC*. Steps 4–6 add the routing claim on top.

---

## 9 · The one-paragraph pitch

> Recent LLM-based inductive KGC systems supply large amounts of context —
> reasoning paths, neighbouring facts, supporting triples, type constraints,
> background facts — and report accuracy with all of it switched on. None reports
> what each component costs or what it buys. We measure the marginal value per
> token of each enrichment type in the inductive setting, where context is the
> only available signal because test entities are unseen. We then route
> enrichment per element using the two features that carry variance on real
> graphs — whether a description exists at all, and which relation is being
> queried — and show that matched ranking quality is achievable at a fraction of
> the token budget. Every routing decision carries a stated reason, and we
> measure what fraction of those reasons are operative.

No dependence on Chapter 1 anywhere in it.
