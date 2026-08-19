# Chapter 3 — specificity, not quantity

## Your question, stated as a thesis claim

> The field has established that **more context is not better**. But *more* and
> *less* is the wrong axis. The question is not how much information an element
> receives — it is **how specifically that information is matched to the
> element**. No prior work varies specificity while holding quantity fixed.

That is a real gap, and it is yours. Five papers in your corpus report the
quantity finding (P02, P08, P11, P12, P19, P20, P21). MKGL prices context at
91.4 against 811.2 tokens. CATS supplies δ=6 paths, σ=6 neighbours, k=3 supporting
triples; RealKGC supplies three constraint blocks; GS-KGC supplies negatives and
neighbours. **Every one of them gives every element the same treatment.** The
specificity axis is unexamined.

---

## 1 · ★★★ The one design move that makes it testable

**Hold the token budget constant. Vary only the allocation.**

This is the whole method. If levels differ in both quantity and specificity — as
L0–L4 currently do — then no difference can be attributed to either, and the
result is uninterpretable no matter which way it comes out. Your first run already
shows the symptom: L3 uses more tokens *and* is more specific, so its
loss drop tells you nothing about specificity.

```
budget B tokens of context per query, IDENTICAL for every policy
     what changes is WHERE those B tokens go
```

| policy | allocation rule | your level |
|---|---|---|
| **S0 uniform** | every element gets B/2 to head, B/2 to tail, truncated | baseline |
| **S1 entity property** | allocate by a property of the element (has description / degree) | L1 |
| **S2 semantic type** | allocate by the type the relation implies | L2 |
| **S3 label quality** | allocate by how informative the existing label is | L3 |
| **S4 instance** | allocate per (head, relation) query | L4 |

Every row costs the same. Any difference in MRR is **specificity**, and nothing
else. That is the experiment.

★ The comparison you must include is **S0 at budget B**. Without it, a good S4
result is indistinguishable from "we happened to spend more tokens".

---

## 2 · Concrete instantiation for link prediction

**Task.** Given `(h, r, ?)`, rank candidate tails. 50-way filtered, following
CATS and RealKGC. `chapter1/rank.py` already produces MRR and Hits@1/3/10 —
nothing new to build there.

**Setting.** Inductive. Test entities unseen, so the prompt is the only signal
about them and allocation actually matters. Adopt CATS's splits, which RealKGC
also uses.

**The allocation, made concrete.** Suppose B = 120 context tokens per query and
the available blocks are: entity description, typed neighbours, relation
description, exclusion list.

```
S0 uniform      60 tokens to head, 60 to tail, regardless of what they are

S4 instance     head has a description, tail does not
                ->  20 tokens to head (its description is enough)
                   100 tokens to tail (neighbours, because it has nothing else)
```

Same 120 tokens. Completely different distribution. **That is specificity, and it
is measurable.**

⚠️ Implementation detail that decides whether this works: the budget must be
enforced by **truncating to a token count**, not by counting blocks. Blocks have
wildly different sizes, so "two blocks each" is not a fixed budget.

---

## 3 · The budget sweep is the figure

One number per policy is weak. A curve is a result.

```
        MRR
         |                        S4 ____----
         |              S2 ___----
         |      S0 __---
         |___---
         +-------------------------------- budget B
             30      60     120     240
```

Run every policy at B ∈ {30, 60, 120, 240} tokens. Three outcomes, all
publishable, and you should write down which you expect before running:

| shape | conclusion |
|---|---|
| S4 above S0 **at every budget** | ★ specificity pays independently of quantity — the headline |
| curves converge as B grows | specificity matters **only under a tight budget** — a precise, useful claim |
| curves overlap everywhere | specificity is a red herring; only quantity matters. The **specificity analogue** of "more context is not better", and a genuine contribution to that line |

The third outcome is why this chapter cannot fail. It answers a question nobody
has asked, and a null answer still closes it.

---

## 4 · Fix the empty block first

`build_enrichment_extras` currently generates relation descriptions as:

```python
f"the relation '{txt}' links a subject to an object"
```

The same sentence 37 times with one word swapped. It carries no information the
relation name does not already carry — which is why L0 → L1 moved train loss by
0.00025 (0.050475 → 0.050223).

**Generate the 37 relation descriptions with an LLM.** One call each, minutes of
work. Two reasons it matters:

1. ColKGC and RelSemEnh both measured that **generating relation descriptions
   pays** while rewriting entity descriptions gives ~0 gain, precisely because
   relation descriptions are commonly missing. You are leaving the one enrichment
   with prior support switched off.
2. Your thesis is *"Enrichissement… assisté par les LLMs"*. Right now **no LLM
   generates anything** in Chapter 3 — every block is graph-derived or template.
   Those 37 calls are what make the chapter LLM-assisted enrichment rather than
   context selection.

---

## 5 · What to build, in order

| | step | why | cost |
|---|---|---|---|
| 1 | **Inductive split** (adopt CATS's) | makes context the only signal | half a day, no GPU |
| 2 | **37 LLM relation descriptions** | fills the empty block; makes it LLM-assisted | minutes |
| 3 | **Token-budget enforcement** in prompt building | the entire method depends on it | a day |
| 4 | **S0 uniform at 4 budgets** | the baseline every claim rests on | 4 runs |
| 5 | **S2 and S4 at the same 4 budgets** | the specificity claim | 8 runs |
| 6 | **S1, S3** | fills the ladder | 8 runs |
| 7 | **Random allocation at matched budget** | the control: is it the *decisions* or just the budget? | 1–2 runs |

Steps 1–5 give a complete, defensible chapter: *does specificity pay at fixed
cost in inductive link prediction?* Steps 6–7 make it thorough.

⚠️ 12 runs at ~40 min is ~8 GPU-hours. Cut the budget sweep to three points
before cutting policies — the curve shape is the result, but S0 vs S4 is the
claim.

---

## 6 · What you already have that carries over

- `rank.py` — 50-way filtered MRR and Hits@K, and the caption warning about
  N-way comparability
- `routing/faithfulness.py` — per-decision reason auditing. **Keep this.** An
  allocation policy that states why it allocated, and a measurement of how often
  that reason is operative, is the *explicable* half of your title and nobody
  else reports it
- `routing/features.py` — the element features the policies key on
- The VRAM and runtime instrumentation — 3.87 GB → 6.19 GB, 2196 s → 5850 s is
  your motivation paragraph

---

## 7 · The positioning sentence

> Recent LLM-based knowledge graph completion systems supply substantial context
> and report that beyond a point more context does not help. That finding is
> about quantity. We ask a different question: at a **fixed** context budget,
> does it matter **how specifically** the context is matched to the element being
> completed? We define four levels of specificity — entity property, semantic
> type, label quality, and per-instance — enforce an identical token budget
> across all of them, and evaluate on inductive link prediction, where unseen
> test entities make prompt context the only available signal. Each allocation
> decision carries a stated reason, and we measure what fraction of those reasons
> are operative.

No dependence on Chapter 1, and the gap it claims is real.
