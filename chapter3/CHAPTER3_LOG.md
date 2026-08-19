# Chapter 3 — Where should the context budget go?

**Question.** Prior work asks *how much* context helps and finds that beyond a
point more does not. We fix the amount and ask **where it should go**, then price
every answer in tokens.

> At a **fixed** context budget, does it matter **how specifically** the context
> is matched to the element being completed?

**Task.** Link prediction, 50-way filtered, both directions, on an **inductive**
split — test entities are unseen, so context is the only thing the model has.

**Dataset.** WN18RR-subset-inductive, from CATS (AAAI 2025), the splits RealKGC
also adopts.

**Model.** Qwen2.5-1.5B-Instruct + LoRA (r=8, α=16, q/v projections), fp32
weights with fp16 compute and SDPA attention, one T4.

---

# 0 · The design, before any data

## The grid

Every policy receives the **same token budget** and differs only in priority
order, so any MRR difference is attributable to allocation and nothing else.

| id | routes on | level |
|---|---|---|
| **S0** | nothing — the uniform baseline | — |
| **R** | nothing, shuffled — ★ **the control** | — |
| **S1** | does a description exist? | L1 |
| **S2** | type entropy of the query relation | L2 |
| **S3** | how informative the label is | L3 |
| **S4** | per (head, relation) query | L4 |
| **S5** | ★ semantic specificity — depth, polysemy, IDF | L5 ⚠️ gated |

**Budgets:** 0 · 30 · 60 · 120 · 240 tokens. B=0 is the floor.

**Why R matters.** If the best policy ≈ R, the *decisions* added nothing and only
the budget mattered — a clean negative result. Without R, "S4 ≈ S0" cannot
distinguish *specificity does not pay* from *our policy is bad*.

## Counted in tokens, never in blocks

A typed-neighbour list is ~60 tokens and a type tag ~3. "Two blocks each" would
let a policy smuggle in twenty times the context while appearing matched.
`budget.py::allocate` enforces the token count, and oversized non-atomic blocks
are truncated on a word boundary rather than dropped.

---

# Phase 0 — data and gates (free, CPU only)

## 0a · Getting the splits

**The CATS repository contains code only** — 14 files, all `.py`/`.pdf`/`.png`.
Its README points the data at a Google Drive folder, so cloning can never produce
a split. Downloaded the folder (1,973 MB, 93 files) and mounted it as a Kaggle
Dataset.

## 0b · Conversion

`scripts/convert_cats.py` maps CATS's layout onto ours:

| CATS | ours | why |
|---|---|---|
| `train_full.txt` (5,410) | `train.tsv` | the training graph |
| `test.txt` (188) | `test.tsv` | the ranking queries |
| **`inductive_graph.txt` (1,618)** | **`valid.tsv`** | ★ the observable graph for unseen entities |
| `ranking_head.txt` (9,400) | `candidates_tail_*.json` | 50 per query |
| `ranking_tail.txt` (9,400) | `candidates_head_*.json` | 50 per query |

★ **`inductive_graph.txt` → `valid.tsv`** gives it two correct roles at once: it
is filtered against during ranking (those triples are true, so not negatives),
and `GraphIndex` reads it as the support from which neighbour blocks are built.
Discarding it would leave every test entity with no context.

⚠️ **CATS's ranking filenames are inverted** relative to the obvious reading:
`ranking_head.txt` holds the head fixed and varies the tail, i.e. it is *tail*
prediction. Trusting the name produced candidate files with **zero queries**
while reporting success. The converter now detects which slot varies from the
data.

**Result:** 188 queries × 50 candidates, both directions, fingerprint
`f508fa570166`. Using CATS's own sets makes the numbers directly comparable to
CATS and RealKGC.

## 0c · Validation

```
✓ test entities unseen in train    all test entities are new
✓ relations shared                 all 11 test relations seen in train
✓ every test triple has an unseen entity
```

## 0d · The S5 gate

`profile_specificity.py` runs three checks that can each kill the policy before
it is built: depth variance, depth vs log-degree, depth vs cluster depth. S5 is
**not reported** unless ≥2 of 3 pass — reporting the profile as the reason is
itself a finding.

## 0e · Relation descriptions and the N9 quality gate

Generated one description per relation with the model, then scored them **before
use** on five checks: template, distinct, length, informative, typed.

```
passed 6/11   rejected 0
  template     100.0%      distinct     100.0%
  length        63.6%      informative  100.0%
  typed         81.8%
```

★ No paper in the 188-paper corpus scores its generated enrichment before using
it. That gap is **N9** in the skeleton, and this table is the contribution.

## 0f · Building the prompts

**56 cells** = 7 policies × 4 budgets × 2 directions, over 188 queries. Two
guards run automatically.

**Guard 1 — is there anything to allocate?**

```
[data] graph index: 1,806 inference-graph facts added for unseen entities
[data] 174/188 sampled queries have a neighbours block (93%)
```

If this were 0, unseen entities would have no context and the ladder could not
differ from the baseline — a flat result would be an artefact.

**Guard 2 — do the policies actually differ?**

```
B=30  tail  7/8 distinct ✓ informative
      ≡ S1_property, S3_quality   (same feature on this graph)
```

**8 informative cells, 0 dead.**

---

# ★ Phase 0 findings

## S1 ≡ S3 on WordNet, provably

```
FEATURE VARIANCE across 188 queries
  has_description   100.0% true    ✗ CONSTANT -> S1_property cannot decide
  label_words       min 3  max 50  sd 9.08     ✓ varies
  type_entropy      min 1.41 max 2.57 sd 0.124 ✓ varies
  degree            min 2  max 24  sd 4.06     ✓ varies
```

Every WordNet entity has a gloss, so `has_description` is constant and S1 has
nothing to route on. `label_words` varies but S3's threshold is `≥3` and the
**minimum is 3**, so the test never fires false. Both collapse to a constant
ordering and produce byte-identical prompts.

This is the "95.7 % band" defect from Chapter 1 recurring on a different feature,
and it is **pre-registered** in `policies.INTERPRETATION` as *"on this graph
label QUALITY and label PRESENCE are the same feature — a finding about the
graph."* Report it; do not hide it.

## The type tag is nearly contentless

```
[types] 13 distinct types | OTHER = 93.3%
```

S2's *routing decision* varies (relation-level entropy does), but the block it
inserts is the literal string `[OTHER]` for 93 % of entities. This bounds what
S2 and any type-based conclusion can claim on this graph.

## Where each policy actually spends its budget

Mean tokens per query at B=120 — the qualitative evidence that the policies are
doing different things:

| policy | demos | ent_desc | neighbours | rel_desc | tags |
|---|---|---|---|---|---|
| S0_uniform | 8.2 | 22.2 | 39.4 | **47.0** | 3.0 |
| S1_property | **47.4** | 22.6 | **0.1** | 46.8 | 3.0 |
| S3_quality | **47.4** | 22.6 | **0.1** | 46.8 | 3.0 |
| S2_type | 19.7 | 10.1 | **64.5** | 23.0 | 2.4 |
| S4_instance | 13.8 | 22.6 | **63.5** | 19.5 | 0.4 |
| S5_semantic | 14.1 | 22.6 | **67.0** | 14.3 | 1.8 |
| R_random | **59.8** | 7.6 | 28.7 | 22.5 | 1.2 |

Two clean camps: **S1/S3 buy demonstrations and almost no neighbours (0.1
tokens)**; **S2/S4/S5 spend two-thirds on neighbours**. Same 120 tokens, opposite
strategies.

**`utilisation 100 % · blocks dropped on 100 % of queries`** — the budget binds
on every query, so the allocation decision is real and no policy is quietly
getting more context.

---

# Phase 1 — the shared model (~1 GPU-hour)

## Why one model

One model serves every (policy, budget) cell, trained on a **mixture**: random
policy, random budget and random direction per example. P28's context-corruption
idea repurposed — the model sees many context subsets, so no policy is out of
distribution at evaluation time.

## Training set

```
10,820 examples from 5,410 triples
5,410 positive · 5,410 negative (50%)
random policy + budget + direction per example
```

## Result

```
614/614 [59:49, Epoch 2/2]
train_loss 0.00786 · eval_loss 0.01231 · peak VRAM 7.86 GB
adapter: 40.3 MB -> 4.2 MB exported (3.8 MB zipped)
```

⚠️ **Fit verdict — TRAIN/EVAL GAP.** train 0.0079 vs eval 0.0123. The model fits
its training split 1.6× better than held-out data. The shared model was trained
on a mixture precisely so nothing would be out of distribution, and it *still*
shows the gap. **This is the memorisation Chapter 1 measures, appearing again in
Chapter 3's own training. State it explicitly.**

Eval loss barely moved between steps 250 and 500 (0.01232 → 0.01231), so the
model had converged; the number is not a floor.

## Go / no-go

ORACLE was retired (see below). The check is instead: does the baseline differ
from one contrasting policy at all? `S0_uniform` vs `S4_instance` at B=120,
paired-bootstrapped. If they are indistinguishable, the full grid will likely
return a null result — worth knowing before four hours of GPU.

---

# Phase 2 — the claim (~4 GPU-hours)

Three rows carry the chapter: **S0** (uniform), **R** (random control) and the
best policy, at each of B ∈ {30, 60, 120, 240}.

Then three rows that close obvious referee questions:

| | why |
|---|---|
| **untuned** | if allocation pays without fine-tuning, the claim is about the *context*, not our training recipe |
| **head direction** | CATS and RealKGC report both; one direction invites the assumption that the easy side was chosen |
| **relation prediction** | the **only** place a genuine F1 and confusion matrix exist — on the link task, per-relation P/R/F1 are all algebraically equal to Hits@1 |

## The ceiling

Computed **after the fact**: for each query, the best rank achieved by any
policy at that budget. It bounds what a perfect *router over these policies*
could reach, costs no GPU, and reports which policy wins each query.

A ceiling far above every single policy, with wins **spread** across policies, is
the case for per-query routing. A ceiling close to the best single policy says
one fixed rule already suffices.

---

# Phase 3 — the ladder (optional)

S1, S2, S3 at one budget, plus S5 **if its profiler passed**. If S5 is excluded,
the profile is reported as the reason.

---

# ★★ Bugs and artefacts found — and why each mattered

Every one of these was caught by a guard rather than by inspecting results, and
each would have produced a plausible-looking but wrong chapter.

### 1 · Inductive queries had no neighbours at all

`candidate_blocks` read neighbours from `kg.train` only — but inductive test
entities are unseen **by definition**. Measured on a fixture: **0/60 queries had
a neighbours block**, and `S2_type` produced byte-identical prompts to
`S0_uniform`.

> The chapter would have reported *"specificity does not pay"* when what it
> actually measured was *"there was nothing to allocate"*.

Fixed by giving unseen entities their inference graph. With it: **54/60**.

### 2 · The inverse-relation context leak

WN18RR holds symmetric pairs — for `_derivationally_related_form` both `(a,r,b)`
and `(b,r,a)` exist. Excluding only the literal query triple leaves its **mirror**
in the neighbour list, so the context *states the answer*.

`neighbours_of` now applies a rule that storage direction, duplicate rows and
inverse edges cannot evade: no neighbour reached from the anchor by the query
relation may be the gold entity.

★ **Worth reporting.** A system supplying "neighbouring facts" on WN18RR without
this filter is partly reading the answer off its own prompt — and none of the
context-supplying papers surveyed describes such a filter.

### 3 · The training set had no negatives

`build_one` emitted only `output: YES`. A model trained exclusively on positives
asserts everything; ranking sorts by `P(Yes)`, so all 50 candidates score ~1.0
and the ordering is arbitrary. **Every cell in the grid would have measured
noise** — after an hour of training and hours of evaluation, with no error
anywhere.

Now each positive is paired with a negative sharing **identical context**,
differing only in the candidate.

### 4 · ORACLE reported the floor as the ceiling

It allocated on `meta['helps']` — whether a block improves *this* query — which
requires 2^|blocks| forward passes per query and was never computed. So it kept
nothing, spent **0 tokens at every budget**, and reproduced the B=0 floor while
being read as *"no allocation can help here"*. Since it ran **first** as the
go/no-go, it would have ended the chapter on an artefact.

Retired, and replaced with the post-hoc policy-selection ceiling.

### 5 · Training size was silently tied to `--limit`

`train_q = kg.train[: ns.limit * 5]` meant `--limit 300` trained on **1,500 of
5,410** triples. `--limit` sizes the *evaluation* set and should never have
determined training size.

### 6 · Statistical thresholds below the noise floor

The first report declared a result at ±0.005 MRR; the standard error at n=300 is
≈0.02. Verdicts are now driven by a **paired bootstrap** — same queries, same
frozen candidates, so query difficulty cancels — and a difference is a result
only when its 95 % interval excludes zero. Validated on data with a known
answer: 4.0 % false positives against 5 % expected, and pairing narrows the
interval by 52 %.

⚠️ At **188 queries** the smallest detectable paired difference is roughly
**0.06 MRR**. Contrasts smaller than that will not resolve on this dataset.
NELL-995-subset-inductive has 476 queries and is the better choice where power
matters more than the WordNet hierarchy S5 needs.

### 7 · Candidates were identical only by luck

Negatives were sampled inline from a running RNG, reproducible only while seed,
query order **and count** all matched. Change `--limit` and two policies silently
ranked against different negatives. Now frozen to disk with a per-query seed and
fingerprinted; every result file records the fingerprint and `report.py` refuses
to compare across fingerprints.

---

# Environment notes

| issue | symptom | fix |
|---|---|---|
| **`torchao` 0.10** | `ImportError` inside peft's LoRA dispatch | uninstall; it is an unused transitive dependency |
| **T4 ×2** | `RuntimeError: 2 GPUs are visible` — DataParallel breaks autocast | `CUDA_VISIBLE_DEVICES=0` **before torch is imported**; read once at CUDA init |
| **subprocess stalls** | no CPU, no GPU, no output | notebook already owned the CUDA context; train **in-process**, evaluate in a fresh session |
| **orphan processes** | GPU held after an interrupted cell | interrupting a cell does not kill its child; `pkill -f chapter3` |
| **silent cells** | nothing for minutes | subprocess stdout is block-buffered to a pipe; `PYTHONUNBUFFERED=1`, and split on `\r` to catch tqdm |
| **session wipe** | built data gone | `/kaggle/working` does not persist; save the adapter as a Kaggle Dataset |

---

# Status

| | |
|---|---|
| splits converted and validated | ✅ |
| relation descriptions + N9 gate | ✅ 6/11 pass, 0 rejected |
| candidates frozen (CATS's own) | ✅ 188 × 50, both directions |
| prompts built | ✅ 56 cells, 8 informative |
| shared model trained | ✅ 60 min, train 0.0079 / eval 0.0123 |
| go/no-go | ⬜ |
| Phase 2 grid | ⬜ |
| untuned · head · relation prediction | ⬜ |

---

# What this chapter can already claim

Independently of how the grid comes out:

1. **Policies allocate genuinely differently** at matched cost — two camps, one
   buying neighbours and one buying demonstrations, with the budget binding on
   100 % of queries.
2. **S1 ≡ S3 on WordNet**, established from feature variance before any model
   ran. The label-quality rung of the ladder does not exist on a lexical graph.
3. **Induced types are 93 % `OTHER`** here, which bounds every type-based claim.
4. **Generated relation descriptions were quality-gated before use** — the N9
   gap, unoccupied across 188 papers.
5. **Neighbour context leaks the answer on WN18RR** unless inverse relations are
   filtered, which no surveyed paper describes.

Points 2–5 are findings about the benchmark and the method, and they hold whether
or not allocation turns out to pay.
