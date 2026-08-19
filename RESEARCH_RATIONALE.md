# Research Rationale — how this contribution was arrived at

**Thesis.** *Enrichissement explicable et orienté qualité des graphes de connaissances assisté par les LLMs*
**Base paper (supervisor-assigned).** KG-LLM — Yao, Peng, Mao & Luo, *Exploring Large Language Models for Knowledge Graph Completion*, ICASSP 2025 (arXiv 2308.13916, 2023)
**Corpus surveyed.** 188 PDFs, 31 read in full with structured notes (P01–P31)

This document answers, in order:

1. What question did I ask of each paper, and what did I find?
2. What are the limits of KG-LLM specifically?
3. Which of those limits is real, unoccupied, and testable on my hardware?
4. Why this contribution and not the alternatives I considered?
5. Why this model, this dataset, this configuration?
6. What is the environment, and what had to be fixed in it?

---

# 1 · The question I asked every paper

Reading 188 papers requires one question applied uniformly, or the notes become
summaries and nothing accumulates. Mine was:

> **A system reports accuracy X on a benchmark. What else could produce X, and
> did the authors rule it out?**

That question has a small number of answers, and they became the columns of the
literature matrix: *format compliance*, *memorisation of public benchmarks*,
*easy negatives*, *an unmeasured floor in the prompt*, *a protocol that cannot
express failure*. Each is a way of getting a high number without the capability
the paper claims.

Applying it produced a consistent pattern. The field is organised on three axes
— adapt the model, enrich the context, constrain the output — and papers compete
within an axis. **No paper controls for what the model already knew about the
entity names**, and the benchmarks are all long-public.

---

# 2 · KG-LLM: the eleven limits I recorded

These are from the full read (note P17). They are ordered by how much they
affect what the reported number means.

| # | Limit | Why it matters |
|---|---|---|
| **M1** | The headline gain is never decomposed into **format vs. knowledge**. Untuned LLaMA-7B scores 9.1 on a binary task where chance is 50. Their own Table VI shows why: refusals, hedges, pretraining echoes — *not wrong facts*. | ★★★ The founding result of the subfield may be mostly parseability. This became my $\phi$ term. |
| **M7** | **No contamination check.** WN11/FB13 (2013), WN18RR/YAGO3-10 (2018) are all long-public. Their explanation of the gains — *"LLMs contain more general knowledge"* — is indistinguishable from memorisation. Table VI shows raw pretraining text surfacing verbatim. | ★★★ This is the contribution. It is stated by the authors as an explanation and never tested. |
| **M8** | **No ranking, therefore no MRR.** A single greedy generation gives one answer, so Hits@3/10 and MRR are not computable. | ★★ Makes generative KGC incomparable to embedding methods, and hides failure behind a 0.5 chance level. |
| **M2** | The protocol **punishes honest abstention**. GPT-4's *"I cannot verify specific personal information"* is scored wrong. | ★★ Any calibrated system is structurally disadvantaged. Measured: 0.4 % of untuned responses are refusals scored negative. |
| **M5** | **Entity prediction fails and the conclusion elides it.** 0.133 Hits@1 on YAGO3-10; GPT-4 beats the method on the entity-prediction subset (0.24 vs 0.22), uncommented. | ★★ The task that actually completes anything is the one that fails. |
| **M10** | **No variance, no significance, single run** on any table. The `+5Neighbors` vs `KGLM` gap on WN18RR is 0.010. | ★★ Motivated my paired bootstrap. |
| **M4** | Neighbour context is **untyped and randomly sampled**, `K=5`, never swept. | ★ Became Chapter 3. |
| **M3** | **No entity grounding**: substring matching over a 123,182-entity vocabulary. | ★ Consequence: 0.0949 Hits@1. |
| **M6** | The GPT-4 comparison is **not reproducible** — "the web interface", no version, no date, 100 hand-labelled examples. | — |
| **M9** | **Inverse scaling uncommented**: 7B beats 13B on relation prediction. | — |
| **M11** | No confidence, no calibration, no provenance. | ★ Corpus-wide pattern. |

**M1 and M7 are the same defect seen from two sides.** M1 says the gain may be
format; M7 says it may be memorisation. Both say: *a single accuracy figure
cannot distinguish the capability claimed from two cheaper explanations.* That
is the gap, and it is stated by the paper itself as an unargued assumption.

---

# 3 · What the other papers do, and where each stops

I only list the papers that bear directly on the choice.

### Papers that could have pre-empted me

**P22 · Knit** (*Big Data Mining and Analytics* 2026) — the closest descendant of
KG-LLM. Same four benchmarks, same three subtasks, same LLaMA family, and it
criticises KG-LLM head-on. It occupies the fine-tuning and structure-injection
axes.
**Where it stops:** `anonymi*` / `contaminat*` / `leakage` = **0 hits** in 21
pages. It reproduces the confounded comparison exactly (untuned 0.091 → tuned
0.892) and never decomposes it. The word `confidence` does not appear once in a
paper about hallucination.
→ *It takes the method gaps and leaves every measurement gap.*

**P09 · EGIT** (*Information* 2026) — gives the field its formal definition of
hallucination in KGC, including *"over-reliance on the language model's
parametric knowledge rather than graph evidence."* That is my finding, stated as
a definition by someone else.
**Where it stops:** it never measures parametric reliance. On WN18RR its entire
benefit is a Hits@1 reranking gain; MRR moves 0.003 and Hits@10 *degrades*, which
the paper does not acknowledge. Its two headline innovations are never ablated,
and hallucination is never measured despite the title.
→ *It proposes the cure and never runs the diagnostic.*

**P12 · KG-CF** — the closest prior use of my instrument. Its `−te` ablation
anonymises trajectory entities and finds performance falls significantly,
confirming memorisation contributes materially.
**Where it stops:** the ablation checks *one component of their own pipeline*,
not a headline figure, and the paper never names the LLM it uses — no model,
version or size anywhere.
→ *The instrument exists in the corpus. Nobody points it at the founding result.*

### Papers that add context and never remove the name

| Paper | What it adds | Where it stops |
|---|---|---|
| **P27 · CATS** | reasoning paths, latent type constraints | *"It adds type information and reports accuracy rising. It never removes the names."* |
| **P26 · RealKGC** | relation-constrained structure blocks | its own motivation presupposes a long-tail figure it never quantifies; never removes names |
| **P19 · GS-KGC** | subgraph neighbourhoods, negatives | no name control |
| **P30 · KICGPTv2** | in-context demonstrations | selects context by relevance; no name control |
| **P29 · struOKGC** | relation-specific templates | tailors per relation, never prices it |
| **P16 · APE** | auto-generated prompts | **L1**: the core mechanism is not reproducible — no search algorithm, no prompt pool, no stopping criterion, no code. **L3**: generated knowledge worth +10.5 points is verified against nothing. **L4**: claims to avoid fine-tuning, then fine-tunes with LoRA. **L5**: 50.2 GB on 4×A6000 described as "modest hardware" |
| **P20 · MKGL** | vocabulary reconstruction | **the only paper in 188 that prices context** (91.4 vs 811.2 tokens) |

### Verified-empty searches

Run across all 188 PDFs (first 6–8 pages each), and this is what licensed the
claim of novelty rather than an assumption of it:

| pattern | hits |
|---|---|
| `name-invarian*` / invariance loss | **0** |
| mask / randomise / permute **entity name** | **0** |
| exclusion or negative **prompting** | **0** |
| prompt **compression** / token budget | **0** |
| *"when to retrieve"* / per-query compute budget | **0** |
| `anonymis*` | 1 (enterprise privacy, not KGC) |
| accuracy-vs-cost Pareto curve | 1 |
| chain-of-thought | 21 — saturated, avoided |

---

# 4 · Why this contribution, and what I rejected

## 4.1 The argument, in five steps

1. KG-LLM reports 95.5 % on WN11 against a 50 % chance level and explains it as
   *"LLMs contain more general knowledge"* (their words).
2. That explanation and *"the model memorised these public benchmarks"* predict
   **the same accuracy**. The protocol cannot separate them.
3. They differ in exactly one observable way: a relational rule survives the
   replacement of entity names by opaque identifiers; a name-to-label
   association does not.
4. So train the same pipeline twice — once on the real graph, once on an
   anonymised copy — and report both accuracies as a pair. The difference is the
   share carried by surface form.
5. This requires **one extra training run**, no new architecture, and attaches to
   any text-based method without modifying it.

## 4.2 Why the obvious objection is handled

> *"Replacing names with `entity4471` destroys readability as well as identity,
> so of course accuracy collapses. That tells us nothing."*

This objection is correct against an anonymisation-only design, and it is why
**condition S** exists. S keeps every real name in the graph and only permutes
which entity holds which. Vocabulary, token distribution, name lengths and
readability are all preserved; only the name-to-node binding is destroyed.

Measured: MRR falls 0.8169 → 0.2974, which is 71.5 % of all above-chance skill.
The collapse cannot be attributed to unreadable identifiers, because the
identifiers are ordinary English.

**S is the load-bearing condition of the thesis, not B.**

## 4.3 Alternatives I considered and rejected

| Alternative | Why rejected |
|---|---|
| Beat KG-LLM's accuracy with a better method | Knit (P22) already occupies that axis on the same benchmarks with more compute. I would lose on hardware alone. |
| Improve prompting (more context, better templates) | Six papers already do this. And my own result says context cannot help where names carry the signal. |
| Chain-of-thought for KGC | 21 papers. Saturated. |
| Data curation / deduplication to remove leakage | **Ruled out by my own measurement.** The tuned familiarity gap is +0.0036: withholding an entity from fine-tuning changes nothing, so the association is *pretrained*. No change to fine-tuning data can touch it. |
| Larger model | Cannot run 7B on the available hardware, and it would confound scale with method. |

The last row is the important one: a measurement I made **eliminated a whole
class of solutions** before I spent time on them.

## 4.4 Why link prediction rather than triple classification

The decomposition needs a stable, non-zero chance level, since above-chance
performance is its denominator.

| protocol | is it the task? | stable denominator? |
|---|---|---|
| triple classification | **no** — completes nothing | yes, 0.5 |
| full ranking | yes | **no** — chance ≈ 0; 123k passes/query |
| **50-way ranking** | **yes** | **yes**, MRR = 0.0900 |

Sampling 50 candidates (following CATS and RealKGC) is therefore not a cost
compromise but the only protocol that is simultaneously the completion task and
decomposable. It also fixed a real problem: at chance 0.5, three of my
conditions collapsed onto a single answer and scored *exactly at chance* while
having learned nothing. At chance 0.0900 that failure is unmistakable.

---

# 5 · Choice of model, data and configuration

## 5.1 Model — Qwen2.5-1.5B-Instruct

| criterion | reasoning |
|---|---|
| **Size** | Only in-corpus precedent for sub-7B on KGC is GS-KGC's Qwen2-1.5B vs 7B ablation, and CATS also ablates Qwen2-1.5B. The size is therefore **matched by their choice, not by my budget**. |
| **Family** | FLAME validates the Qwen family on this task. |
| **Instruct variant** | KG-LLM used base LLaMA. Using an instruct model raises the untuned baseline (0.6920 vs their 21.1/9.1), which *strengthens* the format argument: much of their tuning gain was format compliance that an instruct model already has. |
| **Not LLaMA-7B** | fp32 = 27 GB, fp16 = 13.5 GB against 14.56 GB usable — training does not fit on a T4. 4-bit QLoRA would fit but changes family, scale **and** quantisation at once. |

**Stated honestly as a threat:** a 1.5 B model may memorise where a larger one
generalises. The paper scopes its claim to this size and names a scale sweep as
the single most valuable extension.

## 5.2 Datasets

**YAGO3-10 primary** (123,182 entities, 37 relations). Its relations are
strongly typed — `wasBornIn` maps a person to a place, `playsFor` a person to a
club — which is the structure the type conditions require.

**WN11 secondary** (38,588 entities, 11 relations). Lexical relations, so an
induced type does not recover a semantic type.

Running both is a **decomposition, not a replication**: memorising on WN11 is
lexical, on YAGO3-10 world-factual.

## 5.3 Fixed configuration

Everything below is frozen across all conditions; only the variable under test
changes.

```
LoRA        r=8, alpha=16, dropout 0.05, target [q_proj, v_proj]
            (KG-LLM's own values, and MKGL's)
training    2 epochs, lr 3e-4, micro-batch 4, grad-accum 8 -> effective 32
data        10,000 triples -> 20,000 instances (1 pos + 1 neg, KG-LLM's recipe)
tokenizer   cutoff 512, dynamic padding
decoding    logit-level P(Yes) vs P(No) — no generation
ranking     500 queries, 50-way, filtered against train u valid u test
```

**Why effective batch 32 and not KG-LLM's 128:** at 20,000 instances, 128 gives
only 156 steps/epoch, so a 100-step warm-up would be 32 % of training.

**Why the logit rule everywhere:** it is format-free by construction, so the
memorisation measurement cannot be contaminated by the parsing artefact that
M1 identifies.

---

# 6 · Environment, and what had to be fixed in it

## 6.1 Platform

| | |
|---|---|
| compute | Kaggle, NVIDIA Tesla T4 ×1, 16 GB (14.56 GB usable) |
| session | 12 h max, ~30 GPU-h/week |
| stack | torch 2.10 + cu128, transformers 4.57.6, peft 0.19.1, trl 1.10.0 |
| persistence | ⚠️ `/kaggle/working` does **not** survive session end |

## 6.2 Correctness fixes that were not optional

**fp16 + eager attention returns NaN on Qwen2.5.** Measured on T4:

| dtype | attention | finite logits | VRAM |
|---|---|---|---|
| fp16 | eager | **False — NaN** | 3.11 GB |
| fp16 | sdpa | True | 3.12 GB |
| fp32 | eager | True | 6.22 GB |

NaN logits surface as `train_loss = 0.0` with `grad_norm = nan`, which **looks
like a finished run**. Config now pins fp32 weights + fp16 compute + SDPA, and
the trainer exits non-zero on `train_loss == 0.0`.

**Right padding is required for scoring.** Left padding creates a fully-masked
prefix whose NaNs propagate into valid positions. Scoring forces right padding
and falls back to unpadded single-prompt inference if any logit is non-finite.

**A 468× performance bug, twice.** `build_relation_type_index` was rebuilt per
call inside negative generation; the same pattern reappeared in Chapter 3's
`candidate_blocks`. Both hoisted.

**An OOM that cost two completed runs.** The SMI baseline loads a second fp32
model while the first is still resident. The evaluation results were saved
first, so the accuracies survived, but the representation measurement was lost.

## 6.3 Measurement artefacts found and removed

**The type-tag leak.** Before use, a one-line rule — *if the candidate tail's
type tag names the query relation, answer yes* — scored **62.4 %** on YAGO3-10's
original test negatives with no model at all. Regenerating the negatives
type-consistently reduced it to **51.3 %** (separation 0.026). The same audit on
WN11 returns **56.8 %** (separation 0.136), concentrated in the taxonomic
relations — `_type_of` 72.5 %, `_member_holonym` 71.4 % — which are exactly those
a type-augmentation method would claim to help.

**Consequence:** no typed condition is reported on WN11, and the code refuses to
run one until the negatives are regenerated. An unmeasured floor of that size is
silently added to every typed result in the literature.

**A base-rate confound in my own analysis.** The familiarity buckets have
positive rates 0.607 / 0.500 / 0.407, so an always-affirmative predictor earns a
+0.20 raw gap having learned nothing. An earlier version of the analysis reported
the raw figure (+0.0964) and asserted a familiarity effect that does not exist.
Balanced accuracy gives **+0.0036**.

**A statistical threshold below the noise floor.** The first report declared a
result at ±0.005 MRR; the standard error at n=300 is ≈0.02. Verdicts are now
driven by a **paired bootstrap** — the same queries against the same frozen
candidates, so query difficulty cancels — and a difference is a result only when
its 95 % interval excludes zero. The estimator is validated on data with a known
answer: 4.0 % false positives against 5 % expected, and pairing narrows the
interval by 52 %.

## 6.4 Reproducibility measures

- **Frozen candidate sets.** Negatives are sampled once, written to disk with a
  per-query seed, and fingerprinted. Every policy ranks against byte-identical
  candidates *by construction*, not by lucky execution order.
- **Filtered against train ∪ valid ∪ test.** The first version omitted valid,
  scoring true validation facts as negatives.
- **Pre-registered interpretations.** Every outcome of every condition was written
  into the code before the runs.
- **Non-zero exits.** Split validation, the type-leak audit and the degenerate-run
  check all halt the pipeline rather than warn.
- **49 end-to-end + 27 unit tests** on a synthetic graph, no GPU, ~10 s, run
  before any GPU spend.

---

# 7 · What I claim, and what I do not

**Claimed.** On YAGO3-10 at 1.5 B parameters, 94.0 % of above-chance link
prediction skill is entity surface form; a permuted-name control attributes 71.5
points of that to the name-to-node binding rather than to readability; a
familiarity split places the binding in the pretrained backbone rather than in
the fine-tuning set; induced type text does not substitute for names and costs
7.3 accuracy points when supplied alongside them.

**Not claimed.**

| | why not |
|---|---|
| "Memorisation is a defect" | On a fixed graph it is a correct strategy. The claim is that it is not what the literature reports itself as doing. |
| "This holds at 7 B" | Not tested. Named as the most valuable extension. |
| "Curated semantic types cannot help" | Our types are *induced*; the inventory is concentrated (largest class 27.4 %). The results bound induced types only. |
| "Negative count does not matter" | Condition E collapsed. That bounds the recipe, not the idea. |
| "Policy A and B are equivalent" | A CI containing zero means the effect was **not measurable**, not absent. |

---

# 8 · One-paragraph summary for a jury

> KG-LLM reports 95.5 % triple classification accuracy and explains it by saying
> language models contain general knowledge. That explanation and "the model
> memorised a benchmark that has been public since 2013" predict the same
> number, and the protocol cannot separate them. I separate them by training the
> same pipeline on the graph and on an anonymised copy of it, and reporting both
> accuracies as a pair. Evaluated as link prediction, where chance is 0.0900
> rather than 0.5, 94 % of above-chance skill disappears when entity names are
> replaced. A control that keeps every real name and only permutes which entity
> holds which loses 71.5 % on its own, so the operative variable is the binding
> between a name and a node, not the readability of the text. A familiarity
> split shows the binding was not learned during fine-tuning, which locates it
> in pretraining. The instrument costs one extra training run and attaches to
> any text-based completion method without modifying it.
