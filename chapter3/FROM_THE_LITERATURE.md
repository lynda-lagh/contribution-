# What P28, P29 and P30 change about Chapter 3

Three papers read in full. Notes: `1_corpus/notes/P28…P30`. Matrix rows added.
Skeleton gains **Path W** and refinements **R32–R34**.

---

## ★★★ 1 · The target changes. This is the biggest one.

**P28 (ToC, IEEE TKDE 2025)** argues and demonstrates that models learn
*co-occurrence* between the query relation and its contextual pattern, so that

> *"noisy relational patterns, which fail to provide evidence for predicting the
> query relation… **will mislead the model into considering [them] as a feature
> supporting** the existence of the query relation."*

**Low-value context is not inert. It is evidence the model learns to trust.**

| | before | after P28 |
|---|---|---|
| hypothesis | matched MRR at fewer tokens | **possibly better MRR at fewer tokens** |
| why allocation helps | assumed (cost only) | **published mechanism** |
| a null result means | allocation does not pay | allocation does not pay *even though noise is known to hurt* — still informative |

★ This is the mechanism I flagged as the professorial gap: *why should
specificity help?* Now it has an answer with a citation, not an intuition.

**Change to make:** state the hypothesis as *"allocation improves ranking quality
at reduced cost"*, and report the accuracy delta as a first-class result rather
than as a constraint to be satisfied.

---

## ⚠️ 2 · WN18RR is the wrong dataset to carry the claim

P28, on their own results:

> *"there are only **11 relations** in this dataset, so the model **has seen
> almost all combinations** of query [relation and pattern]"*

Long-tail combinations barely exist on WN18RR. Since our S1 and S4 policies are
motivated by thin, long-tail elements, **WN18RR is the weakest place to show
allocation matters.**

**Change to make:** keep WN18RR — 11 relations still make the description
generation and its quality gate tractable, which is why we chose it — but
**re-label it as the pipeline-validation dataset and make FB15k-237 the primary.**
The headline budget sweep belongs on FB15k-237.

---

## ★★ 3 · The ladder's floor is already published

**P29 (struOKGC, WWW 2025)** builds **relation-specific templates**, where

> *"`T_h` and `T_t` are token sequences formed by the description of the head and
> tail entity, respectively"*

and reports beating SOTA. That is our **S2** rung, applied unconditionally and
never priced.

**Change to make:** stop asking *"does specificity help?"* — it is settled at the
relation level. Ask **"specificity at the relation level is known to pay; where
does it stop paying, and at what cost?"** That is a better question and it makes
the ladder's shape the contribution rather than its existence.

---

## ⚠️ 4 · One sentence we can no longer write

**P30 (KICGPTv2, IEEE TKDE 2026)**'s Knowledge Prompt *"selects the most relevant
KG context information as demonstrations"*.

**Context selection already exists.** So does path filtering — P28 notes that
KRST and APST *"employ different filtering strategies to select the three most
reliable connection paths"*.

**Change to make:** never claim "we select context". The claim is:

> We allocate a **fixed token budget** across competing context sources and
> compare allocation rules **at matched cost**.

The novelty is the *constraint*, not the *selection*.

---

## ★ 5 · Three concrete additions to the design

### 5a · Demonstrations as a sixth source (from P30)

Our sources are entity description · relation description · type tag ·
neighbours · exclusions. P30 adds **demonstrations**; CATS uses *k=3 supporting
triples sharing the relation*; RealKGC *shows triples sharing `r_q`*.

Demonstrations are **large and plausibly high-value for exactly the long-tail
elements that have nothing else** — so the budget would have to choose between
many cheap tags and one expensive demonstration.

★ That is the most interesting version of the trade-off, and it is currently
absent. One extra `Block` kind in `sources.py`.

### 5b · A training-free arm (from P30)

KICGPTv2 is *"training-free… requires no finetuning"*. Every Chapter 3 policy
currently costs a fine-tuning run.

★ Allocate, then prompt an **untuned** model. Inference only. It answers *how
much of the gain survives without adaptation*, and it is by far the cheapest
extra result available.

### 5c · Context-corruption augmentation (from P28)

P28 deliberately corrupts context during training — randomly discarding and
replacing connection paths — so the model learns to identify noise.

★ Ours *withholds* blocks at inference. A model trained on corrupted context
should be **more robust to a policy that withholds**. One extra training run,
and it tests whether allocation and noise-robustness compose.

---

## ⚠️ 6 · A confound P29 makes explicit

> *"templates have diversity and uncertainty… **subtle changes in words for a
> given template may have a significant impact on performance**"*

If wording alone moves the result, then two policies whose prompts differ in
*phrasing* rather than *allocation* are not a clean comparison.

**Our design already handles this** — `budget.allocate` selects among blocks
whose text is fixed — but it must be *stated*, and P29 is the citation for why it
matters.

---

## 7 · A scoring signal we could borrow

P28 notes that **KRST introduces coverage and confidence scores** for connection
paths. Our policies score blocks with hand-written rules, which is the weakest
part of the design.

★ Using published per-path scores as the allocation priority would partly answer
*"how do I know your rules aren't tuned to this dataset?"* — the objection a
supervisor raises first. Worth considering for the neighbours block.

---

## 8 · Revised claim

> Recent inductive KGC systems supply substantial context, and ToC shows that
> unsupportive context is not merely wasteful — models learn to treat it as
> evidence. struOKGC shows that tailoring the prompt per relation improves
> results, applied unconditionally and unpriced. KICGPTv2 selects relevant
> context to compensate for long-tail information scarcity, without bounding what
> that selection costs. We fix the token budget and compare allocation rules of
> increasing specificity — uniform, random, per-property, per-type, per-quality,
> per-instance — against an oracle ceiling, measuring ranking quality, cost, and
> the operativeness of each stated allocation reason.

---

## 9 · What to change, in order

| | change | cost |
|---|---|---|
| 1 | **FB15k-237 primary, WN18RR for pipeline validation** | re-plan, no compute |
| 2 | **Hypothesis: better MRR at lower cost**, citing P28 | free |
| 3 | **Never say "select"; say "allocate under a fixed budget"** | free |
| 4 | Add **demonstrations** as a sixth block kind | ~half a day |
| 5 | Add the **training-free arm** | inference only |
| 6 | Add the **context-corruption** training condition | 1 run |
| 7 | Consider **KRST coverage/confidence** as neighbour priority | ~a day |

Items 1–3 cost nothing and change what the chapter claims. Do them before any
more compute.
