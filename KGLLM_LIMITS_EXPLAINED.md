# The eleven limits of KG-LLM — each one, with the proof

Expansion of §2 of `RESEARCH_RATIONALE.md`. Every row below is checked against
the paper itself, and every quote is copied from the PDF text.

**Source.** L. Yao, J. Peng, C. Mao, Y. Luo. *Exploring Large Language Models
for Knowledge Graph Completion.* ICASSP 2025 / arXiv:2308.13916.

**How to read this.** Each limit has four parts:

1. **In one sentence** — plain English, no jargon.
2. **The proof** — a quote or a number from their paper.
3. **Why it matters** — what it does to the claim being made.
4. **What we did** — the piece of our work that answers it.

---

## M1 · Nobody separated "the model learned facts" from "the model learned the answer format"

**In one sentence.** Untuned LLaMA-7B scores **9.1 %** on a true/false question
where guessing gets **50 %**. A model that knew nothing would score 50. Scoring
9.1 means it wasn't answering the question at all — it was writing something the
scorer couldn't read.

**The proof.** Their Table II:

| model | WN11 | FB13 |
|---|---|---|
| LLaMA-7B (untuned) | 21.1 | **9.1** |
| LLaMA-13B (untuned) | 28.1 | 17.6 |
| KG-LLaMA-7B (tuned) | **95.5** | 89.2 |

And their scoring rule, verbatim:

> *"If the ground truth is true and the response contains affirmative words like
> 'Yes' and 'yes', or if the label is false and the response contains negative
> words like 'No'/'no'/'not'/'n't', we label the response as correct."*

**Why it matters.** The jump from 9.1 to 89.2 is presented as *knowledge* gained
by fine-tuning. But a large part of it is the model learning to say the word
"Yes" instead of a paragraph. Those are different things and the number cannot
tell them apart. This is the founding result of the whole subfield.

**What we did.** We read one set of generations under **four different decision
rules** — strict substring, lenient substring, constrained, and a
generation-free rule that just compares P(Yes) to P(No). On the untuned model
the choice of rule moves accuracy by **0.0700** against an above-chance signal
of only **0.1920**. More than a third of what a substring protocol calls
"untuned capability" is an artefact of how the output is read. That became the
**format term φ** in our decomposition.

---

## M7 · No one checked whether the model had simply seen the answers before ★★★

**In one sentence.** All four benchmarks have been public for years — WN11 and
FB13 since 2013, WN18RR and YAGO3-10 since 2018. The model was pretrained on the
internet. Nobody tested whether it was recalling rather than reasoning.

**The proof.** The authors' own explanation of their gain:

> *"instruction turning leads the LLM to extract knowledge stored in model
> parameters more efficiently."*

That is a claim about **memorised parameters**, offered as an explanation and
never tested. Their Table VI shows raw pretraining text surfacing verbatim in
the outputs.

**Why it matters.** "The model knows facts about the world" and "the model
remembers this specific benchmark" produce the **exact same accuracy**. Nothing
in the paper distinguishes them. And the entities that motivate knowledge-graph
completion in the first place are new, thin and long-tailed — precisely the ones
where a remembered name is worthless.

**What we did.** This is the contribution. We replaced every entity name with an
opaque code (`entity4471`), retrained the identical pipeline, and reported both
numbers as a pair. On YAGO3-10, **94.0 %** of above-chance ranking skill
disappears — bootstrap CI **[91.5 %, 99.6 %]**.

---

## M8 · You cannot rank anything, so the standard metric is missing

**In one sentence.** Their model writes one answer. One answer is not a ranked
list, so Hits@3, Hits@10 and MRR — the metrics every embedding method reports —
cannot be computed at all.

**The proof.** Every link-prediction table in the paper reports **Hits@1 only**
(Table IV, Table V). No MRR appears anywhere.

**Why it matters.** Two consequences. Their method cannot be compared to TransE,
DistMult, or any classical method on equal terms. And triple classification —
which they *can* score — has a chance level of **0.5**, so a model that answers
"No" to everything scores exactly at chance while having learned nothing. We saw
that happen in three of our own conditions.

**What we did.** We score each of 50 candidates by P(Yes) and sort them. That is
a ranking, so MRR is computable after all, and chance drops from 0.5 to
**0.0900** — an order of magnitude lower, which is what makes the decomposition
readable. The trick is KG-BERT's, not ours; the point is that the stated
limitation was never structural.

---

## M2 · The scoring rule punishes a model for being honest

**In one sentence.** When GPT-4 says *"I don't know"*, the protocol counts it as
a wrong answer.

**The proof.** GPT-4's actual output, quoted in their Table VI:

> *"I cannot verify specific personal information about individuals who are not
> public figures. I'm sorry, but I don't have enough information to confirm
> whether Everett T…"*

The scoring rule looks for "No"/"not"/"n't". This response contains "cannot" and
"don't" — so on a **true** triple it is scored as a wrong "No", and on a false
one it is scored **correct by accident**.

**Why it matters.** Any well-calibrated system is structurally disadvantaged.
The measurement rewards confident guessing over honest abstention.

**What we did.** We measured it: **0.4 %** of untuned responses are refusals
that the lenient rule converts into negatives. Small, but it is the
epistemically correct behaviour being penalised, and it is invisible in the
headline number.

---

## M5 · The task that actually completes something is the one that fails

**In one sentence.** Triple classification just says yes or no — it adds nothing
to the graph. Entity prediction actually fills a gap. On entity prediction their
method scores **0.0949** and GPT-4 beats it, and neither fact is discussed.

**The proof.** Table IV, YAGO3-10 Hits@1:

| | YAGO3-10 | YAGO3-10-100 |
|---|---|---|
| ChatGPT | – | 0.22 |
| **GPT-4** | – | **0.24** |
| KG-LLaMA2-13B | 0.0949 | 0.16 |
| KG-LLaMA2-13B + 5 Neighbors | 0.1330 | **0.22** |

On the 100-instance subset, plain GPT-4 (0.24) beats their best system (0.22).
The paper does not comment.

**Why it matters.** The headline is 95.5 % on classification. The number for the
task that completes anything is 0.0949. Those live in the same paper without the
tension being named.

**What we did.** We made link prediction the **primary** protocol, and said so:
*"The principal result of this paper is therefore a link prediction result."*

---

## M10 · One run, no error bars, differences smaller than the noise

**In one sentence.** Every table is a single run. Some of the differences they
draw conclusions from are **0.010** — about one percent — with nothing to say
whether that is real.

**The proof.** Table IV, WN18RR Hits@1: KGLM **0.3050** vs their best
**0.3151**. A gap of 0.0101, single run, no confidence interval, no significance
test anywhere in the paper.

**Why it matters.** With one run you cannot tell a real improvement from a lucky
seed.

**What we did.** A **paired bootstrap** over per-query reciprocal ranks — pairing
cancels query difficulty, which narrows the interval by about half. It also
caught one of our own overclaims: we had written that condition C ranks *below*
B, and the bootstrap gives **C − B = −0.027, CI [−0.063, +0.009]** — not
significant. We deleted the claim.

---

## M4 · The neighbour trick is their biggest gain, and it was never tuned

**In one sentence.** They show the model 5 nearby facts. Five. Never 3, never 10.
Chosen at random, no types, no ordering — and it is the single largest
improvement in the paper.

**The proof.**

> *"for the entity prediction task, we sample **K = 5** neighboring entities
> (excluding the target entity) for the given entity and tell the model as:
> 'Giving the neighbors of Steve Jobs: Steve Wozniak|USA|Bill Gates|male|
> California.'"*

Its effect, Table IV: YAGO3-10 **0.0949 → 0.1330** (+40 %), WN18RR **0.2682 →
0.3151**.

**Why it matters.** A 40 % gain from a number nobody swept. Is 5 the right
number? Would typed neighbours do better? Unknown.

**What we did.** Chapter 3 swept the budget from 30 to 240 tokens across eight
policies: **−0.0044, CI [−0.0275, +0.0179], p = 0.655**. Eight times the context
moved nothing. And we added **P6**, which reproduces their K=5 exactly — with a
stricter guard, because they exclude only the target entity while we also drop
every edge on the query relation, which otherwise names a true answer outright.

---

## M3 · There is no real matching — it just looks for the word

**In one sentence.** To check the answer, they search the generated text for the
entity name. Over a vocabulary of 123,182 entities, with no linking, no
normalisation, no disambiguation.

**The proof.**

> *"In the case of LLMs, the response is considered correct if it contains the
> label words."*

We measured the consequence on the same data: **3.93 %** of YAGO3-10 entities
share a surface form with another entity — worst case, *"washington county"*
appears **25 times**.

**Why it matters.** Substring matching also fires inside other words. Their own
`"No"` rule matches **k·no·w** and **ca·nno·t**. And with 25 different
"washington county" entities, a match proves nothing about which one was meant.

**What we did.** A **generation-free** decision rule: compare P(Yes) against
P(No) at the answer token. No text, no matching, nothing to fool.

---

## M6 · The GPT-4 comparison cannot be reproduced

**In one sentence.** They typed prompts into a website. No model version, no
date, no API, and only 100 hand-labelled examples.

**The proof.**

> *"We input our designed prompts to the **web interface** of GPT-4 and ChatGPT
> to obtain results."*

And Table III is over **100 test instances** of FB13, hand-labelled.

**Why it matters.** GPT-4 changed repeatedly across 2023–2024. Without a version
and a date, nobody can reproduce that row — and it is the row that says their
7B model matches GPT-4.

**What we did.** Nothing directly; we don't compare to closed models. But it is
why we pin the exact backbone, the seed, and every hyperparameter, and why the
config is a single declarative file.

---

## M9 · The bigger model is worse, and nobody mentions it

**In one sentence.** On relation prediction the 7B model beats the 13B. Twice
the size, worse result, no comment.

**The proof.** Table V, YAGO3-10 relation-prediction Hits@1:

| model | YAGO3-10 | YAGO3-10-100 |
|---|---|---|
| **KG-LLaMA-7B** | **0.7028** | **0.71** |
| KG-LLaMA-13B | 0.6968 | 0.64 |

The same inversion appears in Table IV's YAGO3-10-100 column: 7B **0.16** vs 13B
**0.13**.

**Why it matters.** Scaling normally helps. When it doesn't, that is a finding —
either about the task or about the training setup. Passing over it silently
leaves a reader unable to judge which.

**What we did.** Nothing — we run one size. But it is why our Threats section
states the scale limit plainly instead of implying the result generalises.

---

## M11 · No confidence, no calibration, no provenance

**In one sentence.** The model never says how sure it is, and never says where an
answer came from.

**The proof.** The words *confidence*, *calibration* and *abstention* do not
appear in the paper. This is not specific to them: across our 188-paper corpus,
abstention appears in **0**, calibration in **2**.

**Why it matters.** A completion system that fills gaps in a knowledge graph
without a confidence score cannot be deployed safely — you cannot tell which of
its additions to trust.

**What we did.** We report **ECE**, **Brier** and **MCE**, and used calibration
as an independent instrument. It corroborates from a signal ranking never
touches: ranking uses the *order* of P(Yes) and is invariant to rescaling;
calibration uses its *magnitude* and is blind to order. When we destroy the
name-to-node binding, mean confidence falls **0.809 → 0.511** — near
indifference — while accuracy falls only to 0.603. The model registers that it
is guessing.

---

# The short version

**M1 and M7 are the same hole seen from two sides.**

- M1 says the gain might be **formatting**.
- M7 says the gain might be **memorising**.

Both say the same thing: **one accuracy number cannot tell a real ability apart
from two much cheaper explanations.**

That is the gap. And it is not something we had to argue for — the paper states
it itself, as an assumption it never checks:

> *"instruction turning leads the LLM to extract knowledge stored in model
> parameters more efficiently."*

Our whole contribution is one extra training run that turns that assumption into
a measurement.

---

## Which limit each part of our work answers

| limit | our answer | where |
|---|---|---|
| M1 format vs knowledge | four decision rules on one set of generations | §IV-G |
| M7 contamination | conditions **B** and **S**, the decomposition | §III-A, Fig. 1 |
| M8 no ranking | 50-way scoring → MRR, chance 0.0900 | §III-E |
| M2 refusals punished | generation-free rule; refusal rate measured | §IV-G |
| M5 entity prediction ignored | link prediction is the primary protocol | §I |
| M10 no variance | paired bootstrap over per-query ranks | §IV-D |
| M4 K=5 never swept | budget sweep (ch. 3) + **P6** reproduction | §VI |
| M3 substring matching | P(Yes) vs P(No), no text | §III-E |
| M6 not reproducible | pinned backbone, seed, declarative config | §IV-A |
| M9 inverse scaling | not addressed — one size, stated as a threat | §V-C |
| M11 no calibration | ECE / Brier / MCE as a third instrument | §IV-G |
