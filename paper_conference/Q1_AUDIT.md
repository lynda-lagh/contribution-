# Q1 audit of the conference paper

> ⚠️ **This is not the checklist from the LinkedIn article you linked.** That page
> is login-walled and could not be read. This is an audit against the standard,
> widely-agreed Q1 criteria — the same ones your supervisor's three PDFs encode.
> Paste the article text and I will redo this against its specific roadmap.

**Scored:** `paper_conference/main.tex`, state as of the bibliography fix.

---

## First, a terminology problem worth naming

**"Q1" is a journal quartile** (JCR / Scopus), not a quality grade that applies to
conferences. IEEE BigData 2026 is a **conference**, not a Q1 journal. So the paper
cannot literally be "a Q1 paper" while targeting this venue.

Two coherent readings, and they lead to different work:

| Reading | What it means | What it needs |
|---|---|---|
| **"written to Q1 standard"** | rigour, framing and evidence at journal level, submitted to a conference | what this audit measures |
| **"submitted to a Q1 journal"** | actually target a Q1 venue | *Knowledge-Based Systems* (Q1 — where RealKGC is), *Information Sciences* (Q1), *Expert Systems with Applications* (Q1) |

★ The natural path: submit the conference version now, extend it into the Q1
journal version afterwards. Journals expect a **30–40% delta** over the
conference paper — the extra conditions, extra scale, and extra dataset listed
below are exactly that delta. **This is an argument for submitting the short
version rather than against it.**

---

## Scorecard

| # | Q1 criterion | Verdict | Evidence / what is missing |
|---|---|---|---|
| 1 | **Novelty explicitly stated** | ✅ **strong** | §I contributions list; the instrument is new *as a measurement*, and §V-B distinguishes it from KG-CF's use of the same operation as a robustness check |
| 2 | **Gap identified and justified** | ✅ **strong** | §I ¶3 + §IV-D. Uses the "acknowledge existing research" pattern, avoids the banned phrasing, and names the cost of leaving the gap open |
| 3 | **Theoretical grounding** | ✅ **good** | shortcut learning (Geirhos, *Nat. Mach. Intell.* 2020) — a tradition neither RealKGC nor CATS cites. Formal decomposition in Eq. (1)–(2) |
| 4 | **Definitions rigorous** | ✅ **strong** | §II fixes six terms against sources, and openly reports where the corpus has no fixed definition (closed/open-domain) instead of inventing one |
| 5 | **Reproducibility** | ✅ **strong** | exact hyperparameters, single declarative condition file, validator with non-zero exit, randomised test suite with printed seed, code released |
| 6 | **Limitations acknowledged** | ✅ **strong** | §IX, six threats each with a mitigation, including the one that could sink the chapter |
| 7 | **Pre-registration** | ✅ **rare and valuable** | §V-C interpretations fixed before the runs; §VIII written as branches. Very few papers in this area do this |
| 8 | **Structure and flow** | ✅ **good** | standard IEEE arc; each section's last paragraph sets up the next |
| 9 | **Results** | ❌ **BLOCKING** | §VII is a skeleton. **No Q1 or conference reviewer can assess this paper as it stands.** Everything else is moot until this is filled |
| 10 | **Baseline comparison** | ⚠️ **structural risk** | the paper **beats nobody**. It measures a property. Reviewers trained on "SOTA + x%" will ask what the improvement is |
| 11 | **Statistical rigour** | ⚠️ **partial** | 3 seeds planned on **B vs C only**. Q1 expects variance on every reported contrast |
| 12 | **Generalisability — scale** | ⚠️ **weak** | one model, one size (1.5B). The size-matching defence is good but does not answer "does this hold at 7B?" |
| 13 | **Generalisability — data** | ⚠️ **weak** | two datasets, and **YAGO3-10's labels are self-generated**, so those numbers are not comparable to published work |
| 14 | **Qualitative analysis** | ❌ **absent** | no error taxonomy, no examples of what the anonymised model gets right. KG-LLM's own Table VI is qualitative and is one of its most-cited parts |
| 15 | **Figures** | ⚠️ **thin** | one schematic. No results figure yet — a gap-by-condition bar chart and a risk–coverage curve are both expected |
| 16 | **Language** | ✅ **good** | consistent register; British spelling throughout — check the venue does not mandate US |

**11 pass · 5 warn · 2 blocking**

---

## The three things that actually decide acceptance

### 1. Results — nothing else matters until this is done

Conditions A and B exist. That is enough for the **headline decomposition**, which
is the paper's contribution. It is not enough for the seven-condition grid the
paper promises in Table V.

**Two honest ways to close it:**

- **Narrow the paper to what you measured.** Drop the grid to A / B / S, retitle
  §V around the decomposition, and present C–G as future work. The paper becomes
  smaller and completely defensible.
- **Run the grid.** Larger claim, more risk.

⚠️ **Do not submit the grid as written with placeholder cells.** A table of
conditions the paper does not report reads as overclaiming, and it is the single
easiest thing for a reviewer to reject on.

### 2. You beat nobody — turn that into a claim rather than hiding it

This is a **measurement paper**, and the field's reflex is to look for a win.
Papers of this kind succeed when they say so explicitly and early. Add one
sentence to §I, near the contributions:

> *We do not propose a new completion method and we do not report improved
> accuracy. We report a measurement that existing methods do not make, and we
> apply it to the system that founded the subfield.*

Then give the reviewer something that behaves like a comparison. The strongest
available is a **cross-system table**: run the anonymisation gap on two or three
released checkpoints (KG-LLM's is public) and rank them **by memorisation share
rather than by accuracy**. That reorders the leaderboard, which is a result, and
it is far more compelling than any single-system number.

### 3. Fix the "one model, one scale" objection cheaply

You do not need 7B. **Two sizes of the same family** — Qwen2.5-0.5B and 1.5B —
would show whether the memorisation share moves with scale. Even two points make
the difference between "we measured one model" and "we measured a trend", and
0.5B is cheap.

---

## Quick wins, in order of value per hour

| Do | Cost | Buys |
|---|---|---|
| Add the "we beat nobody" sentence to §I | 5 min | reframes the whole review |
| Run `seen_unseen` | free, no GPU | a second instrument, and it can invalidate the headline — better to know |
| Add a qualitative table: 5 triples the real model gets right and the anonymised model gets wrong | 30 min | criterion 14, and it is the most readable thing in the paper |
| Gap-by-condition bar chart | 30 min once results exist | criterion 15 |
| Add Qwen2.5-0.5B for A and B | ~1 GPU-h | criterion 12 |
| 3 seeds on A vs B, not just B vs C | ~2 GPU-h | criterion 11 on the contrast that carries the claim |

---

## What is already better than most papers in your corpus

Worth knowing, because it is what you argue from:

- **Pre-registered interpretations.** Almost nobody does this.
- **Four instruments that fail in different ways**, reported jointly — including
  one (SMI) that superficially contradicts the conclusion. Reporting a
  disconfirming instrument is a strong signal of good faith.
- **A stated direction of error.** §II-F argues the memorisation figure is an
  *upper bound* with a known direction. Most papers cannot say which way their
  bias runs.
- **A named, reproduced defect in published code**, with the observation that it
  works *against* your own case — you are being generous to the paper you
  critique.
- **Honest reporting of a refuted hypothesis.** The chapter's original claim was
  killed by your own data and the paper says so.
