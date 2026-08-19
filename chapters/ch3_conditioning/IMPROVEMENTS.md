# Chapter 3 — what to change, in order of value

Read from the five uploaded run files plus `ch3_analysis_YAGO3-10.json`.

```
level  train_loss  runtime s  VRAM GB  tokens (est)  skip   faithful
L0       0.0505       2196     3.87     17,866,755   0.0%      —
L1       0.0502       2400     4.11      3,080,290   0.0%    100%
L2       0.0347       4312     5.22      3,081,760   0.0%    100%
L3       0.0218       5850     6.19      2,977,580   3.9%    8.3%
L4       0.0217       5842     6.19      3,114,700   2.1%    8.2%
```

No accuracy. No eval curves — so no overfit verdict either, unlike Chapter 1.

---

## 1 · ★★★ The biggest win costs no compute: connect it to Chapter 1

Chapter 1 measured that **96.8% of above-chance accuracy on YAGO3-10 is entity
surface form**, and residual relational knowledge is 0.008 — indistinguishable
from chance.

If that is true, then **enrichment should be nearly worthless on this graph**, and
a flat L0→L4 ladder is not a null result. It is a *prediction confirmed by a
second, independent experiment*.

> The model answers by recognising entity names. Adding descriptions, types,
> neighbours and exclusion lists gives it more of what it is not using. Chapter 1
> predicts Chapter 3's flat ladder; Chapter 3 confirms Chapter 1 on different
> data with a different intervention.

Two corroborations already in hand:

| Chapter 1 | Chapter 3 |
|---|---|
| **G < A** — adding types *cost* 7.4 points when names were present | **L2** (semantic type) buys nothing over L1 |
| condition **C**'s eval loss halved vs B (0.058 vs 0.107) and test accuracy moved +0.010 | **L3/L4** train loss halves vs L0 (0.0505 → 0.0218) — expect the same non-transfer |

★ This reframes Chapter 3 from "our routing did not help" to "our routing could
not help, and Chapter 1 explains why." Same experiments, far stronger chapter.

⚠️ It requires the accuracy numbers to actually be flat. Run
`chapters.ch3_conditioning.evaluate` before committing to this framing.

---

## 2 · ★★ The control that makes any result interpretable

Chapter 1 became defensible because of condition **S** — a control that could
falsify the reading. Chapter 3 has no equivalent, and needs one:

**Random routing at a matched skip rate.** Take L3's action distribution
(3.9% skip, the rest description-only) and assign the actions *at random* to
elements instead of by the learned policy. Same token budget, same action mix,
no intelligence.

| outcome | conclusion |
|---|---|
| random ≈ learned | ★ the router's *decisions* add nothing; only the action mix matters. A strong, clean negative result |
| learned > random | the routing policy is doing real work, and you can quantify how much |

Roughly one training run. Without it, "L3 ≈ L1" is ambiguous between *granularity
does not pay* and *our particular router is bad*.

---

## 3 · ★★ Route on something with variance

The router cannot win as configured:

```
quality_bands   moderate 95.7%   rich 3.9%   poor 0.4%
has_description_rate  7.9%
skip rate       L1 0%   L2 0%   L3 3.9%   L4 2.1%
L3_usable: False
```

**95.7% of entities land in one band**, so every level routes nearly all of them
identically. The ladder is flat *by construction*, not by finding — and a
reviewer will say so.

Two features with real spread on this graph:

- **Relation** — 37 of them, EIR 19,672, wildly different semantics.
  `hasGender` needs no enrichment at all (2 possible tails); `isLocatedIn` may
  need neighbours. Routing per relation gives 37 genuine decisions instead of
  one band covering 95.7% of elements.
- **Degree percentile** — continuous and always has spread (median 10, max
  61,044). `router.py` already computes it and only uses it for two tie-breaks.

★ Switching the routing feature from *description quality* (degenerate here) to
*relation* is a small change with a large effect on whether the chapter can say
anything.

---

## 4 · ★ Ask the allocation question, not the reduction question

Right now the chapter compares 17.9M tokens against 3.1M — a strawman baseline
against a fixed policy. The interesting engineering question is different:

> **Given a fixed token budget, where should it be spent?**

Hold the budget equal across policies and vary only *where* enrichment goes.
That turns the chapter from "we removed 83% of tokens" — which is easy if you
are allowed to remove useful ones — into "at equal cost, targeted enrichment
beats uniform enrichment by X", which is a claim nobody in the corpus has made.

---

## 5 · ⚠️ Do not trust the loss drop

L0 0.0505 → L4 0.0217 looks like a 57% improvement. It is not evidence of
anything yet:

- it is **training** loss, on **different prompts**. More context makes the next
  token easier to predict whether or not the model generalises better;
- Chapter 1 already produced this exact trap. Condition **C**'s eval loss was
  0.058 against B's 0.107 — nearly halved — and the test accuracy difference was
  **+0.010**, with C landing on the tag-only floor and answering "Yes" 92% of the
  time.

**Report loss and accuracy together or not at all.**

---

## 6 · Missing instrumentation

- **No eval curves** in any `ch3-*-lora.json`, so no overfit/underfit verdict.
  Chapter 1's runs have them. These runs predate that instrumentation — re-running
  with the current `sft.py` would add the fit verdict for free.
- **No per-relation breakdown.** With one relation at 34.6% of triples and `male`
  appearing in 61,044, an aggregate can be one relation.
- **No degenerate check** until now — `evaluate.py` adds it.

---

## Suggested order

| | what | cost |
|---|---|---|
| 1 | `measure` + `evaluate` + `report` on what exists | ~25 min GPU |
| 2 | decide the framing from §1 based on whether accuracy is flat | free |
| 3 | random-routing control (§2) | 1 run |
| 4 | relation-based routing (§3) | 1–2 runs |
| 5 | equal-budget comparison (§4) | 2 runs |

Steps 1–3 are enough for a defensible chapter. Steps 4–5 are what would make it
a contribution rather than a well-measured negative result.
