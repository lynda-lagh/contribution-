# Chapter 1 — what we inherited, what we changed, what still has to change

Written 19 Aug 2026, after a full audit of `chapter1/`, `src/routing/` and
`scripts/`. Four questions, in the order a supervisor or reviewer will ask them.

---

## 1 · What we take from KG-LLM

KG-LLM (Yao, Peng, Mao, Luo — ICASSP 2025 / arXiv 2308.13916) is the system we
reconstruct. We inherit its **recipe**, not its code, and we keep the inherited
parts byte-identical so that the one thing we vary is the only thing that moves.

| inherited | what it is | where it lives now |
|---|---|---|
| **the question format** | `Is this true: {h} {r} {t}?` → `Yes` / `No` | `chapter1/data.py::render`, prompt `P0` |
| **the instruction wrapper** | Alpaca, no-input variant | `src/data/prompts.py::ALPACA_NO_INPUT` |
| **negative construction** | `random.choice(entity_list)`, 1 negative per positive | condition **A**, `negatives="random", n_negatives=1` |
| **entity / relation verbalisation** | `entity2text.txt`, `relation2text.txt` — the same files KG-BERT shipped | `src/data/loaders.py::load_kg` |
| **the benchmarks** | WN11, FB13, WN18RR, YAGO3-10 | `scripts/fetch_data.py` |
| **LoRA adaptation** | parameter-efficient tuning of a decoder LLM | `src/train/sft.py`, r=8 α=16 on q,v |
| **the headline number to interpret** | 95.5 % triple-classification accuracy on WN11 (KG-LLaMA-7B) | quoted in §I of the paper |

Condition **A is KG-LLM's recipe**. Every other condition is A with exactly one
factor changed. That is what makes the comparison a measurement rather than a
horse race.

---

## 2 · What we changed from KG-LLM

### 2.1 The scientific changes — these are the contribution

| # | change | why it exists |
|---|---|---|
| **1** | **Condition B — anonymisation.** Every entity surface form → `entity{i}`, relations and structure untouched. | A relational rule survives this; a name→label association does not. The gap between A and B is the measurement. |
| **2** | **Condition S — permutation.** Every real name kept, only *which entity holds which* deranged. | Answers the one objection anonymisation cannot: *"`entity4471` is unreadable, so of course it fails."* Under S the vocabulary, lengths and readability are identical. **Not in any of the 188 papers surveyed.** |
| **3** | **Link prediction instead of triple classification.** Score 50 candidates by `P(Yes)` and sort. | Classification's chance level is 0.5, so a model that collapses onto one answer scores *exactly at chance while having learned nothing* — which happened in three of our conditions. The N-way protocol moves chance to `MRR = 0.0900`, an order of magnitude lower, and it actually completes something. |
| **4** | **A decomposition, not a score.** `acc − chance = format + memorisation + residual`. | Reporting `(accuracy, gap)` as a pair rather than accuracy alone. |
| **5** | **Type conditions C and G.** Does knowing *what* a thing is replace knowing *what it is called*? | CATS, Knit and RealKGC all **add** types and never **remove** names, so none of them can separate a type effect from a familiarity effect. |
| **6** | **Familiarity split.** Test triples bucketed by whether their entities were in the training sample; names left intact. | Independent of anonymisation — shares no machinery with it. It localises the binding to the **pretrained backbone** (+0.0036 balanced) rather than to fine-tuning recall. |
| **7** | **Generation-free decision rule.** `P(Yes)` vs `P(No)` at the answer token. | KG-LLM parses generated text with substring matching, which fires on *k**no**w* and *ca**nno**t* and scores a refusal as a negative. On the untuned model the choice of rule moves accuracy by 0.0700 against an above-chance signal of 0.1920. |
| **8** | **Exogenous semantic types.** WordNet / YAGO classes instead of relation-position tags. | See §2.3 — this one is a correction to *our own* earlier design. |

### 2.2 The engineering changes — machinery KG-LLM has no equivalent of

- **`chapter1/preflight.py`** — 8 checks, no fallbacks, non-zero exit. Refuses to
  start a run that cannot produce a valid number.
- **`chapter1/audit_data.py`** — 7 stages of checks on the *data* rather than the
  code. On WN11 it independently reproduces every defect our Threats section
  reports.
- **`chapter1/showcase.py`** — the model's actual answers on random queries, saved
  to JSON and LaTeX. KG-LLM's Table VI is qualitative and is one of its most-read
  parts; we had no equivalent.
- **`chapter1/check_type_leak.py`** — measures what a one-line heuristic scores on
  the type prompt *before* any model is trained.
- **`chapter1/rescore.py`** — AUC and best-threshold balanced accuracy, so a model
  that ranks correctly but sits at the wrong operating point is not mistaken for
  one that learned nothing.
- **Per-run adapter archiving** — `train_and_save()` zips after *every* run, to the
  top level of `/kaggle/working`, outside the git clone.

### 2.3 A correction to our own earlier design

**Induced types were endogenous.** `[_type_of::tail]` means *"this entity is
usually the tail of a `type of` edge"* — computed from the very edges the model
is asked to predict. The tag partly restates the question. That circularity was
measurable: **a one-line rule scored 62.4 %** on YAGO3-10 with no model at all.

Regenerating the negatives type-consistently pushed that to 51.3 %, which does
not make the tag meaningful — it makes it uninformative in both directions. So
condition C landing at chance may have been a property of the construction
rather than a finding about types.

**Now:** types come from outside the graph. YAGO was built by joining Wikipedia
(the entities) to WordNet (the classes), so every YAGO entity already carries a
WordNet class. `--require-semantic` refuses the induced fallback outright.

```
before   Is this true: entity14008 [_type_of::tail] type of entity8129 ...
after    Is this true: entity52 [actor] acted in entity4 [film]?
```

---

## 3 · What can stand in the paper as it is

These are measured, checked, and survive the audit.

| claim | number | status |
|---|---|---|
| Memorisation share, ranking | **94.0 %** | ✓ A and B are both unaffected by the S bug. Bootstrap CI **[91.5 %, 99.6 %]**, p < 0.0001. |
| Memorisation share, classification | **96.8 %** | ✓ Independent protocol, different chance level, same sign. |
| Condition C sits at chance | 0.0945 vs floor 0.0900 | ✓ Bootstrap p = 0.80 — statistically indistinguishable from chance, which is *stronger* than the original wording. |
| Familiarity split | **+0.0036** balanced | ✓ Untuned control (+0.0182) establishes the buckets are not intrinsically different. |
| Decision-rule spread | 0.0700 of 0.1920 untuned | ✓ Same generations, four parsers. |
| Calibration under S | 0.809 → 0.511 confidence | ✓ Independent signal — ranking uses order, calibration uses magnitude. |
| SMI layer 21 | 0.08239 vs 0.01683 (4.9×) | ✓ |
| Type-tag leak audit | 62.4 % → 51.3 % | ✓ Reproduced by `audit_data`. |
| WN11 label defects | 2,220 dup train · 193 self-loops · 7/10,542 bad negatives | ✓ **Independently reproduced by `audit_data.py` this session.** |
| Condition D | −10.6 points from harder negatives | ✓ |
| Condition G | −7.3 points from adding types to real names | ✓ |
| Condition E | degenerate, withheld | ✓ Honest. |

---

## 4 · What must change before submission

### 4.1 Blocking — a reported number is wrong

**① The S arm must be re-ranked.** `chapter1/rank.py` never applied
`cond.shuffle`, so condition S was scored on the **real** graph: an adapter
trained on a deranged world, tested on undamaged names. That is a train/test
mismatch, not the permuted-name control. Proof from the saved output — the S run
carries the identical query strings to A:

```
A: Stan_Collymore, playsFor, England_national_football_team
S: Stan_Collymore, playsFor, England_national_football_team    ← identical
```

`m(S) = 0.2974` and therefore the **71.5 / 22.5 / 6.0 decomposition** came from
it. That figure appears in the **abstract, Fig. 1, Table IV, the Discussion and
the Conclusion**. Fixed in code; notebook cell 5b re-runs it (~80 min, inference
only, no retraining). `evaluate.py` had the same defect, so the classification S
row (0.6030) is affected too.

**② `main.tex` §IV-A misstates the protocol.** It says *"filtered against
train ∪ valid ∪ test"*; `rank.py` filtered against **train only**. Fixed in code;
the bias was conservative (it depressed MRR in every arm), but the sentence was
not true. Re-run to match it.

**③ The pre-registration claim is too strong.** §III-C says *"the interpretation
of every outcome was fixed in the released code before the runs."* The grid
pre-registered `S ≈ B` and `S ≈ A`; the outcome that occurred — **S strictly
between them** — had no entry. The three-way decomposition was written after
seeing the result. Soften to: the A/B contrast was pre-registered, the
decomposition is exploratory and awaits confirmation on a second graph.

### 4.2 Already fixed in the text this session

- Deleted: *"across the 188 papers… no paper uses it during training"* — an
  unverifiable universal negative that KG-CF's ablation likely contradicts.
- Corrected: FLAME studies **frozen** models, not what fine-tuning installs.
- Deleted: *"the diagnostic is model-independent"* (contradicts §V-C Scale).
- Softened: *"the founding system of this subfield"* → *"an early and widely
  cited system"*.
- Deleted: *"Condition C is not merely close to B but below it"* — the bootstrap
  gives C − B = −0.027, CI [−0.063, +0.009], **not significant**.
- Corrected: the Fig. 2 caption claimed YAGO3-10 has 93.3 % `OTHER`, a figure
  that appears nowhere in the code, data or results and contradicts §V-B's own
  27.4 %. Replaced.
- Fixed: `evaluate.py` divided the memorisation share by 0.5 for **typed**
  conditions, whose chance level is the 0.513 tag floor. C: 0.8376 → **0.9649**,
  D: 0.7092 → 0.7812, G: 0.8473 → 0.9159.

### 4.3 Should change, not blocking

- **Add the bootstrap CIs to Table III.** The machinery exists in
  `chapter3/stats.py`; the per-query ranks are on disk. "Single run, no variance
  estimate" is the reviewer's easiest attack and it is already answerable.
- **Re-dump all 500 per-query ranks.** `rank.py` truncated to 200, which widened
  every interval by ~1.6× for no saving worth having. Fixed; needs a re-run.
- **Run the untuned ranking baseline** (cell 5b). Inference only. If the untuned
  model already ranks far above chance on real names and collapses without them,
  the binding lives in **Qwen2.5 itself**, not in the LoRA — which answers *"a
  1.5B model can only memorise"* by making the claim not about capacity at all.
- **Report `audit_data`'s warnings.** The 2,220 duplicate training triples and
  193 self-loops are already in Threats; the **2.2 % unseen-entity share** and the
  **shared-RNG candidate sampling** are not.
- **Re-run condition C and G with semantic types.** The current C/G numbers bound
  what *induced* types can do. §V-B hedges this in one sentence near the end —
  promote it, and drop *"a direct challenge to CATS / Knit / RealKGC"*, because
  those papers mean semantic types and we tested a different object.
- **Fill the title and author block.** Still `[TITLE — choose one]`.

### 4.4 Known and accepted, state them plainly

- One backbone, 1.5 B parameters, one seed.
- One graph carries the decomposition.
- YAGO3-10's negatives are generated, so its numbers are not comparable to
  published YAGO3-10 results.
- Candidate pools come from one shared RNG, so arms are comparable only when run
  at the same `--limit`. Chapter 3 fixed this with per-query seeds; chapter 1 was
  never back-ported.

---

## 5 · How to verify any of this yourself

```bash
python -m chapter1.test_chapter1                          # 24 unit tests, ~2 s
python -m chapter1.preflight --dataset YAGO3-10 --require-semantic
python -m chapter1.audit_data --dataset YAGO3-10          # 7 stages of data checks
python -m chapter1.check_type_leak --dataset YAGO3-10     # the tag-only floor
python -m chapter1.showcase --dataset YAGO3-10 --task both --top5
```

`preflight` asks *can this run start?*. `audit_data` asks *is the data telling the
truth?*. Both have already failed silently in this project, which is why they now
exit non-zero.
