# Chapter 1 — Memorisation vs Relational Knowledge

> **What does fine-tuning for KGC actually install?**
> Measured on WN11: **91% of the above-chance accuracy is entity surface form.**

---

# 0 · Why this chapter changed

The chapter was designed to test *"fine-tuning installs output format, not knowledge."*
**Our own first run refuted it.**

| | untuned | tuned |
|---|---|---|
| strict | 0.6775 | 0.9315 |
| lenient | 0.6880 | 0.9315 |
| logit | 0.6920 | 0.9315 |
| constrained | **0.7475** | 0.9310 |
| non-answers | 1.65% | 0% |

**format cost 0.0105 · format ceiling 0.0040**

KG-LLM reports untuned LLaMA-7B at **21.1 (WN11) / 9.1 (FB13)** against a chance level of
50. We get **67.75**. The difference is that KG-LLM used a **base** LLaMA-7B; we use
**Qwen2.5-1.5B-Instruct**. The founding paper's 80-point gap was largely a 2023 base model
unable to follow an output convention.

★ **Report the refutation** — it is a clean re-audit of the subfield's founding result — but
it cannot carry a thesis. What replaced it is stronger.

## The finding that replaced it

| | logit accuracy |
|---|---|
| tuned, real names | **0.9315** |
| tuned, anonymised | **0.5385** |
| chance | 0.5000 |

```
logit_above_chance   0.4315
memorisation         0.3930   ← 91%
residual knowledge   0.0385   ←  9%
```

And SMI rose **0.0105 → 0.0474** (3.5×) over the same tuning.

> ★★ **SMI says representations became label-informative. Anonymisation says that
> information is surface form. SMI alone cannot tell the two apart — anonymisation can.**
> That is the wedge against the closest prior work.

---

# 1 · Provenance — every step, its closest paper, and why we differ

## 1.1 Data format and prompt

| | |
|---|---|
| **We do** | KG-LLM's four files (`entity2text.txt`, `relation2text.txt`, `train.tsv`, `test.tsv`) and its question verbatim: `"Is this true: {h} {r} {t}?"` → `"Yes, this is true."` |
| **Closest** | **KG-LLM** (Yao, Peng, Mao & Luo, ICASSP 2025), §III-A + `instructions_WN11.py` |
| **They did** | Established the template the subfield reuses. Descriptions are KG-BERT's — the field standard |
| **Why us** | Re-auditing their result **requires their task and their format**. Any deviation makes the numbers incommensurable. Serialization is a **controlled variable**: 16 distinct representations exist across 188 papers and **not one compares two of them** |

## 1.2 Anonymisation — the memorisation control

| | |
|---|---|
| **We do** | Replace every entity surface form with `entity{i}`, keep relations. Train **and** test anonymised |
| **Closest** | **P12 / KG-CF**, ablation `−te`: *"This aims to detect data leakage… The significant performance drop confirms this issue"* |
| **They did** | Used it as a **leakage ablation on their own method**. Never decomposed a fine-tuning gain |
| **Why us** | ★ It is the only instrument in 188 papers that separates *pretraining memorisation* from *relational inference*. We turn a robustness check into a **measurement** |
| ⚠️ **Verify** | `PLAN_FINAL` records 31/188 matching contamination terms, **7 checked** — all were duplicate-node leakage, privacy anonymisation, or the inverse-relation issue. **24 unchecked.** Finish before claiming novelty |

**Worked example**

```
real:  Is this true: dog, a domesticated carnivore _hypernym mammal, a warm-blooded animal?
anon:  Is this true: entity5 _hypernym entity0?
```

Relations survive; only entity identity is destroyed. Both arms are evaluated on their
**own** test set — scoring an anonymised model against real names would measure
distribution shift, not memorisation. *(This was a real bug in the first implementation.)*

## 1.2b ★★ Condition S — shuffled names, the control that defends the chapter

**The one attack that could sink this chapter:**

> *"Replacing names with `entity4471` destroys ALL information, so of course accuracy
> collapses. That tells us nothing about memorisation."*

They would have a point. **Condition S removes it.**

Keep every real name in the graph; only **permute which entity holds which**:

```
real:      Is this true: dog, a domesticated carnivore _hypernym mammal…?
shuffled:  Is this true: metamorphic _hypernym lepiota?
anonymised: Is this true: entity5 _hypernym entity0?
```

| | preserved | destroyed |
|---|---|---|
| vocabulary · name lengths · token distribution · readability | ✅ identical | |
| name ↔ entity **binding** | | ❌ |

Verified: the permutation is a **derangement**, deterministic at a fixed seed, and the
multiset of names is unchanged. (2 of 38,588 entities keep their name — they share a
surface form with another entity; WN11's ambiguity is 22.7%.)

**Three outcomes, all interpretable:**

| | reading |
|---|---|
| **S ≈ B** | ★ the binding WAS the signal. The objection is answered *with a measurement* |
| **S ≈ A** | the model never used names; anonymisation destroyed something else — investigate |
| in between | quantifies how much is binding vs readability |

`src/data/loaders.py::shuffle_surface_forms` · **[OURS]** · 1 run, ~36 min.
**The cheapest insurance in the project.**

## 1.3 Type tags

| | |
|---|---|
| **We do** | `entity5 [Person] bornIn entity0 [Location]` |
| **Closest** | **Knit** (Big Data Mining & Analytics 9(2) 2026), Tables 2–4: *"NN denotes nouns, VB denotes verbs, RB denotes adverbs, JJ denotes adjectives"* |
| **They did** | POS tags in the instruction-tuning prompt. WN11, FB13, WN18RR, YAGO3-10 — **our exact four datasets** |
| **Why us** | Knit **adds** types with names present and reports accuracy up (0.2240 → 0.2490). It cannot tell whether the model *used* the type or just had more tokens. **We remove the names.** `anonym` appears **0 times** in Knit |

### ⚠️⚠️ The type-extraction trap — measured, and it cost us two silent bugs

`entity_types(method="auto")` reads a POS marker out of the identifier (`stool_NN_2`,
WN18RR style). **WN11 identifiers are `__east_indian_1` — no marker.**

```
WN11 raw    pos      →   0 types · OTHER 100.0%          ← unusable
WN11 raw    induced  →  22 types · OTHER 1.0% · H 3.42 bits
WN11 anon   induced  →  22 types · OTHER 1.0% · H 3.42 bits   ← identical ✓
```

Result: condition C rendered prompts **byte-identical** to B, and C/D/E/G would all have
silently run as B while the ladder "showed" that types do not help.

**Fix:** fall back to **induced types** — an entity's type is the relation positions it
occupies (`_has_instance::head`). Derived from structure, which anonymisation preserves,
so induced types are the *same* before and after — exactly what B→C needs.

⚠️ **Report what "type" means.** An induced type is *"participates in these relations"*,
weaker than a semantic type. And its concentration matters: on WN11 the top type covers
**27%** of entities, so 22 types carries less than the count suggests. Entropy (3.42 bits)
is the honest measure.

**Two guards now prevent a silent repeat:**
1. `build_types` refuses if >50% of entities are OTHER
2. every condition with `types=True` asserts a type tag actually appears in its prompts
   → `[guard] C: 200/200 prompts carry a type tag ✓`

## 1.4 Negatives

| | |
|---|---|
| **We do** | 1 random (baseline) · 1 type-consistent · 6 type-consistent |
| **Closest** | **KG-LLM** `random.choice(tmp_ent_list)`, one per positive · **RealKGC** 6 (3 head + 3 tail) · **CATS** 12 |
| **They did** | KG-LLM's uniform corruption is usually **type-violating and trivially separable** — name lookup suffices. P10 was flagged for the same practice |
| **Why us** | If negatives can be rejected on the name alone, training **never forces** rule use. Hardness (C→D) and count (D→E) are separated, which neither RealKGC nor CATS does |
| ⚠️ **Confound** | More negatives = more **instances** (1→20k, 6→70k). E differs in difficulty **and** data volume. Report instances and GPU-hours per row |

## 1.5 Seen / unseen split — the free second instrument

| | |
|---|---|
| **We do** | Split test accuracy by whether the entity appeared in the 10,000 sampled training triples |
| **Closest** | **Analyzing Bias** (Applied Sciences 2026): EIR, PopBS, *"degree imbalance, popularity bias, long-tail underrepresentation"* |
| **They did** | A survey defining the metrics. No experiments |
| **Why us** | ★★ **Independent confirmation.** Anonymisation destroys names; this leaves them intact and asks whether *familiarity* is doing the work. Two instruments agreeing from different directions is far harder to dismiss. **Costs nothing** — we trained on 10k of WN11's 112,581 triples, so roughly half the test entities were never seen. We accidentally built an inductive split |

## 1.6 Four parsers + logit scoring

| | |
|---|---|
| **We do** | Read one set of generations four ways; add generation-free P(Yes) vs P(No) |
| **Closest** | **KG-LLM** `eval_WN11_ft.py` — a **strict** rule for tuned models and a **lenient** one for untuned |
| **They did** | Knew parsing was a problem. Their lenient rule tests `res.find("no")` — unanchored substring matching that fires inside *k**no**w*, *can**no**t*, *a**no**ther* |
| **Why us** | We reproduce the defect **exactly** so the published number stays recoverable, and report `spurious_substring` and `refusal_scored_as_no` **separately** — a parser artefact and a penalised abstention are different phenomena |
| **Result** | On Qwen2.5-Instruct the pathology is nearly absent: 1.65% untuned, 0% tuned. **Use the logit parser as the default** — after tuning all four agree at 0.9315 |

## 1.7 SMI — the second instrument

| | |
|---|---|
| **We do** | Sliced mutual information between hidden states and labels, across layers |
| **Closest** | **FLAME**: *"these representations reach fine-tuned-level SMI values, indicating that **fine-tuning primarily aligns representations rather than injecting knowledge from the KG training set**"* |
| **They did** | The nearest claim to ours in the entire corpus — measured with SMI alone |
| **Why us** | ★ **Our SMI went UP 3.5× while anonymisation showed 91% was surface form.** SMI cannot separate memorisation from knowledge. This is the precise limitation of the closest prior work, and it is what our control resolves |

### ⚠️ SMI alone says the OPPOSITE of our thesis — and that is the point

Measured on WN11:

```
SMI   layer 21 (untuned)  0.01052
      layer 28 (tuned)    0.04742        +3.5×
interpretation printed:   "representation enriched -> tuning installed KNOWLEDGE"
```

Read on its own, that **contradicts** the memorisation claim. Read beside the gap it
sharpens it:

> SMI rose **3.5×** *while* **91%** of the above-chance accuracy is surface form.
> ★ **Representations became more label-informative, and the information they encode
> is the entity NAME.** SMI cannot tell the two apart. Anonymisation can.

**That is the contribution against FLAME**, and it only exists because the two instruments
are reported *together*. Either alone is misleading — which is why `evaluate.py` emits a
`joint_reading` field and flags the case where the two point different ways.

```bash
CUDA_VISIBLE_DEVICES=0 python -m chapter1.run --evaluate --condition A --smi
```

⚠️ Slow: 600 samples × 2 model loads per condition, so it is **off by default**.
**Run it on A and B at minimum** — those two carry the comparison.

---

## ★ Three instruments, three mechanisms, one claim

| | what it does | what it destroys | costs |
|---|---|---|---|
| **anonymisation** (A→B) | replaces names with opaque ids | *all* surface information | 1 run |
| **shuffled names** (S) | permutes real names across entities | only the name↔entity **binding** | 1 run |
| **seen / unseen** | splits by training familiarity | **nothing** — names stay intact | **free** |
| **SMI** | reads the representations directly | — | slow, no training |

★ They fail in different ways, so agreement between them is evidence rather than
repetition. And SMI arriving at a *superficially opposite* answer is what makes the joint
reading a finding instead of a redundancy.

## 1.8 Prompt ablation P0–P4

| | |
|---|---|
| **We do** | Same checkpoints, five prompts. P0 bare · P1 type tags · P2 instruction · P3 both · P4 demonstrations |
| **Closest** | **RealKGC** soft constraints · **CATS** type-aware module · **Knit** POS tags |
| **They did** | RealKGC's abstract states our finding as its **motivation**: *"supervised finetuning inadvertently encourages models to circumvent genuine reasoning by **exploiting statistical shortcuts** between graph contexts and labels."* It asserts; nobody measures. Its actual mechanism is **prompt text**, not a loss term: the total loss is plain cross-entropy over two prompt views, `L = −(1/B) Σ log[P(y\|x_RSD)·P(y\|x_SCR)]` |
| **Why P4** | RealKGC's type constraint works by **showing other triples that use the same relation**, so the model compares head/tail types against real instances. P1 *states* the type; **P4 demonstrates it.** If P4 ≫ P1, demonstration beats declaration |
| ⚠️ **P2/P3 are inert on the tuned model** | Loss is masked to the response and the response is the fixed string `"Yes, this is true."` Step-by-step instruction-following was **tuned out**. Only the untuned arm can react — and *that contrast is itself a finding*: fine-tuning for KGC destroys the ability to use structural instructions |
| ⚠️ **Only increases are interpretable** | The tuned model saw only P0. A drop under P1–P4 may be distribution shift |

## 1.9 Link prediction by ranking

| | |
|---|---|
| **We do** | For `(h, r, ?)`, score every candidate by `P(Yes | h, r, t)` and sort. 50-way, filtered |
| **Closest** | **KG-BERT** ranks entities by its classification score. **KICGPT** and **ColKGC** rerank on the same principle |
| **They did** | Standard protocol; not our invention |
| **Why us** | Triple classification **completes nothing**, and the thesis is about *enrichissement*. This converts the classifier into a completion system with **no retraining** |
| ★★ **It removes a stated limitation** | The spec says repeatedly *"MRR is NOT computable: sampling yields a SET, not a ranking."* True for generative decoding. **Scoring candidates produces an ordering, so MRR is computable** |
| ✅ **50-way is the field standard, not a compromise** | **RealKGC**, §4.1: *"Following standard practice like CATS and ToC, we rank each answer tail (or head) entity against **50 randomly sampled negative entities**."* Both of our closest competitors use exactly this, and RealKGC adopts CATS's splits for direct comparability. Cite it as *"50-way, following CATS and RealKGC"* — the R12 worry mostly dissolves |

## 1.10 ★ What the four key papers actually do — read, not assumed

### RealKGC (Knowledge-Based Systems 349, 2026) — the closest, and an **ally**

It does **not** merely assert the shortcut. **RQ6 measures one**, with a different
instrument from ours:

> *"Although these injected triples seemingly provide a structural connection between h
> and t, they are **purely synthesized based on statistical frequency** rather than any
> valid logical entailment. This setup allows us to examine whether LLMs prioritize
> **spurious statistical patterns** over genuine semantic verification."*

They inject synthetic high- vs low-frequency relational context and report **False
Positive Rate**. High-frequency injection spikes FPR for every other LLM method; RealKGC
stays low.

★★ **It is a DIFFERENT shortcut from ours, and that is the opening.**

| | shortcut measured | instrument | metric |
|---|---|---|---|
| **RealKGC** | high-frequency **relation co-occurrence** | inject synthetic context | FPR |
| **Ours** | **entity surface-form** memorisation | anonymisation | accuracy gap |

They ask *does the model over-trust frequent relation patterns?*
We ask *does the model know anything once the names are gone?*
**Neither instrument can see the other's shortcut.**

Their mechanism is prompt text, not a loss term — total loss is plain cross-entropy over
two prompt views, `L = −(1/B) Σ log[P(y|x_RSD)·P(y|x_SCR)]`. Setup: LoRA r=16 α=32,
**lr 1e-5**, batch 8, **1 epoch**, **6 negatives** (3 head + 3 tail). Backbones include
**Qwen2-1.5B** — our exact size.

### CATS (AAAI 2025)

Type-aware reasoning + subgraph reasoning, Qwen2-7B + LoRA, **12 negatives**, inductive,
+7.2% average MRR over 18 settings. Ablation is module-level (TAR / SR / full).
`anonym`, `memoris`, `surface form`, `entity name` — **0 hits each.**

### Knit (Big Data Mining & Analytics 9(2), 2026) — the type prompt, verbatim

> *"…taking into account the **attribute classification** of the other entity, where
> **NN denotes nouns, VB denotes verbs, RB denotes adverbs and JJ denotes adjectives**."*

Plus a **KG-integrated information adapter** injecting `<head embedding>` /
`<relation embedding>` beside the text. So Knit is our P1 **and** soft structural
injection. ⚠️ It also confirms that WordNet-style "types" are **parts of speech**, not
`Person`/`Location` — which is why the type conditions belong on **FB13**.

### Geirhos et al., *Nature Machine Intelligence* 2(11):665–673, 2020 — the frame

**Shortcut learning** = *"decision rules that perform well on standard benchmarks but fail
to transfer to more challenging testing conditions."* Their toy case: a network classifying
stars vs moons by **location**, because position correlated with class in training.

That is our result exactly — entity name correlates with label, so the model uses the name.
★ **Neither RealKGC nor CATS cites this literature.** Connecting KGC to it is ours.

### The positioning sentence

> RealKGC shows that LLM-based KGC is vulnerable to high-frequency relational
> co-occurrence, measured by injected context and FPR. We show the same systems are
> vulnerable to a **second, independent** shortcut — entity surface-form memorisation —
> measured by anonymisation and **invisible to their instrument**. Both are instances of
> shortcut learning (Geirhos et al., 2020).

---

# 2 · The grid

## Training conditions — one variable per step

| ID | Names | Types | Negatives | Instances | Isolates |
|---|---|---|---|---|---|
| **A** | Real | ✗ | 1 random | 20,000 | baseline = KG-LLM |
| **B** | Anon | ✗ | 1 random | 20,000 | **entity names** |
| **C** | Anon | ✓ | 1 random | 20,000 | ★ **can types substitute for names?** |
| **D** | Anon | ✓ | 1 hard | 20,000 | negative **hardness** |
| **E** | Anon | ✓ | 6 hard | 70,000 | negative **count** |
| **G** | Real | ✓ | 1 random | 20,000 | ★★ **do types help when names exist?** |
| **S** | **Shuffled** | ✗ | 1 random | 20,000 | ★★ **the control that defends the chapter** (§1.2b) |

**F (12 hard) dropped.** Matches CATS, but CATS is 7B and inductive. ~5 GPU-h to answer
"does 12 beat 6?" — not a question this thesis asks. Spend it on **G**.

## Reporting: three columns, never one

| cond | acc real | acc anon | **gap** |
|---|---|---|---|

**The gap is the contribution.** A model with high real accuracy and a large gap is a
memoriser. One with slightly lower accuracy and a small gap has learned something
transferable — and that trade-off must be **shown**, not assumed.

⚠️ **P11**: at 10% noise, *every* cleaner tested made KGC **worse than doing nothing**.
If every constrained condition loses on both columns, report that.

## Pre-registered interpretations

| outcome | conclusion |
|---|---|
| `C ≈ B` | types cannot substitute for names — LoRA at 1.5B installs no type rule |
| `C ≫ B` | the model **can** use types and simply does not when names are easier |
| `D > C` | it is negative **hardness** that forces rule use, not the type text |
| `E > D` | negative **count** matters — ⚠️ confounded with 3.5× instances |
| `G ≈ A` | ★ types add nothing when names are available — a direct challenge to CATS / Knit / RealKGC, whose gains may be token volume |
| `G > A` | types add real signal; the anonymised ladder then says what kind |
| **`S ≈ B`** | ★★ the name↔entity **binding** was the signal — the "anonymisation destroys everything" objection is answered with a measurement |
| **`S ≈ A`** | the model never used names at all; anonymisation broke something else. **Investigate before publishing** — this would undermine the B result |

---

## ★★ Is this a contribution whatever the results?

**Mostly yes — but because of design choices, not automatically.**

### Already banked, outcome-independent

| | |
|---|---|
| the memorisation decomposition | **0.9315 → 0.5385**, 91% of above-chance accuracy in surface form. Measured. No future run removes it |
| the SMI result **against FLAME** | SMI **+3.5×** while the gap is 0.393 → SMI cannot separate memorisation from knowledge. Measured, and it corrects the closest prior work |
| the KG-LLM re-audit | untuned **67.75** vs their reported 21.1 / 9.1 — a correction to the subfield's founding result, verifiable in an afternoon |
| the evaluator defect | `res.find("no")` firing inside *k**no**w* / *can**no**t*, in published code, unremarked |
| **the instrument** | a protocol separating memorisation from relational knowledge, where the field had none. Methods outlive the numbers they produce |

### The single point of failure

**Everything rests on one instrument, and it has one obvious attack** (§1.2b). Two things
close it, and both are cheap:

1. **`seen_unseen.py`** — names intact, asks only whether *familiarity* drives accuracy.
   **Zero GPU.** Different mechanism, same conclusion.
2. **Condition S** — real names, permuted. **One run.**

> **Two instruments agreeing from opposite directions is a different class of claim from
> one instrument asserting.** Without them you have one instrument and one objection,
> aimed at each other.

### Two things a jury will press on

**Transductive memorisation is arguably legitimate.** If the graph is fixed, memorising is
a correct strategy. The enrichment framing is the defence — but it is an *argument* until
`seen_unseen` makes it a *measurement*: the "neither seen" bucket **is** the enrichment case.

**1.5B invites "it is a scale problem."** Say explicitly that **CATS and RealKGC both
report Qwen2-1.5B in their ablations**, so the comparison is size-matched by their own
choice — otherwise a reader assumes 7B was unaffordable.

**Report `% of the gap recovered`, not raw accuracy.** Ceiling = 0.9315 − 0.5385 = **39.3
points**. *"P4 recovers 12% of the gap"* is informative; *"accuracy rose to 0.588"* is not.

---

# 3 · Files

| File | Purpose | GPU |
|---|---|---|
| `conditions.py` | ★ the grid in one declarative place — conditions, prompts, interpretations | — |
| **`profile_data.py`** | ★ **understand a dataset BEFORE designing on it** — ids · descriptions · relation balance · which type source works and how concentrated · ambiguity · coverage by training size. Ends with a **verdict** on which conditions are viable | — |
| **`validate.py`** | ★ **hard checks, non-zero exit** — run after ANY dataset modification. Labels · leakage · unknown ids · negatives that are actually true | — |
| **`seen_unseen.py`** | ★★ **the second instrument** — reads an EXISTING result, no retraining | — |
| `data.py` | builds every condition; two guards against silent no-ops | — |
| `evaluate.py` | scores one checkpoint on **both** test sets → the gap | ✓ |
| `report.py` | confusion matrix · per-class P/R/F1 · **degenerate check** · majority baseline · per-relation · ECE/Brier · risk–coverage · McNemar | — |
| `rank.py` | classifier → ranker; Hits@K and **MRR**, 50-way filtered | ✓ |
| `analysis.py` | gap table · seen/unseen · calibration-by-familiarity | — |
| `run.py` | orchestrator (`--plan` prints schedule, cost, interpretations) | — |
| `test_chapter1.py` | **22 tests, a different random graph each run** (seed printed) | — |
| `chapter1_kaggle.ipynb` | the Kaggle runner | |
| `../scripts/make_test_negatives.py` | ★ generate ±1 labels for YAGO3-10 / WN18RR | — |

## Run order

```bash
# ── free, no GPU ───────────────────────────────────────────────────────────
python -m chapter1.test_chapter1                    # 22 tests, ~2 s
python -m chapter1.seen_unseen --dataset WN11       # ★★ the second instrument
python -m chapter1.profile_data --dataset YAGO3-10 WN11
python -m chapter1.validate    --dataset YAGO3-10 WN11
python -m chapter1.run --plan

# ── data ───────────────────────────────────────────────────────────────────
python -m scripts.fetch_data --datasets YAGO3-10 WN11
python -m scripts.make_test_negatives --dataset YAGO3-10   # YAGO ships no ±1
python -m chapter1.validate --dataset YAGO3-10             # gate on this
python -m chapter1.data --all --dataset YAGO3-10

# ── train · evaluate · rank ────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python -m chapter1.run --condition C --train
CUDA_VISIBLE_DEVICES=0 python -m chapter1.run --condition C --evaluate
CUDA_VISIBLE_DEVICES=0 python -m chapter1.run --condition A --evaluate --smi   # ★ A and B
CUDA_VISIBLE_DEVICES=0 python -m chapter1.rank --adapter checkpoints/ch1-YAGO3-10-C --condition C
python -m chapter1.analysis --dataset YAGO3-10
python -m chapter1.report   --dataset YAGO3-10
```

**Cost:** 5 training runs ≈ 5 GPU-h (E is ~2 h alone). Everything else is inference or free.
★ **Order: `seen_unseen` → C, G, S → D, E.** The first is free; the next three carry the claim.

---

# 3b · Datasets — which graph answers which question

| | WN11 | FB13 | **YAGO3-10** |
|---|---|---|---|
| entities | 38,588 | 75,043 | **123,182** |
| ±1 test labels | ✅ shipped | ✅ shipped | ❌ **we generate them** |
| relations | lexical (`_type_of`) | mixed | ★ **typed** (`wasBornIn`, `playsFor`) |
| `Person –bornIn→ Location` | absent | partial | **native** |
| induced types ≈ semantic types | ✗ | partial | **✓** |
| descriptions | 1 word (bare names) | — | — |
| ambiguity | **22.7%** (`break` ×32) | — | — |
| memorising means | **lexical** — *"dog is a mammal"* | world-factual | **world-factual** |

★ **YAGO3-10 is primary.** Its 37 relations are strongly typed —
`wasBornIn` Person→Place, `isCitizenOf` Person→Country, `playsFor` Person→Club — which is
the structure the type/rule question needs and which WN11's lexical relations lack. On a
typed graph an induced type **recovers** the semantic type.

⚠️ **YAGO3-10 negatives are OURS, not the benchmark's.** `make_test_negatives` filters
every candidate against train ∪ valid ∪ test and reports how many were rejected — that
count is the closed-world exposure. Numbers are **not comparable to published YAGO3-10
results**; all comparisons are internal, and every caption must say so.

**Running WN11 as well is a decomposition, not a replication:** lexical vs world-factual
memorisation, and 3.2× the vocabulary. Worth 6 runs only if time allows — one dataset done
properly beats two done partially. WN11 A/B already exist (0.9315 / 0.5385).

★ **3 seeds on B vs C.** That contrast carries the central claim; a one-seed difference is
not defensible. `seed_variance` prints the CI half-width.

---

# 4 · Environment — learned the hard way

| | |
|---|---|
| **One GPU per job** | Two visible → HF Trainer wraps in `DataParallel` → autocast never reaches the replicas → `mat1 and mat2 must have the same dtype`. Pin with `CUDA_VISIBLE_DEVICES=0` |
| **fp16 + `sdpa`** | fp16 + `eager` returns **NaN** on Qwen2.5 — surfaces as `train_loss=0.0`, `grad_norm=nan`, which looks like a finished run. Measured: fp16/eager non-finite, fp16/sdpa finite at 3.12 GB |
| **Remove `torchao`** | Kaggle ships 0.10.0; transformers refuses to import below 0.16, killing LoRA. Nothing here uses it |
| **Pin `transformers==4.57.6`** | Bare `pip install -U peft` pulls 5.x, which breaks torchao and drops bitsandbytes |
| **Chapter 1 needs only LoRA** | No MoRA fork, no BOFT kernel — **none** of the conflicts that blocked Chapters 2 and 3 |

---

# 5 · What we do NOT claim

| | Why not |
|---|---|
| "Shortcuts are wrong" | In a transductive setting memorisation is legitimate and effective. Our claim is that it is **not what the literature reports itself as doing**, is invisible under standard evaluation, and does not transfer to enrichment |
| "The model learned the rules" | Even if P3/P4 improve, that shows the model can *exploit* structure when guided — not that tuning installed a rule |
| "Constrained models are better" | Must be measured. See P11 |
| Data efficiency | Answered — **FLAME** 0.6%→97%, **COSIGN** ~40%, **GLR** §5.4 |
| Constrained decoding as novel | Refuted in print by **MKGL** §3.5 |

**Deferred:** training on *rationales* rather than the fixed string `"Yes, this is true."` —
that is CoT distillation and a different project. Name it as future work.

**Framing for the paper**

> We do not claim that surface-form memorisation is an error. In a transductive setting it
> is a legitimate and highly effective strategy. We claim that it is **not what the
> literature reports itself as doing**, that it is invisible under standard evaluation, and
> that it does not transfer to the enrichment setting these systems are proposed for.

★ Cite **shortcut learning** (Geirhos et al., *Nature Machine Intelligence*, 2020). It
places the finding in an established tradition rather than as a complaint about one
benchmark. RealKGC gestures at the same idea and cites nothing.

---

# 6 · ★★ Relation to KG-LLM — the assigned paper

> **"My supervisor asked me to work on KG-LLM. Is this still that?"**
> **Yes — more so than a reimplementation would be. KG-LLM is not our baseline; it is our
> OBJECT OF STUDY.**

## KG-LLM's own claim, and the half it never tested

Yao et al. write that instruction tuning teaches the model

> *"to answer like training triples"* **and** *"to be more aware of a fact"*

**The first clause they demonstrate. The second they assert.** Their headline numbers are
untuned **21.1** (WN11) / **9.1** (FB13) → tuned **89.2**, and that ~80-point movement is
read by the field as fact-awareness. **Nobody decomposed it.** Chapter 1 is the
decomposition — i.e. the experiment KG-LLM's own sentence implies and does not run.

## Everything we inherit, deliberately

| Inherited verbatim | Why not change it |
|---|---|
| the four-file data format (`entity2text` / `relation2text` / `train.tsv` / `test.tsv`) | KG-BERT descriptions, the field standard |
| the prompt `"Is this true: {h} {r} {t}?"` → `"Yes, this is true."` | changing it makes our numbers incommensurable with theirs |
| triple classification on **WN11 / FB13** | their primary reported task — a re-audit must use it |
| LoRA r=8, α=16, dropout 0.05, `[q_proj, v_proj]`, lr 3e-4 | their exact constants |
| loss masked to the response only | ★ this is *why* Chapter 1 exists |
| adapter-only checkpointing | their `state_dict` monkey-patch |

## What we change, and the reason for each

| Changed | Reason |
|---|---|
| Qwen2.5-1.5B-Instruct, not LLaMA-7B base | compute — **and it produced our first finding** |
| effective batch **32**, not 128 | 128 was tuned for 10⁵ triples; at 20k instances warmup would be 32% of training |
| dynamic padding, cutoff **512**, not `max_length` at 50 | theirs silently truncates |
| **+ anonymised** variant | the control they never ran |
| **+ type tags · hard negatives · 6 negatives** | the axes their `random.choice` fixes at one setting |
| **+ 4 parsers, logit scoring, ranking** | their evaluator is a single string match |

## Four results that are statements *about KG-LLM*

**1 · Their untuned number is a model artefact, not a knowledge failure.**
KG-LLM reports 21.1 / 9.1 against a chance level of 50. We get **67.75** with an *Instruct*
model. Format cost **0.0105**, non-answers **1.65%**. The ~80-point gap was largely a 2023
**base** model unable to follow an output convention — a clean re-audit of the subfield's
founding result.

**2 · Their evaluator has a documented defect, and it strengthens their own argument.**
`eval_WN11_ft.py`'s lenient rule tests `res.find("no")` — unanchored substring matching
that fires inside *k**no**w*, *can**no**t*, *a**no**ther*. Because it is an `if/elif`,
every negative-labelled example containing that substring scores correct. **The authors
were being generous to the untuned model and it still scored five times below chance.**
That is a verifiable methodological finding sitting in published code.

**3 · The second half of their claim does not hold.**
Tuned **0.9315** → anonymised **0.5385**. Memorisation **0.393**, residual knowledge
**0.0385**. The model was made *"aware of a fact"* only in the sense of learning
entity-name → label lookup.

**4 · Their negative sampling is why.**
`random.choice(tmp_ent_list)`, one per positive, produces mostly **type-violating**
negatives that are separable on the name alone — so training never forces rule use.
Conditions D and E test exactly that, against RealKGC's 6 and CATS's 12.

## One paragraph for your supervisor

> I took KG-LLM as the object of study rather than as a baseline to beat. I reproduced its
> pipeline — same data format, same prompt, same task, same LoRA constants — and then ran
> the control it never ran: entity anonymisation. KG-LLM claims instruction tuning teaches
> the model "to answer like training triples" and "to be more aware of a fact." The first
> is demonstrated in their paper; the second is asserted. My results show that 91% of the
> above-chance accuracy disappears when entity names are replaced by opaque identifiers, so
> what tuning installed is surface-form memorisation rather than relational knowledge. I
> also found that their untuned baseline of 21.1/9.1 is an artefact of using a base rather
> than an instruction-tuned model — a modern Instruct model scores 67.75 untuned — and that
> their lenient evaluator contains a substring-matching bug that, if anything, *understates*
> their own case. The contribution is a diagnostic instrument for LLM-based KGC, applied
> first to the paper that founded the subfield.

★ **This is a stronger use of the assignment than reimplementing it.** A reimplementation
reproduces a number; this explains what the number was measuring.

---

# 7 · Reading before writing

| | Status | What it gave us |
|---|---|---|
| **RealKGC** (KBS 349, 2026) | ✅ **read** | RQ6 measures a *different* shortcut (co-occurrence → FPR). 50-way protocol. 6 negatives. Qwen2-1.5B in ablations. → §1.10 |
| **CATS** (AAAI 2025) | ✅ **read** | Type-aware module, 12 negatives, inductive, +7.2% MRR. `anonym`=0. Supplies the splits RealKGC adopts |
| **Knit** (P22) | ✅ **read** | The type prompt verbatim (NN/VB/RB/JJ) + KG-integrated adapter. Confirms WordNet "types" are POS, so type work belongs on **FB13** |
| **Geirhos et al. 2020** | ✅ **read** | *Nat. Mach. Intell.* 2(11):665–673. Shortcut learning; the stars/moons-by-location case. **Neither RealKGC nor CATS cites this literature** |

**Still outstanding:** the **24 unchecked contamination papers** (§1.2). `PLAN_FINAL` says
*"Verify before claiming it."* The novelty of the anonymisation control rests on it.
