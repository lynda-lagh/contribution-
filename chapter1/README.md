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

## 1.3 Type tags

| | |
|---|---|
| **We do** | `entity5 [Person] bornIn entity0 [Location]` |
| **Closest** | **Knit** (Big Data Mining & Analytics 9(2) 2026), Tables 2–4: *"NN denotes nouns, VB denotes verbs, RB denotes adverbs, JJ denotes adjectives"* |
| **They did** | POS tags in the instruction-tuning prompt. WN11, FB13, WN18RR, YAGO3-10 — **our exact four datasets** |
| **Why us** | Knit **adds** types with names present and reports accuracy up (0.2240 → 0.2490). It cannot tell whether the model *used* the type or just had more tokens. **We remove the names.** `anonym` appears **0 times** in Knit |
| ⚠️ **Dataset** | WN11's "types" are only parts of speech. **`Person –bornIn→ Location` is a Freebase statement.** Run the type conditions on **FB13** |

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
| ⚠️ **R12** | 50-way Hits@1 is **not** the same quantity as full-ranking Hits@1. Say "50-way, filtered" in every caption. All our comparisons are internal under one protocol |

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

**Report `% of the gap recovered`, not raw accuracy.** Ceiling = 0.9315 − 0.5385 = **39.3
points**. *"P4 recovers 12% of the gap"* is informative; *"accuracy rose to 0.588"* is not.

---

# 3 · Files

| File | Purpose |
|---|---|
| `conditions.py` | ★ the grid in one declarative place — conditions, prompts, interpretations |
| `data.py` | builds every condition; stamps seen/unseen at build time |
| `evaluate.py` | scores one checkpoint on **both** test sets → the gap |
| `rank.py` | classifier → ranker; Hits@K and **MRR**, 50-way filtered |
| `analysis.py` | gap table · seen/unseen · calibration-by-familiarity |
| `run.py` | orchestrator (`--plan` prints the whole schedule and cost) |
| `test_chapter1.py` | 17 tests, each a worked example. No GPU, ~2 s |
| `chapter1_kaggle.ipynb` | the Kaggle runner |

## Run order

```bash
python -m chapter1.test_chapter1          # 2 s, no GPU
python -m chapter1.run --plan             # schedule + cost + interpretations
python -m chapter1.data --all --dataset WN11
CUDA_VISIBLE_DEVICES=0 python -m chapter1.run --condition C --train
CUDA_VISIBLE_DEVICES=0 python -m chapter1.run --condition C --evaluate
CUDA_VISIBLE_DEVICES=0 python -m chapter1.rank --adapter checkpoints/ch1-WN11-C --condition C
python -m chapter1.analysis --dataset WN11
```

**Cost:** 4 training runs ≈ 4 GPU-h (E is ~2 h alone). Evaluation and ranking are inference.

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

# 6 · Reading before writing

| | Why |
|---|---|
| **RealKGC** (KBS 349, 2026) | States our finding as its motivation. Closest work in the corpus. **Currently unread** |
| **CATS** (AAAI 2025) | Type-aware reasoning, Qwen2-7B, 12 negatives, inductive, +7.2% MRR. Reports Qwen2-1.5B in ablations — our exact size. `anonym` = 0 |
| **Knit** (P22) | POS type tags, **our four datasets**, already cited for Table 10 |
| **Geirhos et al. 2020** | The shortcut-learning frame |
