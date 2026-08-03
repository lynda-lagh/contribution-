# Project Guide — read this first

A plain-language walkthrough of the whole project: what it asks, why each piece exists, and exactly what to run, in order.

No prior context needed. Where a step has a trap in it, the trap is named.

---

## 1. The one-paragraph version

You are testing **what actually happens when you fine-tune a language model to complete a knowledge graph**.

A knowledge graph is facts stored as triples: `(Paris, capital_of, France)`. "Knowledge graph completion" (KGC) means judging or predicting missing pieces. A prior paper, **KG-LLM**, fine-tuned large models to do this and reported big gains. This project asks a sharper question: **when accuracy goes up, what was actually bought?** Did the model *learn facts*, or did it just *learn to answer in the expected format*?

That question splits into four chapters:

| Ch | Question in plain words | Training runs |
|---|---|---|
| **1** | Did fine-tuning teach it **facts**, or just **how to phrase answers**? | 2 |
| **2** | Is **LoRA's low rank** the thing stopping it from memorising facts? | ~22 |
| **3** | How much **extra context** in the prompt is worth paying for? | 6 |
| **4** | Can it know **when to shut up**, and can we trust its stated reason? | 0 (inference only) |

**~32 training runs total.** Chapter 4 trains nothing — it re-reads checkpoints from Chapters 2 and 3.

**Model:** Qwen2.5-1.5B-Instruct. **Hardware:** Kaggle, 2× Tesla T4.

---

## 2. Vocabulary (skip if you know it)

| Term | What it means here |
|---|---|
| **Triple** | One fact: `head → relation → tail`, e.g. `(Paris, capital_of, France)` |
| **WN11 / FB13 / WN18RR / YAGO3-10** | The four knowledge graphs used. WN11 and FB13 are **binary** — each test item is one triple labelled true (`1`) or false (`-1`) |
| **Instance** | One training example. Each triple produces **2**: one true, one false. 10,000 triples → 20,000 instances |
| **Fine-tuning / SFT** | Supervised fine-tuning — showing the model examples and nudging its weights |
| **PEFT** | Parameter-Efficient Fine-Tuning: train a tiny add-on instead of all 1.5B weights |
| **LoRA** | The standard PEFT method. Adds a **low-rank** update. ~1.09M trainable params (0.07%) |
| **MoRA** | LoRA rival. Same budget, **square** matrix instead of low-rank. Its paper claims low rank hurts *memorisation* — exactly Chapter 2's question |
| **BOFT** | Different mechanism: **multiplies** weights by an orthogonal rotation instead of adding to them. Pitched as knowledge-*preserving* |
| **Probe** | A frozen control: no training at all, just a classifier on the model's internal activations. If a probe matches fine-tuning, fine-tuning added nothing |
| **DPO** | Preference training — learns from (better, worse) answer pairs rather than one gold answer |
| **Adapter / checkpoint** | The trained add-on file, ~20–100 MB (not the whole model) |
| **Anonymised** | Every entity name replaced with `entity0`, `entity1`, … Kills any advantage from having memorised real names during pretraining |
| **Logit** | The model's raw score for a token, before it picks words. Reading logits tells you what it *believed* even when it *said* something malformed |

---

## 3. The idea that holds the whole thesis together

KG-LLM reported that an **untuned** model scored **21.1%** and **9.1%** on binary tasks.

Binary means **coin-flip = 50%**. Scoring 9% is not ignorance — a genuinely clueless model lands near 50%. To score 9%, you have to be reliably *wrong*, which is nearly as informative as being right. The likelier explanation: the model **knew things but answered in a format the scoring script couldn't read**, and the script scored unreadable as wrong.

So the project separates two things that accuracy alone blends together:

- **Format** — can the model emit `"Yes, this is true."` in the exact expected shape?
- **Knowledge** — does it actually know whether the fact is true?

Everything downstream is built to keep those apart.

---

## 4. The setup you only do once

### 4.1 Get the data

```bash
python -m scripts.fetch_data --datasets WN11 FB13
```

Downloads from KG-LLM's repo into `data/`. Each dataset is four files:

```
data/WN11/
    entity2text.txt      entity_id  <TAB>  human-readable name
    relation2text.txt    relation_id <TAB>  human-readable name
    train.tsv            head <TAB> relation <TAB> tail
    test.tsv             head <TAB> relation <TAB> tail <TAB> label   (1 or -1)
```

Add `WN18RR YAGO3-10` before Chapter 2 — YAGO3-10 is ~1M triples and slow, so don't fetch it until needed.

> **Kaggle tip:** upload these once as a Kaggle Dataset named `kgc-data` and attach it via *Add Input*. Re-downloading every session burns quota, and commit runs have no internet.

### 4.2 Push code to GitHub

The Kaggle notebooks **clone from GitHub** — they do not read your local folder. So:

```bash
git add -A && git commit -m "wip" && git push
```

**Any local edit that you haven't pushed does not exist as far as Kaggle is concerned.** This is the single most common way to waste a session.

### 4.3 Kaggle notebook settings

| Setting | Value | Why |
|---|---|---|
| Accelerator | **GPU T4 ×2** | Needs tensor cores (SM 7.5). Not P100, not TPU |
| Internet | **ON** | For `pip install` and the model download |
| Persistence | Variables and Files | Keeps `/kaggle/working` between sessions |

**If you forget the accelerator, everything fails** with `Torch not compiled with CUDA enabled`. That single mistake produces dozens of unrelated-looking errors.

---

## 5. The two-session split — the thing that confuses everyone

**MoRA and BOFT cannot be installed at the same time.**

- MoRA exists only in a **fork**, `kongds/MoRA`, built on **peft 0.9.0**
- BOFT is in **official peft**, and did not exist in 0.9.0
- Both install under the import name `peft`, so installing one **overwrites** the other

There is no clever fix. The project splits into two Kaggle sessions:

| | **Session A** — `kaggle_session_A_official.ipynb` | **Session B** — `kaggle_session_B_mora.ipynb` |
|---|---|---|
| peft | official | the fork |
| Can run | `lora`, `boft`, `probe`, `dpo` | `mora`, `lora` |
| Chapters | 1, 3, 4, part of 2 | part of 2 |

**Never change `ENV` inside a notebook.** That's the entire reason there are two files.

### The cross-environment control

Two different peft versions is a **confound**. A reviewer can fairly ask: *is MoRA's margin a property of MoRA, or of peft 0.9.0?*

The answer: **run LoRA in both sessions with identical settings.**

```bash
python -m chapters.ch2_adaptation.run \
    --dataset YAGO3-10 --triples 10000 --peft lora --entities 123182 --seed 42
```

If the two LoRA numbers agree within seed noise, the library version isn't driving anything and the three-way comparison stands. Afterwards:

```bash
python -m scripts.verify_env_control
```

Every result JSON is automatically **stamped** with the environment that produced it, so this check works after the fact. One extra run turns an unavoidable obstacle into a reported control.

---

## 6. Step 0 — the smoke test (always run this first)

```bash
!CUDA_VISIBLE_DEVICES=0 python -m scripts.smoke_test
```

~5 minutes. It exists so you find out **now**, not 11 hours into a 12-hour session, that something is broken. It checks:

GPU present · bf16 reality-check · bitsandbytes · transformers API names · library versions match across sessions · which peft environment this is · tokenizer has a pad token · **which dtype produces finite logits** · LoRA trains · MoRA trains · BOFT trains · logit scoring works.

### Reading the output

| Marker | Meaning |
|---|---|
| `[OK]` | Passed |
| `[n/a]` | **Expected** to fail in this environment. Not a problem. No traceback |
| `[FAIL]` | Real. Fix before running anything long |

In **Session A**, MoRA always shows `[n/a]` — it lives in Session B. That is correct.

### Two findings baked into the config

**1. `attn_implementation` is a correctness setting, not a speed setting.** The smoke test measures it:

```
dtype  attn    finite  max|logit|  VRAM GB
fp16   eager    False        nan      3.11   <-- NaN. Silent killer.
fp16   sdpa      True       27.2      3.12   <-- what we use
fp32   eager     True       27.3      6.22
fp32   sdpa      True       27.3      6.23
```

`fp16 + eager` returns **NaN on the first forward pass**. That surfaces as `train_loss=0.0` with `grad_norm=nan` — which looks like a *finished* run, not a broken one. This is why the smoke test rejects a loss of exactly zero.

**2. The T4 has no real bf16.** `torch.cuda.is_bf16_supported()` returns `True` because it counts *emulation*. Ignore it. Use fp16.

---

## 7. Step 0b — the BOFT CUDA kernel patch (Session A only)

**Symptom:** a wall of `nvcc` errors, then BOFT quietly running a different method than you asked for.

**Cause:** peft 0.19.1 ships `fbd_cuda_kernel.cu` written against an old PyTorch C++ API. Lines 66 and 97 pass `Tensor.type()` where current torch requires `.scalar_type()`. The compile fails, and peft's response is to **silently set `boft_n_butterfly_factor` from 2 down to 1**.

**Why that matters:** the butterfly factor is BOFT's *structural* parameter. Silently running 1 instead of 2 means the method you report is not the method that ran. So `src/train/sft.py` **refuses** and raises instead of continuing.

**The fix** — notebook section `0b`, which patches the two words and verifies the compile:

```python
import pathlib, shutil, subprocess, sys, peft
cu = pathlib.Path(peft.__file__).parent / "tuners/boft/fbd/fbd_cuda_kernel.cu"
src = cu.read_text()
new = (src.replace("input.type()", "input.scalar_type()")
          .replace("grad_output.type()", "grad_output.scalar_type()"))
cu.write_text(new)
shutil.rmtree("/root/.cache/torch_extensions", ignore_errors=True)
# ... then verifies via peft's own get_fbd_cuda()
```

| Patch cell says | Do this |
|---|---|
| `BUILD OK` | Set `boft_n_butterfly_factor: 2` in `configs/base.yaml` — BOFT as published, no limitation to report |
| `BUILD STILL FAILS` | Leave it at `1`, and **report** that BOFT ran single-butterfly-stage |

**Status: the patch worked (2026-08-02).** Config is at `2`. Confirmed by trainable params going **0.39M → 0.74M** — a second butterfly stage really is present.

> ⚠️ It edits `site-packages`, so it **must re-run in every fresh session**, before the smoke test. Any BOFT result produced before this patch ran at factor 1 — **discard those, don't mix them in**.

---

## 8. Step 1 — build the instruction data

```bash
python -m src.data.build_instructions --dataset WN11 --n_triples 10000 --seed 42
python -m src.data.build_instructions --dataset WN11 --n_triples 10000 --seed 42 --anonymise
```

Fast, no GPU. Turns triples into prompt/answer pairs and writes `data/WN11/built/` and `data/WN11-anon/built/`.

**What happens inside:**

1. **Sample** 10,000 triples, stratified by relation, min 10 per relation so rare relations survive
2. **Generate a negative** for each — swap the tail for a wrong entity. Default `random` (KG-LLM's baseline); alternatives `type_consistent` and `kge_near_miss` are harder
3. **Wrap in the alpaca prompt template**, with the entity description included
4. Result: **10,000 triples → 20,000 instances** (1 positive + 1 negative each). **Report both numbers** — they measure different things

**`--anonymise` is the contamination control (KG-CF).** It replaces every entity name with `entity0`, `entity1`, … Any accuracy that survives cannot come from the model having memorised real names during pretraining.

> Verified: anonymisation replaces the whole `ent2txt` map, and every prompt surface form reads through `ent2txt`. So entity *descriptions* are anonymised too — no leak.

**Sanity numbers from the real run:** `n_relations: 11`, `edge_imbalance_ratio: 28.96` (commonest relation appears 3157 times, rarest 109). That skew is large and will matter when reading per-relation results.

---

## 9. Chapter 1 — format or knowledge?

**Two training runs**, then everything else is inference.

```bash
# train (run these two in parallel, one per T4)
python -m chapters.ch1_diagnostic.run --dataset WN11
python -m chapters.ch1_diagnostic.run --dataset WN11 --anonymise

# analyse — no training
python -m chapters.ch1_diagnostic.analyse --dataset WN11 --smi
```

Each run takes ~33 min and produces `checkpoints/ch1-WN11-lora/` and `checkpoints/ch1-WN11-anon-lora/`.

### Running two jobs on two T4s

```python
def pair(a, b):
    pa = subprocess.Popen(f"CUDA_VISIBLE_DEVICES=0 {a}", shell=True)
    pb = subprocess.Popen(f"CUDA_VISIBLE_DEVICES=1 {b}", shell=True)
    return pa.wait(), pb.wait()
```

Two **independent** jobs, one per card — **never** DataParallel. The smoke test hard-fails if it sees 2 visible GPUs, because HF Trainer would wrap the model in DataParallel, autocast wouldn't reach the replicas, and fp32 adapters would meet fp16 base weights.

### The four parsers — the heart of the chapter

The model generates **once**. The same outputs are then read **four different ways**:

| Parser | What it does | What it measures |
|---|---|---|
| **strict** | Exact expected string only | Format compliance |
| **lenient** | Substring match — deliberately reproduces KG-LLM's `find("no")` bug | What the original paper would have scored |
| **logit** | Ignores the text; compares P("Yes") vs P("No") directly | **What the model believed**, regardless of phrasing |
| **constrained** | Forces a choice between the two valid answers | Belief again, via a second route |

The `lenient` parser reproduces the bug **on purpose**, as an object of study — `find("no")` fires inside "k**no**w" and "can**no**t". Quantifying that is a contribution.

### The decomposition — the actual result

```
format_cost        = lenient − strict        recoverable formatting errors
format_ceiling     = logit   − lenient       knew it, wouldn't say it
memorisation       = logit   − logit(anon)   how much was just entity-name recall
residual_knowledge = logit(anon) − 0.5       real knowledge, above chance
```

A large `format_ceiling` is the headline: the model knew, and the scoring hid it.

### SMI — the second, independent instrument

`--smi` computes **sliced mutual information** between hidden states and labels, before and after tuning. FLAME reports that fine-tuning "primarily aligns representations rather than injecting knowledge". **If SMI barely moves while the logit gap is large, two instruments agree from opposite directions.** One instrument is an argument; two is evidence.

### Early signal from the actual run

| Run | train_loss | eval_loss |
|---|---|---|
| WN11 (real names) | 0.045 | 0.027 |
| WN11-anon | 0.117 | 0.104 |

The anonymised model fit the *same task* ~4× less well. Consistent with some of the plain model's performance coming from pretrained knowledge of real names. Suggestive only — the real result is the accuracy decomposition, not train loss.

---

## 10. Chapter 2 — is low rank the bottleneck?

**The decisive design choice:** sweep the number of entities |E| by **subsampling inside YAGO3-10** (10k → 25k → 50k → 123,182), *not* by switching datasets. FB15k-237, WN18RR and YAGO3-10 differ in relation count (237/11/37), density and label quality — comparing across them would confound |E| with everything else.

```bash
python -m chapters.ch2_adaptation.run --sweep     # print the plan, run nothing
```

**The ~22-run grid:**

| Arm | Runs | Purpose |
|---|---|---|
| LoRA × 4 |E| points | 4 | the sweep |
| MoRA × 4 |E| points | 4 | the sweep (**Session B**) |
| BOFT @ largest |E| | 1 | preservation hypothesis |
| **frozen probe** @ largest |E| | 1 | **the control — no training at all** |
| data axis (3k & 50k triples at |E| extremes) | 8 | does the data need scale with |E|? |
| 3 seeds, LoRA vs MoRA @ largest |E| | 4 | is any margin bigger than noise? |

```bash
python -m chapters.ch2_adaptation.run --dataset YAGO3-10 --triples 10000 --peft lora  --entities 123182
python -m chapters.ch2_adaptation.run --dataset YAGO3-10 --triples 10000 --peft boft  --entities 123182
python -m chapters.ch2_adaptation.run --dataset YAGO3-10 --triples 10000 --peft probe --entities 123182
python -m chapters.ch2_adaptation.analyse --forgetting
```

**Both outcomes are publishable:**

- MoRA's margin **grows** with |E| → low-rank bottleneck **confirmed**
- MoRA's margin **flat** → bottleneck **refuted** — and that's exactly what Chapter 1 predicts if tuning installs format. MoRA's own paper says it's "comparable on other tasks", so a null can't be read as failure.

**Why the frozen probe is non-negotiable:** without it, a flat MoRA result is ambiguous — you can't tell "MoRA didn't help" from "nothing helps". With it, you can.

**`--forgetting`** measures perplexity on held-out text before and after — did adapting to the KG damage general language ability? This is where BOFT's preservation claim gets tested.

**DPO (phase 2, winner only):**

```bash
python -m chapters.ch2_adaptation.run --peft dpo \
       --sft-adapter checkpoints/ch2-mora-E123182-T10000-s42 \
       --negatives type_consistent --entities 123182
```

---

## 11. Chapter 3 — how much context is worth paying for?

A ladder of five conditioning levels:

| Level | Adds |
|---|---|
| **L0** | nothing — entity description only = the KG-LLM baseline |
| **L1** | entity / relation descriptions |
| **L2** | semantic type |
| **L3** | label quality |
| **L4** | per-instance detail |

```bash
python -m chapters.ch3_conditioning.run --plan               # the 6-run plan
python -m chapters.ch3_conditioning.run --analyse            # routing + faithfulness, NO training
python -m chapters.ch3_conditioning.run --level L3 --train   # one training run
```

**6 runs total:** one adapter per level (L0–L4 = 5) plus an enrich-everything control. That's **content routing, not adapter routing** — one adapter per *level* rather than one per *bucket* (24+). A deliberate departure from the original spec, to fit the compute budget.

**Why this chapter cannot fail:** the **∅ (do-not-enrich) branch**. If conditioning hurts at every level, the router learns to always skip, and the finding is "the optimal policy is not to condition" — with evidence across four granularities. That echoes GS-KGC, which found neighbours *alone* scoring **below** no-context on 3/3 datasets.

> ⚠️ **Before the real Chapter 3 run:** relation descriptions are currently a template string derived from the relation label. Fine for a smoke test, but for real results they **must be LLM-generated** — otherwise L1 injects boilerplate and the first rung measures nothing.

> A past bug worth knowing: the level flags were switched on but **no content was ever supplied**, so all five levels emitted identical prompts and the ladder "showed" that conditioning doesn't help. `PromptConfig.__post_init__` now **raises** if a flag is on without content.

> **Pre-flight check built into `--analyse`:** the ∅ branch fires only for elements in the `rich` quality band. If fewer than 2% reach `rich`, the router can never skip, and `ch3/run.py` warns you — a 0% skip rate would then be a **data problem, not a result**.

---

## 12. Chapter 4 — knowing when to stay silent

**Zero training.** Pure inference over Chapter 2/3 checkpoints. The safest chapter; never cut it.

```bash
python -m chapters.ch4_measurement.run \
       --adapter checkpoints/ch2-lora-E123182-T10000-s42 \
       --dataset YAGO3-10 --limit 2000
```

**Pipeline:**

1. **One sampling loop, two uses** — the same 10 samples per prompt give both Hits@K *and* disagreement-based uncertainty. Sampling twice would be wasted GPU time
2. **Three confidence sources compared:** sequence log-prob · P(True) from logits · sampling disagreement
3. **Calibrate each** (temperature / isotonic) → ECE, Brier, reliability curves
4. **Abstention → risk–coverage curve** — if it only answers the 60% it's surest about, how accurate is it there?
5. **Hallucination, two types:** type 1 = OOV (invented an entity that doesn't exist); **type 2 = type violation** (a real entity of the wrong kind — rarely measured anywhere)
6. **Three computable traces** for the review app: routing reason, abstention reason + candidates, type-check reason

**Faithfulness — the part that matters most.** It's not enough for the model to give a reason; the reason has to be the *actual* cause. `src/routing/faithfulness.py` runs a **decision-flip test**: change the thing the model *said* drove its decision, and check the decision actually changes. If it doesn't, the explanation is decoration.

> A past bug worth knowing: correctness was once `gold.startswith(pred[:3])`, which scored **every empty prediction as correct** (`"".startswith("")` is `True`). A model answering nothing would have posted 100%. Now a `clean_verdict` with three ordered rules: refusal first ("I don't know" contains `n't`), negation before affirmation (the gold negative *"No, this is **not true**."* contains "true"), and word boundaries (so `know` ≠ `no`).

---

## 13. Frozen settings — and why

Everything lives in **one** `configs/base.yaml`. The PEFT method is a CLI flag precisely so every other constant is *guaranteed* identical across conditions.

| Setting | Value | Why |
|---|---|---|
| model | Qwen2.5-1.5B-Instruct | Small enough for a T4; FLAME validates the family |
| dtype | **fp16 + sdpa** | T4 has no real bf16; `eager` gives NaN |
| train_triples | 10,000 → ~20,000 instances | Compute budget |
| effective batch | **32** (micro 4 × accum 8) | **Not KG-LLM's 128** — at 20k instances that's only 156 steps/epoch, making warmup 32% of training |
| cutoff_len | 512, **dynamic** padding | KG-LLM used 50 with `max_length` and truncated |
| epochs / lr / warmup | 2 / 3e-4 / 100 | KG-LLM's values, warmup ≈8% of ~1250 steps |
| LoRA | r=8, α=16, dropout 0.05, `[q_proj, v_proj]` | KG-LLM's and MKGL's. **Fixed across all methods** |
| test_subset | 2,000, fixed | Identical items across conditions → **paired** tests valid |
| seed | 42 | `set_all_seeds()` seeds python/numpy/torch/HF |

> **On seeding:** `TrainingArguments(seed=)` alone seeds the Trainer but **not** numpy, `random`, or the sampling and negative-generation that happen *before* training. With 3 seeds planned, that silently broke significance testing. `load_config()` now calls `set_all_seeds()` in every runner.

> **On the test subset:** 2,000 fixed items, not WN11's full 21,088. That's what makes McNemar's paired test valid. **Not directly comparable to KG-LLM's full-test figures — say so in the paper.**

---

## 14. Full run order

```bash
# ---------- SESSION A (official peft) ----------
# 0  smoke test — ~5 min, before anything long
CUDA_VISIBLE_DEVICES=0 python -m scripts.smoke_test
# 0b BOFT kernel patch (notebook section 0b), then set butterfly factor accordingly

# 1  data — fast, no GPU
python -m scripts.fetch_data --datasets WN11 FB13
python -m src.data.build_instructions --dataset WN11 --n_triples 10000 --seed 42
python -m src.data.build_instructions --dataset WN11 --n_triples 10000 --seed 42 --anonymise

# 2  Chapter 1 — 2 training runs, rest inference
python -m chapters.ch1_diagnostic.run     --dataset WN11
python -m chapters.ch1_diagnostic.run     --dataset WN11 --anonymise
python -m chapters.ch1_diagnostic.analyse --dataset WN11 --smi

# 3  Chapter 2 — the arms official peft can run
python -m chapters.ch2_adaptation.run --sweep
python -m chapters.ch2_adaptation.run --peft boft  --entities 123182
python -m chapters.ch2_adaptation.run --peft probe --entities 123182
python -m chapters.ch2_adaptation.run --peft lora  --entities 123182 --seed 42   # ★ CONTROL
python -m chapters.ch2_adaptation.analyse --forgetting

# 4  Chapter 3 — routing free, 6 runs for the ladder
python -m chapters.ch3_conditioning.run --analyse
python -m chapters.ch3_conditioning.run --level L3 --train

# 5  Chapter 4 — ZERO training
python -m chapters.ch4_measurement.run --adapter checkpoints/ch2-lora-E123182-T10000-s42

# 6  package — /kaggle/working is NOT permanent
zip -qr /kaggle/working/results_sessionA.zip results/

# ---------- SESSION B (MoRA fork, fresh session) ----------
pip install git+https://github.com/kongds/MoRA.git#subdirectory=peft-mora
python -m chapters.ch2_adaptation.run --peft mora --entities 123182
python -m chapters.ch2_adaptation.run --peft lora --entities 123182 --seed 42   # ★ same CONTROL

# ---------- after BOTH sessions ----------
python -m scripts.verify_env_control
```

---

## 15. Traps, ranked by how much time they cost

| # | Trap | Symptom | Fix |
|---|---|---|---|
| 1 | **No GPU attached** | `Torch not compiled with CUDA enabled`, dozens of unrelated errors | Settings → Accelerator → GPU T4 ×2 |
| 2 | **Forgot to `git push`** | Kaggle runs old code; your fix "didn't work" | Push before every session |
| 3 | **fp16 + eager attention** | `train_loss=0.0`, `grad_norm=nan` — *looks finished* | `attn_implementation: sdpa` |
| 4 | **BOFT patch not re-run** | BOFT silently at factor 1 | Run notebook section 0b every session |
| 5 | **Two GPUs visible to one job** | `mat1 and mat2 must have the same dtype` | `CUDA_VISIBLE_DEVICES=0`; two independent jobs, never DataParallel |
| 6 | **MoRA in Session A** | `LoraConfig got unexpected 'use_mora'` | Expected — MoRA is Session B |
| 7 | **`pip install -U peft`** | Pulls a new transformers, breaks the cross-session comparison | Pin `transformers==4.57.6` in **both** sessions |
| 8 | **Results not downloaded** | Gone | `/kaggle/working` is not permanent. Zip and download every session |
| 9 | **12-hour cap** | Run killed | `save_steps: 250` means you can resume from a checkpoint |
| 10 | **Ch3 `rich` band ~0%** | Ladder measures nothing | ∅ branch never fired — data problem, not a result |

---

## 16. What's still open

| # | Item | Kind |
|---|---|---|
| 1 | **Run the grid** — nothing is blocked on code | compute |
| 2 | **PopBS** — the only unwritten spec item (~30 lines). Entity-degree based; EIR (relation-frequency) is done | code, optional |
| 3 | **Real LLM-generated relation descriptions** — ⚠️ **do before Chapter 3** or L1 measures nothing | content |
| 4 | Fill `paper/main.tex` §6 — tables already laid out | writing |
| 5 | Fill `TODO` author fields in `paper/refs.bib` | writing |

**Cut order if compute runs short:** L4 rung → 7B runs → BOFT (only if forgetting is dropped too).
**Never cut:** the |E| sweep · risk–coverage · DPO · the frozen probe control.

---

## 17. Where things live

```
configs/base.yaml        ← ONE frozen config; PEFT method via --peft
src/
  data/     loaders · sampling · negatives · prompts · build_instructions
  train/    sft (LoRA/MoRA/BOFT) · probe (the FLAME control) · dpo
  infer/    generate (one loop, two uses) · scoring (logit comparison)
  eval/     parse (4 parsers) · calibration · abstention · hallucination
            forgetting · smi · significance
  routing/  types · features · router · faithfulness
  utils/    config (seeding, run ids, result IO)
chapters/   ch1_diagnostic · ch2_adaptation · ch3_conditioning · ch4_measurement
scripts/    fetch_data · smoke_test · verify_env_control
notebooks/  kaggle_session_A_official · kaggle_session_B_mora · kaggle_runner
app/        streamlit review app — run LAST, locally, over downloaded results
results/    one JSON per run, each stamped with its peft environment
```

**Deliberately absent:** `src/eval/metrics.py`. Hits@K lives in `infer/generate.py`, accuracy in `eval/parse.py`. A third module would only add indirection.

---

## 18. If you read nothing else

1. **Smoke test first, every session.** It costs 5 minutes and saves 12 hours.
2. **Push to GitHub before every session.** Kaggle clones; it doesn't see your laptop.
3. **`[n/a]` is not `[FAIL]`.** MoRA failing in Session A is the design working.
4. **Run LoRA in both sessions.** One extra run is what makes the whole Chapter 2 comparison defensible.
5. **Download `results/` before the session ends.** `/kaggle/working` does not survive.
6. **A null result is a result here.** Every chapter is built so both answers are publishable — that's the point of the controls.
