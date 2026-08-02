# Adaptation for Knowledge Graph Completion

**Enrichissement explicable et orienté qualité des graphes de connaissances assisté par les LLMs**

> **Thesis claim:** Adaptation for KGC — **parameterisation** × **objective** × **conditioning** — plus the measurement stack needed to know what any of it bought.

Built on **KG-LLM** (Yao, Peng, Mao & Luo, *ICASSP 2025*) — [`yao8839836/kg-llm`](https://github.com/yao8839836/kg-llm).

---

## Chapters

| Ch. | Question | Runs | Status |
|---|---|---|---|
| **1** | What does instruction tuning install — output **format** or entity **knowledge**? | **2** | 🟡 in progress |
| **2** | Is **low-rank a bottleneck** for entity memorisation? | 24 | ⬜ |
| **3** | At what **granularity** does conditioning stop paying? | 6 | ⬜ |
| **4** | Can the model know **when to stay silent** — and can we trust the reason? | **0** (inference) | ⬜ |

**Total ≈ 32 training runs.** Chapter 4 costs no training — it is inference over checkpoints from 2 and 3.

---

## What we change relative to KG-LLM

| | KG-LLM | This work |
|---|---|---|
| Model | LLaMA-7B/13B, ChatGLM-6B | **Qwen2.5-1.5B-Instruct** (7B for confirmation) |
| PEFT | LoRA only | **LoRA · MoRA · BOFT · frozen probe** |
| Objective | supervised (SFT) | **SFT vs DPO** |
| Negatives | `random.choice(all_entities)` — uniformly random | **+ KGE near-miss, + type-consistent** |
| Padding | `padding="max_length"`, `cutoff_len=50` | **dynamic padding**, `cutoff_len=512` |
| Effective batch | 128 (tuned on 112k–316k triples) | **32** (correct for 10k triples) |
| Answer parsing | substring match — ⚠️ `find("no")` fires on "k**no**w" | **4 parsers incl. logit comparison** |
| Confidence | none | **calibrated + abstention + risk–coverage** |
| Explanation | none | **computable traces + faithfulness test** |

---

## Setup

### Local

```bash
git clone <this-repo>
cd kgc-adaptation-thesis
pip install -r requirements.txt
```

### Kaggle (T4 ×2)

Open `notebooks/kaggle_runner.ipynb`, or in a cell:

```python
!git clone https://github.com/<you>/kgc-adaptation-thesis.git
%cd kgc-adaptation-thesis
!pip install -q -r requirements.txt
!python -m chapters.ch1_diagnostic.run --config configs/base.yaml --dataset WN11
```

⚠️ **Enable Internet** in notebook settings, and use **Save & Run All (Commit)** for long jobs — interactive sessions die on disconnect.

### Data

Download the four KGs + KG-BERT descriptions from KG-LLM and place them as:

```
data/{WN11,FB13,WN18RR,YAGO3-10}/
    entity2text.txt     # entity_id \t surface text
    relation2text.txt   # relation_id \t surface text
    train.tsv           # head \t relation \t tail
    test.tsv            # head \t relation \t tail \t label      label ∈ {1,-1}
```

**Upload once as a Kaggle Dataset** — re-downloading each session wastes quota.

---

## Fixed configuration

**Everything in `configs/base.yaml` is frozen across all runs.** Only the variable under test changes.

```
model            Qwen2.5-1.5B-Instruct, fp16      (T4 has no bf16)
train_triples    10,000  → ~20,000 instances      (1 positive + 1 negative each)
sampling         stratified by relation, min 10/relation, seed 42
cutoff_len       512 with DYNAMIC padding         (short prompts cost little)
effective batch  32  (micro 4 × accum 8)          ⚠️ NOT KG-LLM's 128
epochs           2 · lr 3e-4 · warmup 100         (≈8% of 1,250 steps)
LoRA             r=8, α=16, dropout 0.05, [q_proj, v_proj]
checkpoints      adapter only (~20–100 MB)
```

---

## Run order

```bash
# 0. smoke test — 15 minutes, catches API breakage before you lose a session
python -m scripts.smoke_test

# 1. build instruction data
python -m src.data.build_instructions --dataset WN11 --n_triples 10000 --seed 42

# 2. Chapter 1 — 2 training runs, everything else inference
python -m chapters.ch1_diagnostic.run --dataset WN11
python -m chapters.ch1_diagnostic.run --dataset WN11 --anonymise
python -m chapters.ch1_diagnostic.analyse        # the 4-parser decomposition
```

---

## Layout

```
configs/     base.yaml          ← single frozen config; PEFT method via --peft
src/
  data/      loaders · sampling · negatives · prompts · build_instructions
  train/     sft (LoRA/MoRA/BOFT) · dpo · probe
  infer/     generate · scoring (logit comparison)
  eval/      parse · calibration · abstention · hallucination
             forgetting · smi · significance
  routing/   types · features · router · faithfulness
  utils/     config (seeding, run ids, result IO)
chapters/    ch1_diagnostic · ch2_adaptation · ch3_conditioning · ch4_measurement
app/         streamlit review app (main · graph_backend)
scripts/     smoke_test
notebooks/   kaggle_runner.ipynb
results/     one JSON per run
```

> There is **one** config file, not four. The PEFT method is a CLI flag
> (`--peft lora|mora|boft|probe|dpo`) precisely so that every other constant is
> guaranteed identical across conditions.
>
> There is no `eval/metrics.py`: Hits@K lives in `infer/generate.py` and accuracy
> in `eval/parse.py`. A third module would only add indirection.

---

## Key references

| Ref | Used for |
|---|---|
| **KG-LLM** (ICASSP 2025) | base pipeline, prompts, data format |
| **FLAME** | frozen-probe control; **SMI** instrument; *"fine-tuning primarily aligns representations rather than injecting knowledge"* |
| **MKGL** (NeurIPS 2024) | cost evidence: prompt context = 10.6× GPU-hours for lower accuracy |
| **MoRA** (arXiv 2405.12130) | *"low-rank updating may limit the ability of LLMs to learn and memorize new knowledge"* |
| **BOFT** (arXiv 2311.06243) | multiplicative orthogonal updates; knowledge preservation |
| **GS-KGC** | exclusion constraints; sampling for Hits@K; OOV rate 38.9–45.3% |
| **ColKGC** | ABSENT-vs-IMPROVABLE: rewriting existing descriptions ≈ 0 gain |
| **P08 / TraceVal** | abstention bucket; trace vs explanation |
| **P12 / KG-CF** | entity-anonymisation contamination test |
