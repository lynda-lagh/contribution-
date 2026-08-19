# Chapter 1, file by file — what each one is for

Every file in `chapter1/`, plus the four support modules it depends on. For each:
**what it does**, **the idea behind it**, and **the specific failure it exists to
prevent** — because almost every guard here was written after something went
wrong silently.

Ordered the way you actually run them.

---

# The idea in one paragraph

A language model fine-tuned on a knowledge graph scores 95 % and everyone calls
it knowledge. But *"the model learned a rule"* and *"the model recognised a
name"* produce the **same accuracy**. So we run the same pipeline twice — once
normally, once with every entity name replaced by a meaningless code — and
report the two numbers **as a pair**. The gap between them is how much of the
score was really just name recognition.

Everything in this folder exists to make that one comparison trustworthy.

---

# 1 · The design — what we are measuring

## `conditions.py` — the whole experiment, in one file

**What it does.** Defines the seven conditions (A, B, C, D, E, G, S), the seven
prompt variants (P0–P7), the measured type-tag floors, and — importantly — what
each possible outcome would *mean*, written down before any run.

**The idea.** The chapter's whole argument is *one thing changes at a time*.
That is only checkable if the grid is declarative rather than scattered across
scripts. Every condition is A with exactly one factor changed.

| | what changes |
|---|---|
| **A** | nothing — this is KG-LLM's recipe |
| **B** | names → `entity4471` |
| **S** | names kept, but shuffled onto the wrong entities |
| **C** | B + type tags |
| **D** | C + harder negatives |
| **E** | D + six negatives instead of one |
| **G** | real names + type tags |

**The failure it prevents.** `INTERPRETATION` fixes the reading of each outcome
*in advance*, so a result cannot be reinterpreted after the fact. The audit
caught us breaking our own rule here: the outcome that actually happened —
S landing *between* A and B — had no entry, so the three-way decomposition was
written after seeing the data. It now says so, in the file.

**Also here.** `TYPE_TAG_FLOOR` and `check_type_gate()` — see
`check_type_leak.py`.

---

## `context.py` — the extra information for P5, P6, P7

**What it does.** Builds the optional context block: a relation description, five
neighbouring facts, or a two-hop path.

**The idea.** P6 is **KG-LLM's own K=5 neighbours**, which we had never
reproduced. It is their single biggest gain (YAGO3-10 Hits@1 0.0949 → 0.1330),
and it has never been tested with the names removed.

**The failure it prevents.** Showing "facts near the entity" is the fastest way
to hand the model the answer. Four routes, all closed:

1. the query triple itself
2. its mirror — many graphs store `(a,r,b)` **and** `(b,r,a)`
3. the gold entity reached by **any other relation**
4. **any** edge on the query relation — which KG-LLM does *not* guard

`assert_no_leak()` proves it, and proves it isn't vacuous by forcing the leak on
an adversarial index: *"0 leaks in 300 queries; guard removed the gold in
150/150 forced cases."*

---

# 2 · Before you spend a GPU-hour

## `validate.py` — is this dataset usable at all?

**What it does.** File-level integrity: are all four files present, do ids
resolve, are there ±1 labels, is the test set balanced, do negatives collide with
training facts. **Exits non-zero**, so a notebook cell cannot sail past it.

**The failure it prevents.** YAGO3-10 ships **5,000 test triples with no
labels**. Without this check, every test instance is built as a negative, the
run completes, and the accuracy is meaningless.

---

## `profile_data.py` — what does this graph actually look like?

**What it does.** Describes the dataset before you design an experiment on it:
identifier format, description length, relation imbalance, hubs, type sources,
ambiguity, degree distribution, coverage by training size.

**The idea.** `validate` says *"is it broken?"*. This says *"what is it?"* —
which conditions are even viable here.

**What it found on YAGO3-10.**

- POS types return **100 % OTHER** — ids carry no type marker
- descriptions are bare labels, **median 2 words** — so "description enrichment" has nothing to work with
- **`male` appears in 61,044 triples (2.8 %)** — check it isn't carrying the score alone
- **3.93 %** of entities share a surface form; *"washington county"* appears **25 times**
- at 10k training triples, only **8 %** of test triples have both entities seen

Every one of those is a sentence the paper needs.

---

## `check_type_leak.py` — can a one-line rule beat the model?

**What it does.** Measures what a trivial heuristic scores using only the type
tag: *"if the tail's tag names the query relation, answer yes."*

**The idea.** If a rule with **no model at all** already scores 62 %, then a
model scoring 65 % has learned almost nothing — and you would never know.

**The failure it prevents.** This is real. On YAGO3-10's original negatives the
rule scored **62.4 %**, i.e. 12.4 points of free accuracy. Regenerating the
negatives type-consistently brought it to **51.3 %** against a 0.5 floor. On
WN11 it is **56.8 %**, which is why WN11's typed conditions are blocked.

**Run it again after any change to `test.tsv`.** `preflight` only *reads* the
stored constant; it cannot know the file changed underneath it.

---

## `preflight.py` — eight checks, no fallbacks

**What it does.** Refuses to start a run that cannot produce a valid number.
Dataset files, labels, type source, leak, permutation, arm distinctness, type
invariance, writable checkpoints.

**The idea.** Every check corresponds to something that **already failed
silently** in this project. There are deliberately no fallbacks: the whole point
is to convert a silent downgrade into a stop.

**Check 5 is unusual** — it reads `rank.py`'s own source to confirm it applies
`cond.shuffle`. That bug shipped once and produced a wrong headline number; this
makes it impossible for it to come back.

**A bug found in the checker itself.** `SystemExit` derives from
`BaseException`, not `Exception`, so `except Exception` never caught it. Every
failure killed the process on the *first* one, skipped the rest, and never
printed a summary — so a failing preflight looked like one that stopped early.

---

## `audit_data.py` — is the data telling the truth?

**What it does.** Seven stages on the **data** rather than the code: raw graph,
split, labels, built instances, condition pairs, types, candidate pool.

**The idea.** `preflight` asks *"can this run start?"*. This asks *"is the data
honest?"*. Different questions, and both have failed here.

**Why we trust it.** On WN11 it independently reproduced **every** defect the
paper's Threats section reports — 2,220 duplicate training triples, 54 duplicate
test triples, 193 self-loops, 7/10,542 negatives that are true in training — none
of which were hardcoded.

**Severity is proportional.** `FAIL` stops you. `WARN` is *a number you must
report, not a bug to hide.*

---

# 3 · Building the data

## `data.py` — one builder for the whole grid

**What it does.** Turns triples into training instances. Renders every prompt.
`render()` is the **only** place a prompt string is ever produced.

**The idea.** Serialisation is a **controlled variable**. Sixteen distinct input
formats exist across our corpus and no paper compares two of them — so anything
that isn't the variable under test is held byte-identical.

**Three failures it prevents.**

**Empty type tags.** `entity_types(method="auto")` reads a part-of-speech marker
out of the identifier. WN11's ids have none → 1 type, 100 % OTHER → every type
block renders nothing → **condition C came out byte-identical to condition B**.
Training succeeded, accuracy looked plausible, and the ladder "showed" that types
don't help when types were never there. `build_types()` refuses.

**Unlabelled test sets.** `YES if t.label == 1 else NO` is silent when the label
is `None` — every instance becomes a negative. Now refuses, and refuses *early*,
before the expensive work.

**Context that isn't there.** A P6 build whose neighbour block comes out empty
**is** a P0 build. `--min-context` refuses below 50 %. On WN11 it measured P6 at
**93.0 %** and P7 at **5.3 %** — P7 is refused by default, because *"paths don't
help"* and *"paths don't exist"* are not the same finding.

---

## `run.py` — the orchestrator

**What it does.** `--build`, `--train`, `--evaluate`, `--rank`, `--plan`.

**The idea.** Every model is evaluated on **both** test sets. The runner will not
produce a single accuracy alone, because a single accuracy cannot express the
claim.

**The failure it prevents.** `--prompt` used to be ignored when computing paths,
so `--train --prompt P6` did two destructive things silently: it read the **P0**
instances, and it **overwrote the P0 adapter** with the result. The prompt is now
part of a run's identity.

---

# 4 · Scoring

## `evaluate.py` — both test sets, always

**What it does.** Scores a checkpoint on the real set and the anonymised set, and
returns the **gap**.

**The idea.** `gap = acc_real − acc_anon` is the result. Scoring is
generation-free — P(Yes) vs P(No) at the answer token, no text produced, nothing
for a substring rule to get wrong.

**Three failures it prevents.**

- **S scored on the wrong set.** Condition S resolved its "real" side to `{ds}-A`, so an adapter trained on a deranged graph was scored on undamaged names — a train/test mismatch, not the permuted-name control.
- **Silent base-model fallback.** `if adapter and Path(adapter).exists()` returned the **untuned** model when the path was wrong, filed under a tuned condition. Now raises.
- **Wrong denominator.** The memorisation share divided by a hardcoded 0.5, but for typed conditions chance is the measured tag floor. C: 0.8376 → **0.9649**.

There is also a `try/except` around the post-hoc analysis — deliberately. A
`TypeError` in a calibration helper once discarded two complete 40-minute runs
after all the GPU work was done. The accuracies are the result; analysis is a
convenience.

---

## `rank.py` — turning the classifier into a ranker

**What it does.** Scores 50 candidates by P(Yes) and sorts. That gives Hits@K
**and MRR**, without retraining.

**The idea.** Triple classification completes nothing, and its chance level of
0.5 lets a model that collapses onto one answer score exactly at chance. Ranking
moves chance to **0.0900** — an order of magnitude lower — which is what makes
the decomposition readable. The precedent is KG-BERT's; the stated limitation
*"MRR is not computable in this paradigm"* was never structural.

**Three failures it prevents.**

- **`cond.shuffle` was never applied** — condition S was ranked on the **real** graph. `m(S) = 0.2974`, and therefore the whole 71.5 % binding term, came from a train/test mismatch. **Fixed; the S arm must be re-ranked.**
- **Filtering used `train` only** while the paper claimed train ∪ valid ∪ test.
- **Only 200 of 500** per-query ranks were saved, widening every bootstrap interval by ~1.6× for no saving worth having.

⚠️ **One thing still open.** Candidate pools come from one shared RNG, so they
depend on `--limit` and query order. Chapter 3 fixed this with per-query seeds;
Chapter 1 never got the back-port. **Run every arm at the same `--limit`.**

---

# 5 · Reading the result

## `analysis.py` — three instruments, one claim

**What it does.** The gap table, the seen/unseen split, and calibration.

**The idea.** Instruments 1 and 2 are **independent**. Anonymisation destroys
names; the familiarity split leaves them completely intact and asks whether
*having seen the entity* is doing the work. Two instruments agreeing from
opposite directions is a different class of claim from one instrument asserting.

⚠️ This module reports **raw** accuracy in `seen_unseen`. Use
`chapter1/seen_unseen.py` for the balanced version — see below.

---

## `seen_unseen.py` — the second instrument

**What it does.** Splits test triples by whether their entities appeared in the
training sample. **Names stay completely intact.**

**The idea.** The obvious attack on condition B is *"replacing names with
`entity4471` destroys all information, so of course it collapses."* This answers
from a completely different direction, using results you already have, with no
GPU.

**Why it matters more than it looks.** The balanced familiarity gap is
**+0.0036** — fine-tuning on a triple's entities improves accuracy by essentially
nothing, while anonymisation removes 0.3940 from the same model on the same test
set. Two orders of magnitude apart. So the association is **pretrained**, not
recalled from our training set. That is a sharper claim than "the model
memorises", and it rules out the most obvious alternative explanation.

**The trap it avoids.** Bucket positive rates are 0.607 / 0.500 / 0.407, so a
predictor that always answers "Yes" scores a **+0.20 raw gap** having learned
nothing. **Read the balanced column.**

---

## `report.py` — everything accuracy hides

**What it does.** Per-class precision/recall/F1, confusion matrix, degenerate
check, per-relation breakdown, risk–coverage curve, McNemar.

**The idea.** *"A model that always answers Yes scores 50 % on a balanced set and
looks like a coin; a model that memorises scores 93 % and looks like it
understands."* This produces the numbers that tell those apart.

**Per-relation matters here.** YAGO3-10's top relation is **34.6 %** of all
triples, and `male` alone is 2.8 %. A headline mean can be one relation solved.

---

## `rescore.py` — did condition E actually learn?

**What it does.** Recomputes AUC and best-threshold balanced accuracy from
probabilities already on disk. No GPU.

**The idea.** Condition E trains at 1 positive : 6 negatives — **14 % positive** —
and is tested at 50/50. It learned the training prior and answered "No" to
everything: accuracy 0.5010, positive rate 0.0010, every P(Yes) in [0.08, 0.37].

Reported as accuracy that is "learned nothing at chance". **That may be wrong.**
A model can rank positives above negatives *perfectly* while putting every
probability below 0.5 — accuracy at chance, AUC 1.0. Accuracy conflates
*discrimination* with *calibration*; AUC measures only the first.

**Status: never run.** The three readings are pre-registered in the file. This is
the cheapest open item in the chapter.

---

## `compare.py` — our row in KG-LLM's Table II

**What it does.** Puts our number beside the 21 published methods, and adds AUC,
macro-F1 and McNemar.

**The idea.** Our **WN11 92.65 vs their 95.5** is a legitimate comparison — same
dataset, same *shipped* labels, same metric, and the gap is explained by 1.5B vs
7B. Showing it is stronger than hiding it.

**What it refuses.** YAGO3-10 classification (our negatives are *generated*),
FB13 (never ran), and **any** link-prediction number — theirs ranks over 123,182
entities, ours over 50. Ask for an invalid cell and it prints your rows alone
with *"Do NOT paste them into their table."*

---

## `showcase.py` — what the model actually answered

**What it does.** Prints the same query answered by every arm, and with `--story`
writes the whole result as plain English.

**The idea.** Every number in this chapter is an aggregate over 500 queries. KG-LLM's
Table VI is qualitative and is one of the most-read parts of that paper; we had
nothing equivalent.

```
Prince Leopold of Bavaria — hasWonPrize — ?   gold: Grand Cross of the Iron Cross
  A  real names   -> Grand Cross of the Iron Cross     CORRECT
  S  permuted     -> Silver Medal of Military Valor    rank 40   ← still a MEDAL
  B  anonymised   -> Zulu Dawn                         rank 42   ← a film
```

The whole decomposition, on one query. Binding lost at S, readability lost at B.

`--story` translates the metrics too: **71 out of 100 on the first try** with
names, **5 out of 100** without, against **2 out of 100** for random guessing.
That last number is what makes the others mean something.

---

## `test_chapter1.py` — 26 tests, every one a worked example

**What it does.** Runs in ~2 seconds on a random toy graph. No GPU, no model, no
downloads.

**The idea.** The graph, the names and the triple under test are **randomised on
every run**, and the seed is printed. A fixed fixture only ever proves the code
works on that fixture; randomising turns this into light property-based testing.

**It catches the class of bug that produces plausible numbers rather than a
crash** — which is the only kind that matters here. Two of the tests exist
because a bug shipped: condition S had **no test at all**, which is exactly how
`rank.py` shipped without applying the permutation.

---

# The support modules

| file | what it gives |
|---|---|
| `src/data/loaders.py` | `load_kg`, `anonymise`, `shuffle_surface_forms` — the three graph views. Anonymise changes the *text*, never the *ids*, which is what makes types invariant. |
| `src/routing/types.py` | induced types — the (relation, position) an entity occupies most often. **Endogenous**: computed from the edges under test. |
| `src/routing/semantic_types.py` | **exogenous** types — WordNet supersenses, YAGO's WordNet classes, NELL's ontology prefix. Written by someone else before our graph existed. |
| `scripts/make_test_negatives.py` | generates ±1 labels where a benchmark ships none. **Report that they are ours, not the benchmark's.** |
| `scripts/fetch_yago_types.py` | builds `entity2type.txt` from YAGO's own dump, or Wikidata P31 as fallback. Hard-fails below a coverage threshold. |

---

# The honest summary

**What we take from KG-LLM:** the prompt format, the Alpaca wrapper, random 1:1
negatives, the entity/relation text files, the benchmarks, LoRA. Condition A
*is* their recipe.

**What we added:** condition B (anonymisation), condition S (permutation — not in
any of the 188 papers we surveyed), ranking instead of classification, the
decomposition, the familiarity split, the generation-free decision rule, and the
machinery in §2 that makes all of it checkable.

**What we got wrong and fixed:** induced types were **endogenous** — computed from
the edges under test, which is why a one-line rule scored 62.4 %. Conditions C
and G now use exogenous types, and `--require-semantic` refuses to fall back
silently.
