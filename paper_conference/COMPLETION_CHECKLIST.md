# Conference paper — what is done, what is waiting on the runs

**File:** `main.tex` (IEEEtran, `conference` option)
**Preview:** `PREVIEW_non-IEEE-layout.pdf` — built with a substitute class, see *Compilation* below.

Every hole in the paper prints in **red** as `[...]` because of this in the preamble:

```latex
\newif\ifdraft
\drafttrue          % <- set to \draftfalse to hide every TODO before submission
```

There are **91 `\TODO` markers**. Nothing can be submitted with a silent gap.

---

## Written and complete

| Section | Content |
|---|---|
| Abstract | 150 words, no symbols or math (the template forbids them there) |
| I Introduction | 4 paragraphs — hook / specific problem / gap / aims, per the supervisor's structure |
| II Terminology | KG · **enrichment ⊃ completion** · task variants · transductive vs inductive · closed vs open-**domain** · CWA vs OWA + the closed-world penalty |
| III Skeleton | KG-LLM instantiated across the 8 stages (Table III) |
| IV Limitation | the untested half of the claim · why the protocol can't see it · negative sampling · gap statement |
| V Method | the decomposition · the 7-condition grid · 4 instruments · prompt ablation · classifier→ranker |
| VI Setup | datasets · model · controls · metrics · reproducibility |
| IX Threats | 6 threats, each with the mitigation |

## Waiting on the runs

| Section | State | Fill from |
|---|---|---|
| **VII Results** | tables laid out with correct columns and captions, all cells `\TODO{n}` | `results/*.json` |
| **VIII Discussion** | written as **pre-registered branches** — "if C ≈ B then …, if C ≫ B then …". Keep the branch that happened, delete the rest | your own outcomes |
| **X Conclusion** | complete except two numbers | Table V |

Section VIII is deliberately written *before* the results. That is what makes it
pre-registration rather than storytelling, and it is the strongest defence
against a reviewer suggesting the interpretation was fitted to the numbers.

---

## Before submission — hard checklist

1. **Choose the title.** Deliberately left empty. Three candidates sit in a
   comment block directly above `\title{}` in `main.tex`, each 10–12 words with
   abbreviations spelled out and a note on what it costs you:
   *quality-oriented* (closest to the thesis title), *finding-first* (punchiest),
   *instrument-first* (most descriptive). Paste your pick into `\title{}`.
2. **Fill the author block.** Department, university, city, country. Currently
   `TODO`. Review is single-blind, so real names stay on the paper.
3. **Fill every `\TODO`**, then set `\draftfalse`.
4. ~~Verify the 9 bibliography entries marked `% VERIFY`.~~ **Done.** All 22
   entries now carry full author lists, venues and DOIs, read off page 1 of the
   source PDFs in your corpus folder. No `% VERIFY` markers remain. See
   *Bibliography — what was corrected* below.
5. **Finish the contamination check** (§IX, last item). The novelty claim for the
   anonymisation *instrument* rests on it. 24 papers still unchecked.
6. **Delete nothing from the IEEE template** — it is already removed. The red
   warning block, the "Ease of Use" section, the sample table and figure, and the
   guidance bibliography are all gone.
7. **Check the page limit.** Settled: 10 pages including references. See below.

---

## Target venue — IEEE BigData 2026

| | |
|---|---|
| Conference | 2026 IEEE International Conference on Big Data |
| Where / when | Phoenix, AZ, USA — 14–17 Dec 2026 |
| Full-paper submission | **21 Aug 2026** (AoE) |
| Notification | 24 Oct 2026 |
| Camera-ready | 14 Nov 2026 |
| **Page limit** | **10 pages, IEEE 2-column — references counted inside the 10** |
| Review policy | **single-blind** — keep your name and affiliation on the paper |
| Acceptance rate | 18% (2025), 18.4% (2024) |
| Submission system | wi-lab.com/cyberchair/2026/bigdata26 |
| Format | IEEE Computer Society Proceedings Manuscript Formatting Guidelines |

### The page limit is not a problem

The preview runs to 10 pages in a substitute layout; real IEEEtran is denser, so
expect roughly **8–9 pages including references**. That is **inside** the limit
with room to spare. **No cutting is required.** Earlier drafts of this checklist
assumed a 6-page cap — that was wrong, and the cut-list has been removed.

The spare page is better spent on Section VII than saved. If anything is short
after the results land, expand the results and the discussion rather than the
front matter.

### Which topic area to submit under

Two areas in the call fit directly, and the paper should be pitched at whichever
you choose:

- **Foundation Models for Big Data** → *"Big data management for fine-tuning"*.
  This is the closest literal match: the paper is about what fine-tuning on graph
  data actually installs.
- **Big Data Benchmarks** → *"Benchmarks and Evaluation Frameworks"*,
  *"Data-Centric AI Methods, Tools, and Systems"*. This is the stronger pitch,
  because the contribution is an evaluation instrument rather than a model.

A third, weaker fit is **Big Data Science and Foundations** →
*"Data and Information Quality for Big Data"*, which connects to the
*orienté qualité* half of the thesis title.

Because review is **single-blind**, nothing needs to be anonymised, and the
released code can be cited by URL in the paper.

---

## Bibliography — what was corrected

All nine unverified entries were resolved from the source PDFs already sitting in
your corpus folder, not from the web. Authors, venue, volume, article number and
DOI were read off page 1 of each paper.

| Key | Was | Now |
|---|---|---|
| `realkgc` | "Y. Zhang and Z. Li" | Y. Zhang, J. Hu, C. Xiong, Z. Qin, H. Chen, Z. Li — *Knowledge-Based Systems* **349**, art. 116415, 2026 |
| `kgcf` | "Z. Zheng et al., 2024", no venue | Z. Zheng, Y. Dong, S. Wang, H. Liu, Q. Wang, J. Li — **IEEE BigData 2024**, doi 10.1109/BigData62323.2024.10826107 |
| `llmsim` | "J. Na et al." — wrong initial | **D.** Na, N. Kertkeidkachorn, X. Liu, K. Shirai — *GenAIK Workshop* (ICCL), 2025, pp. 78–86 |
| `gskgc` | no authors, year 2024 | R. Yang, J. Zhu, J. Man, H. Liu, L. Fang, Y. Zhou — arXiv:2408.10819, **2025** (v2, Jan 2025) |
| `ape` | no authors | J. Zhu and P. De Meo — **IJCNN 2025**, doi 10.1109/IJCNN64981.2025.11228614 |
| `egit` | no authors, no venue | P. Zhang, X. Xu, J. Wu, X. Lu + 10 more — *Information* **17**(2), art. 207, 2026 |
| `flame` | no authors, no venue, no year | B. Xue, Y. Xu, B. Ma, Y. Song, J. Ding, L. Fu, X. Wang — **ICASSP 2026**, doi 10.1109/ICASSP55912.2026.11464010 |
| `sdohen` | no authors, no venue | T. Shang, S. Yang, T. Zhai + 9 more — *Innovation in Aging* **9**(S1), art. igaf102, 2025 |
| `relsem` | no authors, no venue | J. Si, X. Ouyang, X. Zhu, Y. Zhang — **IEEE DSC 2024**, doi 10.1109/DSC63484.2024.00035 |

### Three things worth knowing

**KG-CF was published at IEEE BigData 2024** — the venue you are submitting to.
That is useful: it establishes that anonymisation-style leakage work is in scope
for this conference, and it gives you a same-venue citation to lean on in the
introduction.

**FLAME is ICASSP 2026, and KG-LLM is ICASSP 2025.** Your two most important
comparison points come from the same conference series one year apart. Worth a
sentence in the related-work framing.

**`llmsim` had the wrong author initial.** It was recorded as "J. Na"; the paper
says **Dong Na** (JAIST). A wrong initial is the kind of error a reviewer who
knows the work will spot immediately.

### A correction to something I told you earlier

I flagged the skeleton's "6.78M edges" for P04 (`sdohen`) as probably wrong,
because the abstract says ~92k triplets. **I was wrong — the skeleton was
right.** Checking the paper's own dataset table settles it:

| Graph | Nodes | Edges |
|---|---:|---:|
| PrimeKG (pre-existing) | 129,356 | 5,847,652 |
| SDoHenPKG (fused) | 139,304 | **6,780,101** |
| of which LLM-generated | — | **≈ 92k** |

Both numbers are real and they measure different things. 6.78M is the fused
graph; 92k is the LLM's contribution to it.

**But the check surfaced something more useful than an error would have been.**
The LLM contributed about **1.4% of the final graph's edges** — it enriched a
5.85M-edge biomedical KG by roughly 1.6% — and the paper's link-prediction gain
comes from that 1.6%. That is a far more interesting result than a big number,
and it is the right comparison for any claim about enrichment being worth its
cost. It also sharpens the N9 gap: the paper never measures the precision of
those 92k generated triplets, so the 1.4% is of unknown quality.

`KGC_pipeline_skeleton.md` now records the full breakdown in a note under the
Stage-0.5 table, so the 6.78M figure can no longer be misread as the size of the
enrichment.

### One real caveat on this citation

`sdohen` is an *Original Research Article* inside a **journal supplement**
(*Innovation in Aging* 9(S1), the GSA meeting supplement), 12 pages. That is a
weaker venue than a regular journal issue. If a reviewer challenges it, the claim
it supports — that some enrichment adds nodes and edges while other enrichment
adds only text — is also carried by `relsem`, cited alongside it, so the argument
does not rest on this one reference.

---

## Compilation

`IEEEtran.cls` is not installed in this sandbox and could not be fetched, so the
preview PDF was produced by substituting a two-column `article` class with stub
definitions for `\IEEEauthorblockN`, `\IEEEauthorblockA` and `IEEEkeywords`.

**What that verification does prove:** the LaTeX is syntactically valid, all 22
`\bibitem` entries resolve, all 20 `\cite` keys are defined, no reference is
undefined, no bibitem is uncited, the TikZ figure compiles, and only one
negligible 3pt overfull box remains.

**What it does not prove:** exact page count and column breaks under IEEEtran.

Compile the real thing on Overleaf (`IEEEtran` is preinstalled) or locally with
`texlive-publishers`. The `algorithmic` package is kept in the preamble because
the IEEE template ships it, though nothing in the paper uses it — drop it if your
installation lacks it.

---

## Where the content came from

| Paper section | Source in your repo |
|---|---|
| Definitions | `thesis analysis/2_synthesis/GLOSSARY.md` |
| The 8 stages, KG-LLM row | `thesis analysis/2_synthesis/KGC_pipeline_skeleton.md`, §P17 + Path N |
| Grid, instruments, provenance | `kgc-adaptation-thesis/chapter1/README.md` §1–2 |
| RealKGC / CATS positioning | `1_corpus/notes/P26_*.md`, `P27_*.md` |
| Structure and rhetoric | the three supervisor PDFs in `chaymouma thesis/` |

The supervisor's guidance was applied literally where it fits a technical venue:
the first paragraph makes a **claim, not an observation**, carries a statistic,
and ends on a **However**; the gap paragraph uses the *"acknowledge existing
research and identify gaps"* pattern and avoids the banned phrase *"no previous
studies have investigated"*; and the gap is made **painful** rather than merely
stated — what is lost is named explicitly at the end of §I paragraph 3 and again
in §IV-D.
