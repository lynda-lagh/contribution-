# Where else this paper can go

Checked 15 Aug 2026. Only venues whose deadline is still open are listed.
Closed already: **AAAI-27** (28 Jul), **KDD 2027** (19 Jul), **ICKG 2026** (19 Jun),
**ISWC 2026** (7 May).

---

## The shortlist

| Venue | Deadline | Length | Blind | Format |
|---|---|---|---|---|
| **IEEE BigData 2026** | 21 Aug 2026 | 10 pp **incl. refs** | single | IEEE |
| **ICASSP 2027** | 16 Sep 2026 | see note | — | IEEE |
| **ICLR 2027** | 24 Sep (abs 19 Sep) | ~9 pp | double | ICLR |
| **ECIR 2027 — full** | 5 Oct (abs 21 Sep) | — | double | Springer LNCS |
| **★ ECIR 2027 — Reproducibility** | **2 Nov** (abs 12 Oct) | **12 pp + unlimited refs** | double | Springer LNCS |
| **WWW 2027** | 11 Oct 2026 | ~9 pp | double | ACM |

---

## ★ Recommendation: ECIR 2027 Reproducibility track

**Southampton, 22–24 March 2027. Abstract 12 Oct 2026, paper 2 Nov 2026.**
Springer LNCS proceedings. Chairs: Johanne Trippas (RMIT), Timo Breuer (TH Köln).

This track was written for the paper you have. Not approximately — the review
criteria map onto it line by line.

### Their criteria against your paper

| Their criterion (verbatim) | Your paper |
|---|---|
| *"Are there new insights or perspectives that emerged from reproduction which were not reported in the original paper?"* | The memorisation decomposition. This is the whole contribution, and it is scored as **Novelty** here |
| *"Is the reproduced paper proposing new evaluation criteria (new measures, statistical tests, etc.)?"* | The gap and the memorisation share, plus McNemar with correction |
| *"A successful reproduction of the work is not a requirement"* | ★ Your reproduction **partly failed** — untuned 67.75 against their reported 21.1. Here that is a finding, not an embarrassment |
| *"the potential difficulties encountered"* | fp16+eager NaN, DataParallel breaking autocast, the `res.find("no")` evaluator defect. Currently buried in a README; here it is **scored content** |
| *"How important is the reproduction... Is the original paper central or marginal to the community?"* | KG-LLM founded the subfield and its code is public. Maximally central |
| *"Are the code and datasets available to reviewers at the time of review?"* | Yes, and your validator and randomised test suite go beyond what most submissions offer |
| *"Submissions from the same authors... will not be accepted"* | You are a different team from Yao et al. ✅ |

### Why this solves your two blocking problems

**"You beat nobody."** The reproducibility track does not ask you to. That
objection disappears entirely rather than being argued around.

**Results.** The deadline is **11 weeks away**, not six days. Time to run the full
seven-condition grid, three seeds, both datasets, and the qualitative table.

**And 12 pages plus unlimited references** — against BigData's 10 including
references. Roughly 3 extra pages of usable space.

### What it costs

| Change | Effort |
|---|---|
| IEEE → **Springer LNCS** template | half a day; content unaffected |
| Single-blind → **double-blind** | remove names, anonymise the repo (anonymous.4open.science), avoid self-identifying phrasing |
| Reframe as a reproduction study | §IV and §V already do this; mostly a retitle and a restructure of §I |
| ⚠️ **Topical fit** | ECIR is Information Retrieval. *"Lack of topical fit"* is a listed desk-reject reason |

**On topical fit — the real risk, and how to manage it.** Lead with the ranking
half of the paper, not the classification half. §V-F already converts the
classifier into a ranker and reports MRR and Hits@K under the 50-way filtered
protocol; ECIR reviewers read that as retrieval. Frame the question as *"does a
neural ranker rank by relevance or by memorised identifiers?"* — which is a
recognisable IR question with its own history. Do not lead with triple
classification.

⚠️ Also note: **do not put this on arXiv** before or during ECIR review. Their
policy discourages it explicitly because it breaks double-blindness.

---

## ICASSP 2027 — the home venue

**Deadline 16 Sep 2026. Toronto, 16–21 May 2027.**

**KG-LLM was published at ICASSP 2025. FLAME at ICASSP 2026.** Your object of
study and your closest representation-level rival are both here. The audience
already knows the work you are auditing, which is worth a great deal.

⚠️ **The page limit is the problem.** The CFP page does not state it; ICASSP's
standard is **4 pages plus a 5th page for references only**. Your paper is 8–9.
That is not a trim, it is a rewrite: one instrument, one table, one claim.

★ **But there is a second option on the same timeline.** The CFP announces an
**IEEE Open Journal of Signal Processing (OJSP) track at 8+1 pages, same review
timeline as ICASSP**. Accepted papers are published in OJSP — *a journal* — and
presented at the conference, though not in the conference proceedings. **8+1
pages is almost exactly your current length**, and it partly answers the
"journal rather than conference" ambition. Verify the page limit and the
separate submission form in the author kit before committing.

---

## The others, briefly

**WWW 2027** — 11 Oct 2026, Dublin. CCF A, prestigious, and there is a genuine
knowledge-graph community. But it is a general web-research venue and a
measurement paper with no system win is a hard sell against method papers.

**ICLR 2027** — 24 Sep 2026 (abstract 19 Sep). Open review, high visibility, and
ICLR does reward analysis papers. Also extremely competitive, and your single
model at a single scale is exactly what ICLR reviewers attack. Only worth it with
the scale sweep done.

**ECIR 2027 full-paper track** — 5 Oct 2026. Same venue, same topical-fit
question, but you would be judged as a method paper, which is the framing that
suits you least. If you are going to ECIR, go to the reproducibility track.

---

## Suggested plan

1. **IEEE BigData, 21 Aug** — submit the narrow version (A / B / S plus the free
   instruments) if it is ready. Low cost, and rejection loses nothing.
2. **ECIR Reproducibility, 2 Nov** — the real target. Run the full grid in the
   eleven weeks between, convert to LNCS, anonymise, reframe around ranking.
3. **Knowledge-Based Systems** (Q1, where RealKGC is published) — the journal
   extension afterwards, with the scale sweep and a third dataset.

Steps 1 and 2 do not conflict: BigData notification is 24 Oct, before the ECIR
paper deadline of 2 Nov. If BigData accepts, ECIR needs to be substantially
different — the full grid plus the reproduction framing clears that bar. If
BigData rejects, you have the reviews in hand before submitting to ECIR.

⚠️ Check the dual-submission rules yourself before relying on that sequencing.
