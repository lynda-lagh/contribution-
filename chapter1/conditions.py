"""
CHAPTER 1 — the experimental grid, defined in ONE place.

Two orthogonal families:

  CONDITIONS (training-time)  A B C D E G   -- what the model was TRAINED on
  PROMPTS    (inference-time) P0…P4         -- what the model is ASKED at test

Keeping them in one file is deliberate: Chapter 1's whole argument is that a
single variable moves at a time, and that is only checkable if the grid is
declarative rather than scattered across scripts.

    python -m chapter1.conditions          # print the grid and exit
"""
from __future__ import annotations

from dataclasses import dataclass, field


# =============================================================================
#  TRAINING CONDITIONS
# =============================================================================
@dataclass(frozen=True)
class Condition:
    id: str
    anonymise: bool          # entity surface forms -> entity{i}
    types: bool              # append [Person] / [Location] style tags
    negatives: str           # "random" | "type_consistent"
    n_negatives: int         # per positive triple
    isolates: str            # the one thing this row changes
    reference: str
    shuffle: bool = False    # ★ keep real names, PERMUTE which entity holds which

    @property
    def name(self) -> str:
        return (f"{'anon' if self.anonymise else 'real'}"
                f"{'+types' if self.types else ''}"
                f"-{self.n_negatives}{'hard' if self.negatives != 'random' else 'rand'}")

    @property
    def n_instances(self) -> int:
        """⚠️ Report this. More negatives = more DATA, not just a harder task."""
        return 10_000 * (1 + self.n_negatives)


CONDITIONS: dict[str, Condition] = {
    "A": Condition(
        "A", anonymise=False, types=False, negatives="random", n_negatives=1,
        isolates="baseline — KG-LLM's exact recipe",
        reference="KG-LLM (ICASSP 2025): random.choice(ent_list), 1 negative per positive"),

    "B": Condition(
        "B", anonymise=True, types=False, negatives="random", n_negatives=1,
        isolates="ENTITY NAMES — everything else identical to A",
        reference="P12 / KG-CF: the only paper in 188 testing pretraining memorisation"),

    "C": Condition(
        "C", anonymise=True, types=True, negatives="random", n_negatives=1,
        isolates="TYPE INFORMATION, with names removed",
        reference="★ the core new question. CATS / Knit / RealKGC all ADD types "
                  "and never REMOVE names, so none can answer this"),

    "D": Condition(
        "D", anonymise=True, types=True, negatives="type_consistent", n_negatives=1,
        isolates="negative HARDNESS (count held at 1)",
        reference="P09 / EGIT: KGE near-miss negatives. P10 flagged for using "
                  "random corruptions that are trivially separable"),

    "E": Condition(
        "E", anonymise=True, types=True, negatives="type_consistent", n_negatives=6,
        isolates="negative COUNT (hardness held constant)",
        reference="RealKGC: exactly 6 negatives (3 head-corrupted + 3 tail-corrupted)"),

    "G": Condition(
        "G", anonymise=False, types=True, negatives="random", n_negatives=1,
        isolates="★★ do types help when names ARE available? The field's actual claim",
        reference="CATS +7.2% MRR · Knit 0.2240→0.2490 · RealKGC +3.83% — all with "
                  "real names, none isolating the type contribution"),

    # ★★ THE CONTROL THAT DEFENDS THE WHOLE CHAPTER
    "S": Condition(
        "S", anonymise=False, types=False, negatives="random", n_negatives=1,
        shuffle=True,
        isolates="★★ real names, PERMUTED — kills the objection that anonymisation "
                 "destroys all signal by construction",
        reference="[OURS]. The obvious attack on B is 'entity4471 is unreadable, so "
                  "of course it fails.' Here every name is real English and only the "
                  "name↔entity BINDING is destroyed. If S ≈ B, the binding was the "
                  "signal, and the objection is answered with a measurement."),
}

# F (12 hard) deliberately dropped: matches CATS, but CATS is 7B and inductive.
# ~5 GPU-h to answer "does 12 beat 6?", which this thesis does not ask.


# =============================================================================
#  INFERENCE-TIME PROMPTS
# =============================================================================
@dataclass(frozen=True)
class PromptVariant:
    id: str
    types: bool
    instruction: bool
    demonstrations: int
    asks: str
    valid_on: tuple[str, ...]        # which checkpoints this is interpretable on
    note: str = ""


PROMPTS: dict[str, PromptVariant] = {
    "P0": PromptVariant(
        "P0", types=False, instruction=False, demonstrations=0,
        asks="baseline — the format the model was trained on",
        valid_on=("untuned", "tuned")),

    "P1": PromptVariant(
        "P1", types=True, instruction=False, demonstrations=0,
        asks="does STATING the type help?",
        valid_on=("untuned", "tuned"),
        note="Knit's POS tags, generalised to semantic types"),

    "P2": PromptVariant(
        "P2", types=False, instruction=True, demonstrations=0,
        asks="does INSTRUCTING the model to check compatibility help?",
        valid_on=("untuned",),
        note="⚠️ INERT ON THE TUNED MODEL. Loss is masked to the response, and the "
             "response is the fixed string 'Yes, this is true.' Step-by-step "
             "instruction-following was tuned OUT. Only the untuned arm can react."),

    "P3": PromptVariant(
        "P3", types=True, instruction=True, demonstrations=0,
        asks="types + instruction together",
        valid_on=("untuned",),
        note="⚠️ same limitation as P2"),

    "P4": PromptVariant(
        "P4", types=False, instruction=False, demonstrations=3,
        asks="★ does DEMONSTRATING the relation's type pattern beat stating it?",
        valid_on=("untuned", "tuned"),
        note="RealKGC's actual mechanism: show other triples sharing r_q so the "
             "model can compare head/tail types against real instances"),
}

STRUCTURAL_INSTRUCTION = (
    "Before answering, consider whether the two entities are compatible with "
    "the relation.")


# =============================================================================
#  THE INTERPRETATION RULES — written down BEFORE any result exists
# =============================================================================
CEILING_NOTE = """
Any 'recovery' is bounded by the memorisation gap measured in A vs B.
On WN11 that is 0.9315 - 0.5385 = 0.3930 (39.3 points).
Report '% of the gap recovered', not raw accuracy:
    recovered = (acc_condition - acc_B) / (acc_A - acc_B)
"""

# ★★ THE TYPE-TAG LEAK — found, measured, and FIXED. 15 Aug 2026.
#     `python -m chapter1.check_type_leak --dataset YAGO3-10`
#
# THE PROBLEM. An induced type is an entity's DOMINANT relation position, so a
# positive's tail is more likely to carry the tag `{r}::tail` than a corrupted
# tail is. With RANDOM test negatives the corrupted tail's tag is essentially a
# random type, and the one-line rule "tail tag == {r}::tail -> Yes" scored:
#
#     test negatives = random            62.4%   pos 52.5% / neg 27.7%   sep 24.8 pts
#
# 12.4 points of free accuracy, before any learning. That would have made the
# pre-registered rule "C >> B => the model can use types" fire on an artefact.
#
# THE FIX. Regenerate the YAGO3-10 test negatives type-consistently, so a
# corrupted tail is drawn from the SAME relation's observed range and its tag is
# distributed like the gold tail's:
#
#     scripts/make_test_negatives.py --strategy type_consistent --regenerate
#     test negatives = type_consistent   51.3%   pos 52.5% / neg 49.9%   sep  2.6 pts
#
# ★ ALSO VERIFIED HERE: C/D/E (anonymised) and G (real names) return byte-identical
#   leak figures, which empirically confirms the design claim in build_types() —
#   induced types are invariant under anonymisation.
#
# ⚠️ THE COST, and it must be reported. Type-consistent corruptions are plausible
#   facts, so more of them collide with the graph: 690 candidates were rejected
#   for already existing in train u valid u test. The ones that did NOT collide
#   are likelier to be true-but-unrecorded, so the closed-world exposure of this
#   test set is higher than the random-negative version's. Numbers from the two
#   versions are NOT comparable — say which one produced each result.
#
# ⚠️ Per-relation residue remains above chance: wasBornIn 60.5 · isLocatedIn 59.7
#   · graduatedFrom 59.5. Report per-relation accuracy, not only the aggregate.
TYPE_TAG_FLOOR = {
    # measured on the TYPE-CONSISTENT test set. Re-run check_type_leak after ANY
    # change to test.tsv — this number is only valid for the set it was measured on.
    "YAGO3-10": 0.513,
    "WN11": None,           # not yet measured; falls back to 0.5
}


def floor_for(cond_id: str, dataset: str) -> float:
    """Chance level for a condition: 0.5, or the tag-only floor if it shows types."""
    if not CONDITIONS[cond_id].types:
        return 0.5
    return TYPE_TAG_FLOOR.get(dataset) or 0.5


INTERPRETATION = {
    # --- typed conditions: compare against the FLOOR, never against B ---------
    "C ~= floor": "★ types cannot substitute for names — C only reproduces the "
                  "tag-only rule, so LoRA at 1.5B installs no type rule",
    "C >> floor": "the model CAN use types beyond the trivial rule, and simply "
                  "does not when names are easier",
    "C < floor":  "⚠️ tuning made the model WORSE than a one-line heuristic on its "
                  "own prompt — check for degeneracy before interpreting",
    "D > C":      "it is negative HARDNESS that forces rule use, not the type text",
    "E > D":      "negative COUNT matters — ⚠️ confounded with 3.5x more instances",
    "G ~= A + (floor - 0.5)":
                  "★ types add nothing beyond the tag artefact when names are "
                  "available — a direct challenge to CATS / Knit / RealKGC, whose "
                  "gains may be token volume plus this same artefact",
    "G > A + (floor - 0.5)":
                  "types add real signal even with names; the anonymised ladder "
                  "then says what kind",
    # --- untagged conditions: chance is 0.5 ----------------------------------
    "B ~= 0.5":   "anonymisation removes essentially all usable signal",
    "S ~= B":     "★★ the name↔entity BINDING was the signal",
    "S ~= A":     "⚠️ the model never used names; investigate before publishing",
}

ONLY_INCREASES = """
⚠️ For the PROMPT variants on the TUNED model: the model saw only P0 during
training, so a DROP under P1-P4 may be distribution shift rather than inability.
Declare in advance: only INCREASES are interpretable. The untuned arm has no such
problem -- it never trained on any format.
"""


def grid() -> list[tuple[str, str]]:
    """Every (condition, prompt) pair that is worth running."""
    out = []
    for c in CONDITIONS:
        out.append((c, "P0"))                     # matched: trained format
    for p in PROMPTS:                             # prompt sweep on A and B only
        for c in ("A", "B"):
            if (c, p) not in out:
                out.append((c, p))
    return out


if __name__ == "__main__":
    print("=" * 78)
    print("CHAPTER 1 GRID")
    print("=" * 78)
    print(f"\n{'id':3s} {'names':6s} {'types':6s} {'negatives':16s} {'instances':>10s}  isolates")
    print("-" * 78)
    for c in CONDITIONS.values():
        print(f"{c.id:3s} {'anon' if c.anonymise else 'real':6s} "
              f"{'yes' if c.types else 'no':6s} "
              f"{f'{c.n_negatives}x {c.negatives}':16s} {c.n_instances:10,d}  {c.isolates}")
    print("\n" + "-" * 78)
    for c in CONDITIONS.values():
        print(f"  {c.id}: {c.reference}")

    print("\n" + "=" * 78)
    print("PROMPT VARIANTS (inference only, no retraining)")
    print("=" * 78)
    for p in PROMPTS.values():
        print(f"\n{p.id}  types={p.types} instruction={p.instruction} demos={p.demonstrations}")
        print(f"    asks:  {p.asks}")
        print(f"    valid: {', '.join(p.valid_on)}")
        if p.note:
            print(f"    {p.note}")

    print(CEILING_NOTE)
    print(ONLY_INCREASES)
