"""
CHAPTER 1 CENTREPIECE -- four ways to read the SAME model output.

The decomposition
-----------------
    strict                     -> what the field reports
    lenient  - strict          -> cost of formatting alone
    logit    - lenient         -> the FORMAT CEILING: what the model knew but would not say
    logit(anonymised)          -> how much of that "knowledge" is memorisation

Why this matters
----------------
KG-LLM reports untuned LLaMA at 21.1 (WN11) / 9.1 (FB13) on a BINARY task where
chance is 50. A score five times below chance cannot be a knowledge failure.

And KG-LLM already knew parsing was a problem: eval_WN11_ft.py uses a STRICTER rule
for tuned models and a LENIENT one for untuned models. But the lenient rule contains
a bug (see LenientParser) that hands the untuned model free credit on the negative
class -- and it STILL scored below chance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["yes", "no", "unparseable"]


@dataclass
class ParseResult:
    verdict: Verdict
    parser: str
    raw: str


# ---------------------------------------------------------------- parser 1
class StrictParser:
    """
    KG-LLM's rule for TUNED models (eval_WN11_ft.py):

        if res.find("Yes") != -1 and label == "1":  correct
        elif res.find("No") != -1 and label == "-1": correct

    Note it is an if/elif: "Yes" is tested first, so a response containing both
    is read as positive.
    """
    name = "strict"

    def __call__(self, text: str) -> ParseResult:
        if "Yes" in text:
            return ParseResult("yes", self.name, text)
        if "No" in text:
            return ParseResult("no", self.name, text)
        return ParseResult("unparseable", self.name, text)


# ---------------------------------------------------------------- parser 2
class LenientParser:
    """
    KG-LLM's rule for UNTUNED models -- reproduced faithfully, bug included.

        if (res.find("Yes") or res.find(" yes")) and label == "1":  correct
        elif (res.find("No") or res.find("not") or res.find("n't")
              or res.find("no")) and label == "-1":                 correct

    THE BUG: `res.find("no")` is substring matching. It fires on
        k[no]w, [no]w, a[no]ther, can[no]t, [no]table, [no]rth, an[no]unced
    i.e. on most English sentences. Because of the if/elif, every negative-label
    example whose response contains "no" anywhere is scored CORRECT -- inflating
    the untuned model's negative-class accuracy.

    We reproduce it exactly so the published number is recoverable, and we report
    `buggy_no_hits` to quantify how much free credit it grants.
    """
    name = "lenient"

    # The four substrings KG-LLM tests, in its order.
    _MARKERS = ("No", "not", "n't", "no")
    # Same markers, but only as standalone words -- what the rule MEANT to match.
    _WORD = {m: re.compile(rf"(?<![A-Za-z]){re.escape(m)}(?![A-Za-z])") for m in _MARKERS}

    def __call__(self, text: str) -> ParseResult:
        if "Yes" in text or " yes" in text:
            return ParseResult("yes", self.name, text)
        if any(m in text for m in self._MARKERS):
            return ParseResult("no", self.name, text)
        return ParseResult("unparseable", self.name, text)

    def buggy_no_hit(self, text: str) -> bool:
        """
        True when the negation branch fired ONLY on substrings buried inside larger
        words -- i.e. the model never actually said a negation.

        Examples that trigger it:
            "I know it well."        -> "no" inside k[no]w
            "I cannot verify."       -> "not" inside can[not]     <- a REFUSAL, not a "No"
            "Another example."       -> "no" inside a[no]ther
            "It was announced."      -> "no" inside an[no]unced

        The third case is the important one: KG-LLM's parser converts refusals into
        negative PREDICTIONS. That is the same failure mode KG-LLM's own Table VI
        shows for GPT-4 ("I cannot verify specific personal information ..." scored
        wrong on a True-labelled triple), and it is why abstention has never been
        measurable in this literature.
        """
        if "Yes" in text or " yes" in text:
            return False                      # the yes-branch won; no negation involved
        fired = [m for m in self._MARKERS if m in text]
        if not fired:
            return False
        # buggy iff NO marker appears as a standalone word
        return not any(self._WORD[m].search(text) for m in fired)


# ------------------------------------------------- refusal / abstention detector
# Phrases that signal the model DECLINED rather than answered "No".
# KG-LLM's parser converts every one of these into a negative PREDICTION.
_REFUSAL_PATTERNS = [
    r"\bi don'?t know\b", r"\bi do not know\b",
    r"\bcannot (verify|confirm|determine|answer|provide)\b",
    r"\bcan'?t (verify|confirm|determine|answer|tell)\b",
    r"\bnot (enough|sufficient) information\b",
    r"\bno information\b", r"\bunable to (verify|determine|confirm|answer)\b",
    r"\bas an ai\b", r"\bi'?m sorry\b", r"\bi am sorry\b",
    r"\bnot sure\b", r"\bunclear\b", r"\bi couldn'?t find\b",
]
_REFUSAL = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(text: str) -> bool:
    """
    The model declined to answer.

    This is the behaviour KG-LLM's Table VI shows for GPT-4 ("I cannot verify
    specific personal information about individuals who are not public figures")
    on a TRUE-labelled triple -- scored WRONG because the response contains "not".
    Knit's Figure 1 goes further and lists "I don't know." as a HALLUCINATION.

    Two independent papers treat a correct refusal as a failure. Counting refusals
    separately is what makes abstention measurable at all (Chapter 4).
    """
    return bool(_REFUSAL.search(text))


def categorise(text: str, lenient: "LenientParser") -> str:
    """
    Why did this response get the verdict it got?

        answered_yes / answered_no   the model actually stated an answer
        refusal_scored_as_no         ★ abstention converted into a negative prediction
        spurious_substring           parser artefact ("no" inside "know")
        unparseable                  nothing matched
    """
    if "Yes" in text or " yes" in text:
        return "answered_yes"
    if is_refusal(text):
        return "refusal_scored_as_no"
    if lenient.buggy_no_hit(text):
        return "spurious_substring"
    if any(m in text for m in LenientParser._MARKERS):
        return "answered_no"
    return "unparseable"


def response_breakdown(texts: list[str]) -> dict:
    """
    Distribution over the categories above -- one of Chapter 1's headline tables.

    'refusal_scored_as_no' + 'spurious_substring' is the share of the published
    accuracy that has nothing to do with the model's beliefs about the triple.
    """
    lp = LenientParser()
    cats = [categorise(t, lp) for t in texts]
    n = len(texts) or 1
    counts = {c: cats.count(c) for c in
              ("answered_yes", "answered_no", "refusal_scored_as_no",
               "spurious_substring", "unparseable")}
    return {
        "counts": counts,
        "rates": {k: v / n for k, v in counts.items()},
        "n": n,
        "not_a_real_answer_rate":
            (counts["refusal_scored_as_no"] + counts["spurious_substring"]
             + counts["unparseable"]) / n,
    }


# ---------------------------------------------------------------- parser 3
class LogitParser:
    """
    *** THE CONTRIBUTION ***

    No generation at all. Compare P("Yes") against P("No") at the first
    response-token position. Format is removed entirely from the measurement:
    the model cannot be penalised for refusing, hedging, rambling, or echoing
    pretraining text.

    Implemented in src/infer/scoring.py -- this class only interprets the scores.
    """
    name = "logit"

    def __call__(self, p_yes: float, p_no: float) -> ParseResult:
        v: Verdict = "yes" if p_yes >= p_no else "no"
        return ParseResult(v, self.name, f"P(Yes)={p_yes:.4f} P(No)={p_no:.4f}")

    @staticmethod
    def confidence(p_yes: float, p_no: float) -> float:
        """Normalised margin in [0,1] -- feeds calibration and abstention (Ch4)."""
        tot = p_yes + p_no
        return abs(p_yes - p_no) / tot if tot > 0 else 0.0


# ---------------------------------------------------------------- parser 4
class ConstrainedParser:
    """
    Belief under a FORCED format: decoding restricted to {"Yes", "No"}.
    Sits between logit (no format) and strict (full format), isolating the cost
    of producing a well-formed sequence rather than a single token.

    See src/infer/scoring.py::constrained_choice.
    """
    name = "constrained"

    def __call__(self, choice: str) -> ParseResult:
        v: Verdict = "yes" if choice.strip().lower().startswith("y") else "no"
        return ParseResult(v, self.name, choice)


# ---------------------------------------------------------------- scoring
def score(verdicts: list[Verdict], labels: list[int],
          unparseable_as: Verdict | None = None) -> dict:
    """
    labels: +1 / -1  (KG-LLM's test.tsv convention)

    unparseable_as:
        None  -> unparseable counts as WRONG   (honest)
        "no"  -> unparseable counts as negative (what substring matching effectively does)
    """
    assert len(verdicts) == len(labels)
    correct = unparseable = 0
    for v, y in zip(verdicts, labels):
        if v == "unparseable":
            unparseable += 1
            if unparseable_as is None:
                continue
            v = unparseable_as
        if (v == "yes" and y == 1) or (v == "no" and y == -1):
            correct += 1
    n = len(labels)
    return {
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "n": n,
        "unparseable": unparseable,
        "unparseable_rate": unparseable / n if n else 0.0,
    }


def decompose(strict: dict, lenient: dict, logit: dict,
              logit_anon: dict | None = None) -> dict:
    """
    The Chapter 1 result.

        format_cost      = lenient - strict     (recoverable formatting errors)
        format_ceiling   = logit   - lenient    (knew it, would not say it)
        memorisation     = logit   - logit_anon (entity-name recall)
        residual_knowledge = logit_anon - 0.5   (above chance without entity names)
    """
    out = {
        "strict_acc": strict["accuracy"],
        "lenient_acc": lenient["accuracy"],
        "logit_acc": logit["accuracy"],
        "format_cost": lenient["accuracy"] - strict["accuracy"],
        "format_ceiling": logit["accuracy"] - lenient["accuracy"],
        "total_format_effect": logit["accuracy"] - strict["accuracy"],
        "chance": 0.5,
        "logit_above_chance": logit["accuracy"] - 0.5,
    }
    if logit_anon is not None:
        out["logit_anon_acc"] = logit_anon["accuracy"]
        out["memorisation"] = logit["accuracy"] - logit_anon["accuracy"]
        out["residual_knowledge"] = logit_anon["accuracy"] - 0.5
    return out


PARSERS = {
    "strict": StrictParser(),
    "lenient": LenientParser(),
    "logit": LogitParser(),
    "constrained": ConstrainedParser(),
}
