"""
STAGE C / N11 -- Selective Enrichment Routing.

The question
-----------
    At what GRANULARITY does conditioning stop paying?

        L0  none          every element treated identically  (KG-LLM's baseline)
        L1  entity vs relation                               (ColKGC already does this)
        L2  semantic type                                    (Knit: 4 POS tags only)
        L3  label quality band                               (nobody)
        L4  instance                                         (nobody)

Saturation at L2 is a useful negative result. Monotone gains to L4 prove
per-instance conditioning matters. A FLAT curve is the specificity analogue of the
corpus's "more context is not better" thread. All three are publishable.

⚠️ DESIGN DECISION -- content routing, not adapter routing
-----------------------------------------------------------
Adapter routing (one adapter per bucket) would keep us strictly on the
parameterisation axis, but it splits 10k triples across k buckets -- each adapter
would see 2.5k triples at L3 -- and multiplies the run count by k.

We use CONTENT routing instead: the router decides WHAT ENRICHMENT each element
receives, and ONE adapter is trained on the routed data. Six runs, not twenty-four.

This does NOT weaken the answer to MKGL's cost finding ("2-hop prompt context =
350.4 GPU-h -> MRR .363 vs 33.1 GPU-h -> .415"). MKGL's objection is to ADDING
prompt content indiscriminately. A router whose ∅ branch REMOVES content where it
does not pay is the response to that finding, not an instance of it -- and
`RoutingPlan.token_savings()` quantifies exactly that.

Adapter routing remains available as a follow-up if content routing shows signal.

★ Every decision carries a REASON that names the features it used. That is what
makes faithfulness mechanically testable (see faithfulness.py) -- prose
explanations have no counterfactual.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from .features import ElementFeatures

Action = Literal[
    "none",                    # ∅ -- do not enrich. THE POINT.
    "description_only",
    "relation_description",
    "exclusion_list",
    "typed_neighbours",
    "full",
]

Level = Literal["L0", "L1", "L2", "L3", "L4"]

# Rough prompt-token cost of each action -- used for the savings report.
# Calibrate against your tokeniser once; the ordering is what matters.
ACTION_TOKEN_COST = {
    "none": 0, "description_only": 25, "relation_description": 20,
    "exclusion_list": 40, "typed_neighbours": 60, "full": 145,
}


@dataclass
class Decision:
    element_id: str
    action: Action
    level: Level
    reason: str                       # human-readable trace
    reason_features: list[str] = field(default_factory=list)   # ★ what faithfulness ablates
    token_cost: int = 0


@dataclass
class RoutingPlan:
    level: Level
    decisions: dict[str, Decision]

    def action_distribution(self) -> dict[str, float]:
        c = Counter(d.action for d in self.decisions.values())
        n = len(self.decisions) or 1
        return {k: v / n for k, v in c.items()}

    def skip_rate(self) -> float:
        """Fraction routed to ∅ -- how often the policy declines to enrich."""
        return sum(d.action == "none" for d in self.decisions.values()) / max(len(self.decisions), 1)

    def token_savings(self, baseline: Action = "full") -> dict:
        """
        ★ The answer to MKGL's cost objection, quantified.
        Compares this plan against enriching every element identically.
        """
        base = ACTION_TOKEN_COST[baseline] * len(self.decisions)
        actual = sum(d.token_cost for d in self.decisions.values())
        return {
            "baseline_tokens": base,
            "routed_tokens": actual,
            "tokens_saved": base - actual,
            "savings_rate": (base - actual) / base if base else 0.0,
            "skip_rate": self.skip_rate(),
        }

    def summary(self) -> dict:
        return {"level": self.level, "n_elements": len(self.decisions),
                "action_distribution": self.action_distribution(),
                **self.token_savings()}


# ------------------------------------------------------------------ the router
class Router:
    """
    Rule-based and deliberately transparent: every branch names the features it
    consulted, so `reason_features` is exact rather than post-hoc.

    Each rule cites the paper that measured it -- see REFERENCES.md.
    """

    def __init__(self, level: Level = "L3"):
        self.level = level

    def route(self, f: ElementFeatures) -> Decision:
        fn = {"L0": self._l0, "L1": self._l1, "L2": self._l2,
              "L3": self._l3, "L4": self._l4}[self.level]
        d = fn(f)
        d.token_cost = ACTION_TOKEN_COST[d.action]
        return d

    def route_all(self, feats: dict[str, ElementFeatures]) -> RoutingPlan:
        return RoutingPlan(self.level, {k: self.route(v) for k, v in feats.items()})

    # -------------------------------------------------------------- L0
    def _l0(self, f: ElementFeatures) -> Decision:
        return Decision(f.element_id, "full", "L0",
                        "no conditioning: every element enriched identically",
                        [])

    # -------------------------------------------------------------- L1
    def _l1(self, f: ElementFeatures) -> Decision:
        """ColKGC's granularity: entities and relations get different treatment."""
        if f.kind == "relation":
            return Decision(f.element_id, "relation_description", "L1",
                            "relation: descriptions are commonly missing, so generating "
                            "one adds information (ColKGC Table 3, RelSemEnh)",
                            ["kind"])
        return Decision(f.element_id, "description_only", "L1",
                        "entity: use the existing description; do not regenerate "
                        "(ColKGC: rewriting gives ~0 gain)",
                        ["kind"])

    # -------------------------------------------------------------- L2
    def _l2(self, f: ElementFeatures) -> Decision:
        if f.kind == "relation":
            return Decision(f.element_id, "relation_description", "L2",
                            "relation: description absent", ["kind"])
        if f.semantic_type == "OTHER":
            return Decision(f.element_id, "typed_neighbours", "L2",
                            "no recoverable semantic type: supply typed neighbours "
                            "as a structural substitute",
                            ["kind", "semantic_type"])
        return Decision(f.element_id, "description_only", "L2",
                        f"semantic type '{f.semantic_type}' is known: description suffices",
                        ["kind", "semantic_type"])

    # -------------------------------------------------------------- L3
    def _l3(self, f: ElementFeatures) -> Decision:
        """
        The rung nobody has: condition on LABEL QUALITY.
        Motivated by UKGEBN (opaque ids), GS-KGC (`stool_NN_2`), MKGL ("14 entities
        named 'call'") and ColKGC (descriptions that already exist).
        """
        if f.kind == "relation":
            return Decision(f.element_id, "relation_description", "L3",
                            "relation: description absent", ["kind"])

        if f.looks_opaque:
            return Decision(f.element_id, "typed_neighbours", "L3",
                            "label is an opaque identifier, so text enrichment cannot "
                            "help (UKGEBN); supply structure instead",
                            ["kind", "looks_opaque"])

        if f.ambiguity > 3:
            return Decision(f.element_id, "typed_neighbours", "L3",
                            f"label shared by {f.ambiguity} entities: needs structural "
                            f"disambiguation (MKGL)",
                            ["kind", "ambiguity"])

        if f.has_description and f.quality_band == "rich":
            # ★ THE ∅ BRANCH -- ColKGC measured that this class gains nothing
            return Decision(f.element_id, "none", "L3",
                            "informative description already present: enrichment adds "
                            "no new information (ColKGC: 0.333 -> 0.332)",
                            ["kind", "has_description", "quality_band"])

        return Decision(f.element_id, "description_only", "L3",
                        f"label quality '{f.quality_band}': description only",
                        ["kind", "quality_band"])

    # -------------------------------------------------------------- L4
    def _l4(self, f: ElementFeatures) -> Decision:
        """Per-instance: the full feature vector, including degree."""
        d = self._l3(f)
        if d.action == "none" and f.degree_percentile < 0.1:
            # rare entities carry the memorisation pressure (Analyzing Bias:
            # long-tail underrepresentation) -- enrich even if a description exists
            return Decision(f.element_id, "typed_neighbours", "L4",
                            f"description present but entity is long-tail "
                            f"(degree pct {f.degree_percentile:.2f}): structure still helps",
                            ["kind", "has_description", "quality_band", "degree_percentile"])
        if d.action == "description_only" and f.degree_percentile > 0.9 and f.has_description:
            return Decision(f.element_id, "none", "L4",
                            f"high-degree entity (pct {f.degree_percentile:.2f}) with a "
                            f"description: well covered by structure already",
                            ["kind", "has_description", "degree_percentile"])
        d.level = "L4"
        return d


def ladder(feats: dict[str, ElementFeatures],
           levels: tuple[Level, ...] = ("L0", "L1", "L2", "L3", "L4")) -> dict[str, RoutingPlan]:
    """Route the same elements at every granularity -- the Chapter 3 experiment."""
    out = {}
    for lv in levels:
        plan = Router(lv).route_all(feats)
        out[lv] = plan
        s = plan.summary()
        print(f"[router] {lv}: skip={s['skip_rate']:.1%}  "
              f"tokens {s['routed_tokens']:,} vs {s['baseline_tokens']:,} "
              f"({s['savings_rate']:+.1%})")
    return out
