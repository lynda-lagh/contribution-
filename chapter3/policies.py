"""
★★ THE ALLOCATION POLICIES — the whole grid, declaratively.

THE QUESTION
------------
Prior work asks *how much* context helps and finds that beyond a point more does
not (P02, P08, P11, P12, P19, P20, P21). Every one of those systems gives every
element the same treatment.

    We fix the budget and vary only WHERE it goes.

Each policy scores candidate blocks; `budget.allocate` fills the budget greedily
by that score. Same budget for all of them, so any MRR difference is
specificity.

THE LADDER
----------
    S0  uniform       no element-level information used at all
    R   random        same action mix, decisions shuffled     <- THE CONTROL
    S1  entity property   does this element have a description?
    S2  semantic type     what type does the relation imply?
    S3  label quality     how informative is the existing label?
    S4  instance          decide per (head, relation) query
    ORACLE            allocate using the gold answer -- infeasible, bounds the rest

★ R is the control that makes every other row interpretable. If S4 ≈ R, the
  DECISIONS add nothing and only the budget matters -- a clean negative result.
  Without R, "S4 ≈ S0" is ambiguous between *specificity does not pay* and
  *our policy is bad*. This is Chapter 1's condition S, transplanted.

★ ORACLE is the ceiling. Run it FIRST: if ORACLE ≈ S0, no policy can win at this
  budget and you have saved yourself the twelve runs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

BUDGETS = (0, 30, 60, 120, 240)     # 0 is the floor: (h, r, ?) and nothing else


@dataclass
class Policy:
    id: str
    name: str
    isolates: str                    # what this policy varies, in one phrase
    level: str | None = None         # the specificity level it instantiates
    seed: int = 42
    _rng: random.Random = field(default=None, repr=False)

    def priority(self, block, ctx) -> float | None:
        raise NotImplementedError

    def reason(self, block, ctx) -> str:
        return f"{self.id}: kept {block.kind} for {block.target}"


# --------------------------------------------------------------- S0 uniform
class Uniform(Policy):
    """
    Every element treated identically. Blocks enter in a fixed order, cheapest
    first, so the budget spreads evenly rather than being eaten by one big list.

    ⚠️ This is the baseline the whole chapter is measured against. It must be a
       FAIR baseline: same blocks available, same budget, only the ordering is
       uninformed. A deliberately bad uniform baseline would make any policy
       look good.
    """
    def priority(self, block, ctx):
        from .budget import KINDS
        return -KINDS.index(block.kind)          # cheap kinds first, no element info

    def reason(self, block, ctx):
        return "uniform: no element-level information used"


# ---------------------------------------------------------------- R random
class RandomPolicy(Policy):
    """
    ★ THE CONTROL. Same blocks, same budget, priorities shuffled.

    Deterministic per (seed, block) so a rerun reproduces the allocation exactly
    -- a random control that changes between runs is not a control.
    """
    def priority(self, block, ctx):
        h = hash((self.seed, block.kind, block.target)) & 0xFFFFFFFF
        return h / 0xFFFFFFFF

    def reason(self, block, ctx):
        return "random: allocated without using any element feature"


# ------------------------------------------------- S1 entity property
class EntityProperty(Policy):
    """
    Route on the one feature with real spread on these graphs.

    YAGO3-10 feature report: has_description_rate = 7.9%. So 92% of entities
    have nothing but a name -- those need neighbours; the 8% that have a
    description may not. That is a genuine 8/92 split.

    ⚠️ Contrast the ORIGINAL router, which keyed on quality bands where
       moderate = 95.7%. One bucket covering 96% of the graph gives the policy
       nothing to decide, and the ladder came out flat BY CONSTRUCTION.
    """
    def priority(self, block, ctx):
        has_desc = block.meta.get("has_description", False)
        if block.kind == "neighbours":
            return 3.0 if not has_desc else 0.5     # no description -> needs them
        if block.kind == "entity_description":
            return 2.5 if has_desc else float("-inf")   # nothing to include
        return 1.0

    def reason(self, block, ctx):
        has = block.meta.get("has_description", False)
        if block.kind == "neighbours":
            return ("entity has no description, so neighbours are its only signal"
                    if not has else
                    "entity already has a description; neighbours are secondary")
        return f"entity property: has_description={has}"


# ---------------------------------------------------- S2 semantic type
class SemanticType(Policy):
    """
    Allocate by the type the query relation implies.

    CATS: "relations within KGs impose latent type constraints to head and tail
    entities being connected... the relation `works in` typically connects a
    person (head) and a location (tail)."

    Where the type is unambiguous the type tag is cheap and sufficient; where the
    relation's range is large and heterogeneous, a tag says little and neighbours
    are worth more.
    """
    def priority(self, block, ctx):
        ent = block.meta.get("type_entropy")          # bits, over the relation's range
        if block.kind == "type_tag":
            return 3.0 if (ent is None or ent < 2.0) else 1.0
        if block.kind == "neighbours":
            return 2.5 if (ent is not None and ent >= 2.0) else 0.5
        return 1.0

    def reason(self, block, ctx):
        ent = block.meta.get("type_entropy")
        if block.kind == "type_tag":
            return (f"relation's type is near-deterministic (H={ent:.2f} bits): "
                    f"a tag suffices" if ent is not None and ent < 2.0
                    else f"relation's type is diffuse (H={ent}): tag is weak evidence")
        return f"semantic type: relation type entropy {ent}"


# ---------------------------------------------------- S3 label quality
class LabelQuality(Policy):
    """
    Allocate by how informative the existing label already is.

    ⚠️ On YAGO3-10 the labels are bare names: median 2 words, 100% bare labels,
       7.9% with any description. If this policy collapses onto S1, SAY SO --
       "label quality and label presence are the same feature on this graph" is a
       finding about the graph, not a failure of the policy.
    """
    def priority(self, block, ctx):
        q = block.meta.get("label_words", 0)
        if block.kind == "entity_description":
            return 3.0 if q >= 3 else (1.0 if q > 0 else float("-inf"))
        if block.kind == "neighbours":
            return 3.0 if q < 3 else 0.5
        return 1.0

    def reason(self, block, ctx):
        q = block.meta.get("label_words", 0)
        return (f"label is {q} words: uninformative, spend on neighbours"
                if q < 3 else f"label is {q} words: already informative")


# -------------------------------------------------------- S4 instance
class Instance(Policy):
    """
    Decide per (head, relation) query, combining the element features.

    The finest specificity in the ladder. If S4 does not beat S2, the honest
    conclusion is that per-instance tailoring costs more than it returns -- the
    SPECIFICITY analogue of "more context is not better", and a contribution to
    that line rather than a failure.
    """
    def priority(self, block, ctx):
        has = block.meta.get("has_description", False)
        words = block.meta.get("label_words", 0)
        ent = block.meta.get("type_entropy")
        need = (0.0 if has else 1.5) + (0.0 if words >= 3 else 1.0) \
               + (1.0 if (ent is not None and ent >= 2.0) else 0.0)
        if block.kind == "neighbours":
            return 1.0 + need
        if block.kind == "entity_description":
            return 3.0 if has else float("-inf")
        if block.kind == "type_tag":
            return 2.0 if (ent is not None and ent < 2.0) else 0.8
        return 1.0

    def reason(self, block, ctx):
        has = block.meta.get("has_description", False)
        words = block.meta.get("label_words", 0)
        ent = block.meta.get("type_entropy")
        return (f"instance: has_description={has}, label_words={words}, "
                f"type_entropy={ent} -> {block.kind} prioritised")


# ------------------------------------------------- S5 semantic specificity
class SemanticSpecificity(Policy):
    """
    ★ Allocate by how SPECIFIC the label's meaning is, not by what type it is.

    THE DISTINCTION, AND WHY IT IS NOT S2
    -------------------------------------
        semantic TYPE          what kind        Person, Location   <- S2, CATS, Knit, P29
        semantic SPECIFICITY   how specific     animal -> terrier -> Yorkshire terrier

    Two entities can share a type and sit at completely different levels of
    generality. Only the first is ever used in the corpus.

    Three cases our other features cannot distinguish, which need DIFFERENT
    KINDS of context, not merely different amounts:

        'male'              general · unambiguous · degree 61,044  -> needs almost nothing
        'Aïmen Demai'       specific · unknown                     -> needs a description
        'washington county' specific · AMBIGUOUS (x25)             -> needs DISAMBIGUATION

    FEATURES (meta keys, filled by sources.py)
        depth       hypernym depth. ★ EXACT on WN18RR: entities ARE WordNet synsets
        n_senses    polysemy -> ambiguity
        idf         corpus rarity, the transferable proxy when depth is unavailable

    ⚠️⚠️ GATED. Do NOT report this policy before `profile_specificity.py` passes.
         Three cheap checks can each kill it:
           1. depth has no variance      -> the 95.7%-quality-band mistake again
           2. depth ~ log(degree)        -> it is long-tail routing in disguise,
                                            which P30/KICGPTv2 already owns
           3. depth ~ cluster depth      -> P31/GS-Quant already derives this by
                                            clustering embeddings, for free

    ★ POSITIONING (P31, GS-Quant): that paper builds a coarse-to-fine hierarchy by
      AGGLOMERATIVE CLUSTERING and uses it to structure discrete representation
      CODES. We take specificity from an EXTERNAL LEXICAL RESOURCE and use it to
      allocate a TOKEN BUDGET. Shared intuition; different source, object and cost
      model. State this or a reviewer states it for you.
    """
    def priority(self, block, ctx):
        d = block.meta.get("depth")             # None when unavailable
        senses = block.meta.get("n_senses", 1)
        idf = block.meta.get("idf")
        general = (d is not None and d <= 4) or (idf is not None and idf < 2.0)
        ambiguous = senses is not None and senses >= 3

        if block.kind == "entity_description":
            # a general word is already understood; a specific unknown one is not
            return float("-inf") if not block.meta.get("has_description") else (
                1.0 if general else 3.0)
        if block.kind == "neighbours":
            # ★ ambiguity is what neighbours resolve: they pin down WHICH referent
            return 3.5 if ambiguous else (0.5 if general else 2.5)
        if block.kind == "demonstrations":
            # expensive; worth it only for specific entities with nothing else
            return 2.0 if (not general and not block.meta.get("has_description")) else 0.3
        if block.kind == "type_tag":
            return 2.5 if general else 1.0      # a general label IS close to a type
        return 1.0

    def reason(self, block, ctx):
        d = block.meta.get("depth")
        senses = block.meta.get("n_senses", 1)
        if block.kind == "neighbours" and senses and senses >= 3:
            return (f"label is ambiguous ({senses} senses): neighbours disambiguate "
                    f"which referent is meant")
        if d is not None and d <= 4:
            return f"label is semantically general (depth {d}): little context needed"
        return (f"label is specific (depth {d}, {senses} sense(s)): "
                f"{block.kind} carries the meaning the name does not")


# ------------------------------------------------------------ ORACLE
class Oracle(Policy):
    """
    ★ THE CEILING, and it is not a method.

    Allocates using `meta['helps']`, computed against the gold answer. Infeasible
    in deployment; its only job was to bound what ANY policy could achieve.

    ⚠️⚠️ NOT USABLE AS SPECIFIED, AND DISABLED FOR THAT REASON.
       `meta['helps']` would have to say whether a block improves THIS query,
       which is only knowable by scoring the model with and without that block:
       2^|blocks| forward passes per query. `sources.candidate_blocks` therefore
       never sets the key, so this policy kept nothing and spent 0 tokens at
       every budget — silently identical to the B=0 floor, while being read as
       "no allocation can help here".

    ★ THE CEILING WE REPORT INSTEAD is computed after the fact in
      `report.py::policy_selection_oracle`: for each query, take the best rank
      achieved by ANY policy at that budget. It bounds what a perfect *router
      over our policies* could reach, costs no extra GPU, and is a tighter and
      more useful bound than "some unknown ideal allocation" would have been.
      It is a different quantity, and the paper must say which one it reports.
    """
    def priority(self, block, ctx):
        raise NotImplementedError(
            "ORACLE cannot be run as an allocation policy: 'helps' is not "
            "computable without 2^|blocks| forward passes per query. Use "
            "report.py::policy_selection_oracle, which bounds what a perfect "
            "router over the implemented policies could achieve.")

    def reason(self, block, ctx):
        return "oracle: uses gold; not a method"


POLICIES: dict[str, Policy] = {
    "S0_uniform":   Uniform("S0_uniform", "uniform", "nothing — the baseline"),
    "R_random":     RandomPolicy("R_random", "random", "★ the control: decisions vs budget"),
    "S1_property":  EntityProperty("S1_property", "entity property",
                                   "presence of a description", level="L1"),
    "S2_type":      SemanticType("S2_type", "semantic type",
                                 "type implied by the relation", level="L2"),
    "S3_quality":   LabelQuality("S3_quality", "label quality",
                                 "informativeness of the label", level="L3"),
    "S4_instance":  Instance("S4_instance", "instance",
                             "per (head, relation) query", level="L4"),
    "S5_semantic":  SemanticSpecificity("S5_semantic", "semantic specificity",
                                        "★ how specific the label's meaning is",
                                        level="L5"),
    # ⚠️ ORACLE is deliberately ABSENT — see the Oracle docstring. It kept no
    #    blocks and spent 0 tokens, so evaluating it burned GPU to re-measure
    #    the B=0 floor. The ceiling is computed post hoc by
    #    report.py::policy_selection_oracle at no additional cost.
}

# ⚠️ S5 is GATED. `profile_specificity.py` must pass before it is reported.
GATED = {"S5_semantic"}

# kept so that older result files and notebooks referring to ORACLE still parse
RETIRED = {"ORACLE": "replaced by report.policy_selection_oracle (post hoc)"}


# ---------------------------------------------------------------------------
#  INTERPRETATION — written down BEFORE any result exists
# ---------------------------------------------------------------------------
INTERPRETATION = {
    "ORACLE ~= S0":
        "★★ STOP. At this budget no allocation can help — uniform is already "
        "optimal. Report it and move the budget, do not run the ladder.",
    "S4 ~= R":
        "★ the DECISIONS add nothing; only the budget matters. A clean negative "
        "result, and the specificity analogue of 'more context is not better'.",
    "S4 > R and S4 > S0":
        "★★ specificity pays at fixed cost — the headline. Quantify as MRR gain "
        "at equal tokens.",
    "S4 > S0 but S4 ~= R":
        "⚠️ the gain comes from the action MIX, not from targeting. Report the "
        "mix as the finding and drop the specificity claim.",
    "S2 ~= S4":
        "specificity saturates at type level — per-instance tailoring costs more "
        "than it returns. Useful: it tells the next person where to stop.",
    "curves converge as budget grows":
        "specificity matters only under a tight budget — a precise, deployable "
        "claim about when to bother.",
    "S3 ~= S1":
        "on this graph label QUALITY and label PRESENCE are the same feature "
        "(YAGO3-10: 7.9% have any description). A finding about the graph.",
    # ── added after the Chapter-3 reading round (P28–P31) ───────────────────
    "any policy > uniform on MRR":
        "★★ P28 predicts this is POSSIBLE, not just permitted: unsupportive "
        "context is learned as evidence, so withholding it can RAISE accuracy "
        "rather than merely save tokens. Report the accuracy delta as a result, "
        "not as a constraint that was satisfied.",
    "S5 ~= S1":
        "semantic specificity reduces to description presence on this graph — "
        "report it, and check the profiler's depth/degree correlation before "
        "concluding anything stronger.",
    "S5 ~= S2":
        "specificity adds nothing over type. Given P29 already published the "
        "type rung, that is the honest place for the ladder to stop.",
    "demonstrations dominate the budget":
        "⚠️ expected at small B: one demonstration can consume the whole budget. "
        "Report tokens_by_kind — a policy that spends everything on one block is "
        "a finding about the cost profile, not a bug.",
}

SANITY = """
⚠️ Before believing any row:
   1. B=0 must be the worst. If it is not, the context is hurting and that is the
      finding — investigate before continuing.
   2. ORACLE must be the best. If a policy beats it, `helps` is mis-computed.
   3. Every policy must actually spend its budget: check `utilisation` ~ 1.0.
      A policy that cannot fill the budget is not being compared fairly.
   4. Prompts must DIFFER between policies at the same budget. If they are
      byte-identical the ladder measures nothing — the exact failure Chapter 1
      hit when condition C came out identical to B.
"""
