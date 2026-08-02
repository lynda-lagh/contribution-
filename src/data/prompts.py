"""
Prompt construction.

Serialization is a CONTROLLED VARIABLE (R30): 16 distinct input representations
exist across the KGC literature and not one paper compares two of them. We fix
KG-LLM's QA format + Alpaca wrapper and never change it, so that any difference
between conditions is attributable to the variable under test (P05's protocol).

Optional components are OFF for Chapter 1 and switched on in Chapter 3. Each
inclusion/exclusion is backed by a measured ablation:

  entity description (existing)   ON   Knit: 0.2240 -> 0.2490 on WN18RR. Free.
  entity description (regenerated) OFF ColKGC: ~0 gain; FB15k-237 DROPS 0.333->0.332
  relation description             Ch3 ColKGC: 0.665 -> 0.670; RelSemEnh's whole premise
  exclusion list                   Ch3 GS-KGC: the component that actually works
  type / POS tag                   Ch3 Knit: "NN denotes nouns, VB denotes verbs..."
  neighbours                       OFF GS-KGC: neighbours ALONE score BELOW no-context
  path filtering                   NEVER MKGL: 10.6x GPU-hours for lower accuracy
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .loaders import KG, Triple

# KG-LLM's Alpaca wrapper, reproduced verbatim (~20 tokens of overhead).
ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n"
    "### Instruction:\n{instruction}\n### Response:\n"
)

YES = "Yes, this is true."
NO = "No, this is not true."


@dataclass
class PromptConfig:
    template: str = "alpaca"
    include_entity_description: bool = True
    regenerate_entity_description: bool = False   # ColKGC: rewriting existing ones ~= 0 gain
    include_relation_description: bool = False
    include_exclusion_list: bool = False
    include_type_tag: bool = False
    include_neighbours: bool = False
    n_neighbours: int = 0
    typed_neighbours: bool = True      # if ever enabled: KG-LLM/APE both verbalise
                                       # neighbours UNTYPED -- a defect we do not repeat
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Fail loudly when a flag is on but the content it needs is absent.

        Silent no-ops here would be the worst possible bug: Chapter 3 would report
        that conditioning "does not help" when in fact no conditioning content was
        ever injected.
        """
        need = {
            "include_relation_description": "relation_descriptions",
            "include_type_tag": "entity_types",
            "include_exclusion_list": "exclusions",
            "include_neighbours": "neighbours",
        }
        missing = [key for flag, key in need.items()
                   if getattr(self, flag) and key not in self.extras]
        if missing:
            raise ValueError(
                f"PromptConfig: flags enabled but extras missing {missing}. "
                f"Populate extras with those keys (see build_enrichment_extras) or "
                f"turn the flags off -- otherwise the prompt silently omits them and "
                f"the experiment measures nothing."
            )


def triple_classification_instruction(t: Triple, kg: KG, cfg: PromptConfig) -> str:
    """KG-LLM verbatim: 'Is this true: {h} {r} {t}?'"""
    h = kg.ent2txt.get(t.head, t.head)
    r = kg.rel2txt.get(t.relation, t.relation)
    tl = kg.ent2txt.get(t.tail, t.tail)

    instr = f"Is this true: {h} {r} {tl}?"

    if cfg.include_relation_description:
        d = cfg.extras.get("relation_descriptions", {}).get(t.relation)
        if d:
            instr += f" ({r} means: {d})"

    if cfg.include_type_tag:
        types = cfg.extras.get("entity_types", {})
        th, tt = types.get(t.head), types.get(t.tail)
        if th or tt:
            instr += f" [types: {th or '?'} -> {tt or '?'}]"

    if cfg.include_exclusion_list:
        excl = cfg.extras.get("exclusions", {}).get((t.head, t.relation), [])
        if excl:
            names = ", ".join(kg.ent2txt.get(e, e) for e in excl[:10])
            instr += f" Known answers to exclude: [{names}]."

    if cfg.include_neighbours and cfg.n_neighbours > 0:
        nbrs = cfg.extras.get("neighbours", {}).get(t.head, [])[: cfg.n_neighbours]
        if nbrs:
            if cfg.typed_neighbours:
                parts = [f"{kg.rel2txt.get(r_, r_)} {kg.ent2txt.get(e_, e_)}"
                         for r_, e_ in nbrs]
            else:
                parts = [kg.ent2txt.get(e_, e_) for _, e_ in nbrs]
            instr += f" Neighbours of {h}: " + " | ".join(parts) + "."

    return instr


def wrap(instruction: str) -> str:
    """Prompt WITHOUT the answer -- what the model sees at inference."""
    return ALPACA_NO_INPUT.format(instruction=instruction)


def to_alpaca_record(instruction: str, output: str) -> dict:
    """KG-LLM's LLaMA/LoRA training format."""
    return {"instruction": instruction, "input": "", "output": output}


def build_enrichment_extras(kg: KG, *, relation_descriptions: bool = False,
                            entity_types: bool = False, exclusions: bool = False,
                            neighbours: bool = False, n_neighbours: int = 5) -> dict:
    """
    Populate the CONTENT the optional prompt blocks need.

    Without this, enabling a flag produces a prompt identical to not enabling it --
    which would make Chapter 3 measure nothing while appearing to run correctly.

    relation_descriptions : generated once per relation (ColKGC/RelSemEnh: relation
                            descriptions are "commonly missing", which is why
                            GENERATING them pays while rewriting entity ones does not)
    entity_types          : from routing/types.py  (Knit's POS tags generalised)
    exclusions            : other known answers for (head, relation)
                            -- GS-KGC's "give an answer outside the list"
    neighbours            : TYPED (relation, entity) pairs. KG-LLM and APE both
                            verbalise these UNTYPED; we do not repeat that.
    """
    from collections import defaultdict

    extras: dict = {}

    if relation_descriptions:
        # placeholder text derived from the relation label; replace with LLM-generated
        # descriptions when running the full Chapter 3 pipeline
        extras["relation_descriptions"] = {
            r: f"the relation '{txt}' links a subject to an object"
            for r, txt in kg.rel2txt.items()
        }

    if entity_types:
        from ..routing.types import entity_types as _types
        extras["entity_types"] = _types(kg)

    if exclusions or neighbours:
        excl: dict[tuple[str, str], list[str]] = defaultdict(list)
        nbrs: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for t in kg.train:
            excl[(t.head, t.relation)].append(t.tail)
            nbrs[t.head].append((t.relation, t.tail))
            nbrs[t.tail].append((t.relation, t.head))
        if exclusions:
            extras["exclusions"] = dict(excl)
        if neighbours:
            extras["neighbours"] = {k: v[:n_neighbours] for k, v in nbrs.items()}

    return extras


def prompt_length_report(prompts: list[str], tokenizer) -> dict:
    """
    Sanity check, not a tuning step: with dynamic padding a generous cutoff is
    nearly free, but we still report the distribution because KG-LLM used
    cutoff_len=50 with padding='max_length' and silently truncated longer prompts.
    """
    lens = sorted(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    n = len(lens)
    q = lambda p: lens[min(n - 1, int(n * p))]
    return {"n": n, "min": lens[0], "median": q(0.5), "p95": q(0.95),
            "p99": q(0.99), "max": lens[-1]}
