"""
Load KG-LLM's data format.

    data/{DATASET}/
        entity2text.txt     entity_id \t surface text
        relation2text.txt   relation_id \t surface text
        train.tsv           head \t relation \t tail
        test.tsv            head \t relation \t tail \t label     label in {"1","-1"}

Descriptions originate from KG-BERT; KG-LLM states it uses "the same entity and
relation text descriptions as in [KG-BERT]".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    label: int | None = None       # +1 / -1 for test files, None for train


@dataclass
class KG:
    name: str
    ent2txt: dict[str, str]
    rel2txt: dict[str, str]
    train: list[Triple]
    test: list[Triple]
    # ★ Optional. Present only when the dataset ships a valid.tsv (CATS splits do).
    #   The FILTERED ranking protocol must filter against train + valid + test:
    #   a triple that is true but happens to live in valid is not a negative, and
    #   scoring it as one silently penalises every model equally but wrongly.
    valid: list[Triple] = field(default_factory=list)

    @property
    def entities(self) -> list[str]:
        return list(self.ent2txt.keys())

    @property
    def relations(self) -> list[str]:
        return list(self.rel2txt.keys())

    def all_true(self) -> set[tuple[str, str, str]]:
        """★ Every triple known to be true, for the filtered protocol."""
        return {(t.head, t.relation, t.tail)
                for t in (*self.train, *self.valid, *self.test)}

    def describe(self) -> dict:
        return {
            "dataset": self.name,
            "n_entities": len(self.ent2txt),
            "n_relations": len(self.rel2txt),
            "n_train": len(self.train),
            "n_valid": len(self.valid),
            "n_test": len(self.test),
        }


def _read_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def _read_triples(path: Path, has_label: bool) -> list[Triple]:
    out: list[Triple] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 3:
                continue
            label = int(p[3]) if (has_label and len(p) > 3) else None
            out.append(Triple(p[0], p[1], p[2], label))
    return out


def load_kg(dataset: str, root: str | Path = "data") -> KG:
    d = Path(root) / dataset
    missing = [n for n in ("entity2text.txt", "relation2text.txt", "train.tsv", "test.tsv")
               if not (d / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"{d} is missing {missing}. Download from github.com/yao8839836/kg-llm/tree/main/data"
        )
    # ★ valid.tsv is optional: KG-LLM's WN11/YAGO3-10 layout has none, the CATS
    #   inductive splits do. Loaded when present so `all_true()` can filter
    #   against it; absent, the protocol degrades to train+test and says so.
    vp = d / "valid.tsv"
    valid = _read_triples(vp, has_label=True) if vp.exists() else []

    return KG(
        name=dataset,
        ent2txt=_read_map(d / "entity2text.txt"),
        rel2txt=_read_map(d / "relation2text.txt"),
        train=_read_triples(d / "train.tsv", has_label=False),
        test=_read_triples(d / "test.tsv", has_label=True),
        valid=valid,
    )


def anonymise(kg: KG) -> KG:
    """
    P12 (KG-CF) contamination control: replace every entity surface form with an
    opaque identifier while KEEPING relations. Any performance retained after this
    cannot come from pretraining memorisation of entity names.

        "This aims to detect data leakage, assessing whether internal knowledge in
         language models provides an unfair advantage."
    """
    anon = {e: f"entity{i}" for i, e in enumerate(sorted(kg.ent2txt))}
    return KG(
        name=kg.name + "-anon",
        ent2txt=anon,
        rel2txt=dict(kg.rel2txt),
        train=list(kg.train),
        test=list(kg.test),
        valid=list(kg.valid),
    )


def shuffle_surface_forms(kg: KG, seed: int = 42) -> KG:
    """
    ★★ THE CONTROL THAT ANSWERS THE ONE FATAL OBJECTION.

    A reviewer will say of `anonymise`:

        "Replacing names with entity4471 destroys ALL information, so of course
         accuracy collapses. That tells us nothing about memorisation."

    They would have a point. This control removes it.

    We keep every real name in the graph and only PERMUTE which entity holds
    which. `dog` might become `Reykjavik`. So:

        vocabulary            identical
        name lengths          identical
        token distribution    identical
        readability           identical
        name <-> entity bond  DESTROYED

    If accuracy also collapses here, the collapse cannot be "opaque ids are
    unreadable" -- the ids are perfectly readable English. It can only be that
    the model was relying on WHICH name went WHERE, which is the definition of
    surface-form memorisation.

    Three outcomes, all interpretable:

      shuffled ~= anonymised   -> confirmed: the binding was the whole signal
      shuffled ~= real         -> the model never used names; anonymisation
                                  destroyed something else (investigate)
      in between               -> quantifies how much is binding vs readability

    ⚠️ The permutation is DERANGED (no entity keeps its own name) and seeded, so
    it is reproducible and no entity is accidentally left un-shuffled.
    """
    import random
    ids = sorted(kg.ent2txt)
    names = [kg.ent2txt[e] for e in ids]
    rng = random.Random(seed)

    order = list(range(len(ids)))
    for _ in range(200):                       # derangement by rejection
        rng.shuffle(order)
        if all(i != j for i, j in enumerate(order)):
            break
    else:                                       # tiny graph: rotate instead
        order = order[1:] + order[:1]

    shuffled = {ids[i]: names[order[i]] for i in range(len(ids))}
    return KG(
        name=kg.name + "-shuf",
        ent2txt=shuffled,
        rel2txt=dict(kg.rel2txt),
        train=list(kg.train),
        test=list(kg.test),
        valid=list(kg.valid),
    )
