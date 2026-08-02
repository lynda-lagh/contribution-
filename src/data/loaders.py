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

from dataclasses import dataclass
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

    @property
    def entities(self) -> list[str]:
        return list(self.ent2txt.keys())

    @property
    def relations(self) -> list[str]:
        return list(self.rel2txt.keys())

    def describe(self) -> dict:
        return {
            "dataset": self.name,
            "n_entities": len(self.ent2txt),
            "n_relations": len(self.rel2txt),
            "n_train": len(self.train),
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
    return KG(
        name=dataset,
        ent2txt=_read_map(d / "entity2text.txt"),
        rel2txt=_read_map(d / "relation2text.txt"),
        train=_read_triples(d / "train.tsv", has_label=False),
        test=_read_triples(d / "test.tsv", has_label=True),
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
    )
