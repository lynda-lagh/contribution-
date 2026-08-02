"""
Graph evidence panel -- backend.

⚠️ READ THIS BEFORE BUILDING THE VISUALISATION
----------------------------------------------
You asked to show "how the LLM searches for the answer". It does not search.

Your model is a fine-tuned generative LLM: it reads a prompt and emits tokens.
There is no graph traversal, no path expansion, no retrieval step. Animating a
walk over the KG would depict a process that does not happen, and a jury member
who understands the method will notice.

(MKGL makes the same point from the other side: putting graph context in the
prompt cost 10.6x the GPU-hours for LOWER accuracy -- which is exactly why this
pipeline does not traverse anything at inference.)

WHAT IS REAL, AND WORTH SHOWING
-------------------------------
Every panel below renders evidence a decision ACTUALLY used:

  1. ROUTING CONTEXT      the entity's neighbourhood, degree, type, label quality
                          -- the features the router consulted (trace 1)
  2. TYPE CONSTRAINT      the relation's observed domain/range, and whether the
                          prediction falls inside it (trace 3 -- the type-2 check)
  3. SAMPLED CANDIDATES   where the k sampled answers sit relative to the gold
                          (trace 2 -- the abstention reason, made visible)
  4. CLOSED-WORLD CHECK   a plausible-but-not-gold prediction, shown in context so
                          a human can judge whether the benchmark is wrong rather
                          than the model

That is a graph panel that ARGUES something. A traversal animation is decoration.

BACKENDS
--------
  memory  (default) -- reads the TSVs, builds an adjacency index. A 1-2 hop
                       neighbourhood over 123k entities is trivial; no server.
  neo4j   (optional) -- same interface, Cypher underneath. Use only if your
                       supervisor requires it: it adds a service that can fail
                       during a defence and buys nothing this app needs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from src.data.loaders import KG
from src.data.negatives import build_relation_type_index


@dataclass
class SubGraph:
    nodes: list[dict] = field(default_factory=list)   # {id,label,group,title}
    edges: list[dict] = field(default_factory=list)   # {source,target,label,group}

    def add_node(self, nid: str, label: str, group: str = "context", title: str = "") -> None:
        if not any(n["id"] == nid for n in self.nodes):
            self.nodes.append({"id": nid, "label": label[:40], "group": group,
                               "title": title or label})

    def add_edge(self, s: str, t: str, label: str, group: str = "context") -> None:
        self.edges.append({"source": s, "target": t, "label": label, "group": group})


class GraphBackend(Protocol):
    def neighbourhood(self, entity: str, hops: int = 1, limit: int = 25) -> SubGraph: ...
    def relation_range(self, relation: str) -> set[str]: ...
    def degree(self, entity: str) -> int: ...


# ------------------------------------------------------------------ memory
class MemoryBackend:
    """Default. No server, no Cypher, no dependency that can break mid-defence."""

    def __init__(self, kg: KG):
        self.kg = kg
        self.out: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.inc: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for t in kg.train:
            self.out[t.head].append((t.relation, t.tail))
            self.inc[t.tail].append((t.relation, t.head))
        self.type_index = build_relation_type_index(kg)

    def _lab(self, e: str) -> str:
        return self.kg.ent2txt.get(e, e).split(",")[0]

    def _rlab(self, r: str) -> str:
        return self.kg.rel2txt.get(r, r)

    def degree(self, entity: str) -> int:
        return len(self.out.get(entity, [])) + len(self.inc.get(entity, []))

    def relation_range(self, relation: str) -> set[str]:
        return self.type_index.get(relation, (set(), set()))[1]

    def neighbourhood(self, entity: str, hops: int = 1, limit: int = 25) -> SubGraph:
        g = SubGraph()
        g.add_node(entity, self._lab(entity), "focus",
                   f"{self._lab(entity)}\ndegree={self.degree(entity)}")
        frontier, seen = [entity], {entity}
        for _ in range(hops):
            nxt = []
            for e in frontier:
                for r, t in self.out.get(e, [])[:limit]:
                    g.add_node(t, self._lab(t), "context")
                    g.add_edge(e, t, self._rlab(r))
                    if t not in seen:
                        seen.add(t); nxt.append(t)
                for r, h in self.inc.get(e, [])[:limit]:
                    g.add_node(h, self._lab(h), "context")
                    g.add_edge(h, e, self._rlab(r))
                    if h not in seen:
                        seen.add(h); nxt.append(h)
            frontier = nxt[:limit]
        return g


# ------------------------------------------------------------------ neo4j
class Neo4jBackend:
    """
    Same interface, Cypher underneath. Only worth it if required.

        pip install neo4j
        Neo4jBackend("bolt://localhost:7687", "neo4j", "password", kg)

    Load once:
        LOAD CSV FROM 'file:///train.tsv' AS row FIELDTERMINATOR '\\t'
        MERGE (h:Entity {id: row[0]}) MERGE (t:Entity {id: row[2]})
        MERGE (h)-[:REL {name: row[1]}]->(t)
    """

    def __init__(self, uri: str, user: str, password: str, kg: KG):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.kg = kg
        self.type_index = build_relation_type_index(kg)

    def _lab(self, e: str) -> str:
        return self.kg.ent2txt.get(e, e).split(",")[0]

    def degree(self, entity: str) -> int:
        q = "MATCH (e:Entity {id:$id})-[r]-() RETURN count(r) AS d"
        with self.driver.session() as s:
            rec = s.run(q, id=entity).single()
            return rec["d"] if rec else 0

    def relation_range(self, relation: str) -> set[str]:
        return self.type_index.get(relation, (set(), set()))[1]

    def neighbourhood(self, entity: str, hops: int = 1, limit: int = 25) -> SubGraph:
        q = (f"MATCH p=(e:Entity {{id:$id}})-[r:REL*1..{max(1,min(hops,2))}]-(n:Entity) "
             f"RETURN p LIMIT $limit")
        g = SubGraph()
        g.add_node(entity, self._lab(entity), "focus")
        with self.driver.session() as s:
            for rec in s.run(q, id=entity, limit=limit):
                for rel in rec["p"].relationships:
                    a, b = rel.start_node["id"], rel.end_node["id"]
                    g.add_node(a, self._lab(a), "context")
                    g.add_node(b, self._lab(b), "context")
                    g.add_edge(a, b, rel.get("name", "REL"))
        return g

    def close(self) -> None:
        self.driver.close()


# ------------------------------------------------------------------ evidence
def evidence_subgraph(backend: GraphBackend, head: str, relation: str,
                      predictions: list[str], gold: str | None = None,
                      hops: int = 1, limit: int = 15) -> tuple[SubGraph, dict]:
    """
    THE panel: the query's neighbourhood, plus every sampled candidate coloured by
    what the type check says about it.

        gold             the benchmark answer
        prediction_ok    predicted, and inside the relation's observed range
        type_violation   ★ predicted, real entity, OUTSIDE the range -> type-2
        oov              ★ predicted string is not an entity at all -> type-1

    The colouring IS trace 3 -- the type-check reason -- made visual.
    """
    g = backend.neighbourhood(head, hops=hops, limit=limit)
    rng = backend.relation_range(relation)
    kg: KG = backend.kg

    def lab(e: str) -> str:
        return kg.ent2txt.get(e, e).split(",")[0]

    # ⚠️ must match the NAME, not the full "name, description" entry -- see
    # src/eval/hallucination.py::build_surface_index for why.
    from src.eval.hallucination import build_surface_index
    surface = build_surface_index(kg)
    stats = {"in_range": 0, "type_violation": 0, "oov": 0}

    if gold:
        g.add_node(gold, lab(gold), "gold", f"GOLD: {lab(gold)}")
        g.add_edge(head, gold, kg.rel2txt.get(relation, relation), "gold")

    for p in predictions:
        eid = surface.get(p.strip().lower())
        if eid is None:
            nid = f"__oov__{p[:24]}"
            g.add_node(nid, p[:24], "oov", f"NOT A KG ENTITY: '{p}'")
            g.add_edge(head, nid, "predicted", "oov")
            stats["oov"] += 1
            continue
        if eid == gold:
            continue
        if rng and eid not in rng:
            g.add_node(eid, lab(eid), "type_violation",
                       f"TYPE VIOLATION: '{lab(eid)}' never observed in the range "
                       f"of '{relation}' (|range|={len(rng)})")
            g.add_edge(head, eid, "predicted", "type_violation")
            stats["type_violation"] += 1
        else:
            g.add_node(eid, lab(eid), "prediction_ok",
                       f"type-valid prediction, not the gold answer "
                       f"-- possible closed-world artefact")
            g.add_edge(head, eid, "predicted", "prediction_ok")
            stats["in_range"] += 1

    stats |= {"range_size": len(rng), "head_degree": backend.degree(head),
              "n_nodes": len(g.nodes), "n_edges": len(g.edges)}
    return g, stats


def make_backend(kg: KG, kind: str = "memory", **kw) -> GraphBackend:
    if kind == "neo4j":
        return Neo4jBackend(kw.get("uri", "bolt://localhost:7687"),
                            kw.get("user", "neo4j"), kw.get("password", "neo4j"), kg)
    return MemoryBackend(kg)
