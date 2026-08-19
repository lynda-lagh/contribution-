"""
CHAPTER 1 — context blocks for the inference-only prompt variants P5, P6, P7.

    P5   relation description   what the relation MEANS, written by hand
    P6   K=5 neighbours         KG-LLM's own mechanism, reproduced
    P7   paths                  the chain of links between head and candidate

★★ THE LEAK, AND THE GUARD
--------------------------
Showing "facts near the head entity" is the fastest way to hand the model the
answer without noticing. Two ways it happens:

  1. THE QUERY TRIPLE ITSELF.  (Alastair Sim, diedIn, London) appears in the
     graph, so a naive neighbour list prints it verbatim.  KG-LLM guards this
     one -- "excluding the target entity".

  2. THE MIRROR.  Many graphs store both directions:
         (a, r, b)   and   (b, r, a)
     Excluding only the literal query triple leaves its mirror in place, still
     stating the answer.  This cost real debugging time in chapter 3.

  3. THE TARGET ENTITY BY ANY ROUTE.  Even off the query relation, the gold
     entity's NAME can appear in the neighbour list:
         (Alastair Sim, diedIn,  London)   <- dropped by rule 4
         (Alastair Sim, livesIn, London)   <- still there, still names London
     Seeing the gold's name narrows 50 candidates to 1. KG-LLM guards this
     ("excluding the target entity") and so do we, via `exclude`.
     ⚠️ This uses knowledge of the answer to BUILD the prompt. It is the
     conservative direction -- it only ever removes information -- and it is
     what the published method does, so we match it and say so.

  4. ★ THE RELATION ITSELF, AND ANYTHING THAT MIRRORS IT.  Even with 1 and 2
     removed, ANY edge from the head along the QUERY RELATION names a true
     answer:
         query   (Alastair Sim, diedIn, ?)
         context  Alastair Sim diedIn London          <- the answer
     KG-LLM does NOT guard this one.  We drop every edge on the query relation
     in both directions -- AND every edge on a relation that is an inverse of
     it under a DIFFERENT name:
         query   (London, isLocatedIn, ?)             gold: England
         context  England hasPart London              <- still the answer
     Dropping only `rel == query_relation` catches the first because YAGO
     happens to store mirrors under one name; it does not catch the second.
     `GraphIndex._find_inverses` learns the pairs from the graph instead of
     trusting the storage convention.

`safe_neighbours` implements all four.  `assert_no_leak` proves it on real
data, reports how many inverse pairs were found, and is called by the test
suite and the preflight.
"""
from __future__ import annotations

import random
from collections import defaultdict

from src.data.loaders import KG, Triple


# =============================================================================
#  P5 — relation descriptions.  Written by hand, so EXOGENOUS: they come from
#       a person who knows what the relation means, not from the edges.
# =============================================================================
RELATION_DESCRIPTIONS: dict[str, str] = {
    "wasBornIn": "links a person to the town or city where they were born",
    "diedIn": "links a person to the place where they died",
    "livesIn": "links a person to where they live",
    "isCitizenOf": "links a person to the country whose citizenship they hold",
    "graduatedFrom": "links a person to a university they studied at",
    "hasAcademicAdvisor": "links a researcher to the person who supervised them",
    "worksAt": "links a person to the institution employing them",
    "playsFor": "links an athlete to a sports team they play for",
    "isAffiliatedTo": "links a person to a club or organisation they belong to",
    "isLeaderOf": "links a person to a group they lead",
    "isPoliticianOf": "links a politician to the country they serve",
    "actedIn": "links an actor to a film or show they appeared in",
    "directed": "links a director to a film they directed",
    "created": "links a creator to something they made",
    "edited": "links an editor to a work they edited",
    "wroteMusicFor": "links a composer to a work they scored",
    "hasMusicalRole": "links a musician to the instrument or role they play",
    "influences": "links one person to another whose work they shaped",
    "isMarriedTo": "links two people who are married",
    "hasChild": "links a parent to their child",
    "isInterestedIn": "links a person to a subject they study or care about",
    "isKnownFor": "links a person to what made them notable",
    "hasWonPrize": "links a person to an award or honour they received",
    "participatedIn": "links a person or country to an event they took part in",
    "isLocatedIn": "links a place to the larger place that contains it",
    "happenedIn": "links an event to where it took place",
    "hasCapital": "links a country to its capital city",
    "hasCurrency": "links a country to the money it uses",
    "hasOfficialLanguage": "links a country to a language it uses officially",
    "hasNeighbor": "links a country to a country on its border",
    "dealsWith": "links two countries that trade or cooperate",
    "exports": "links a country to a good it sells abroad",
    "imports": "links a country to a good it buys from abroad",
    "owns": "links an owner to what they own",
    "isConnectedTo": "links two places joined by a route",
    "hasGender": "links a person to their gender",
    "hasWebsite": "links an organisation to its website",
}


def describe_relation(relation: str, kg: KG) -> str:
    """A hand-written meaning, or a plain fallback built from the surface text."""
    d = RELATION_DESCRIPTIONS.get(relation)
    if d:
        return f"Here, '{kg.rel2txt.get(relation, relation)}' {d}."
    # ⚠️ FALLBACK, and it is deliberately weak: it restates the relation rather
    #    than explaining it. If most relations land here, P5 is not really
    #    testing "does a description help" -- say so instead of pretending.
    return (f"Here, '{kg.rel2txt.get(relation, relation)}' is a relation "
            f"between two entities.")


# =============================================================================
#  Graph index — built ONCE.  Rebuilding it per query turned a 3-minute job
#  into a 9-hour one in chapter 3.
# =============================================================================
class GraphIndex:
    def __init__(self, kg: KG, include_test: bool = False,
                 inverse_threshold: float = 0.5,
                 inverse_min_support: int = 100) -> None:
        self.nbrs: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        src = [t for t in (list(kg.train)
                           + (list(kg.test) if include_test else []))
               if t.label != -1]                    # never build context from a
        for t in src:                               # triple labelled FALSE
            self.nbrs[t.head].append((t.relation, t.tail, "out"))
            self.nbrs[t.tail].append((t.relation, t.head, "in"))
        self.inverse = self._find_inverses(src, inverse_threshold,
                                           inverse_min_support)

    @staticmethod
    def _find_inverses(triples, threshold: float,
                       min_support: int) -> dict[str, set[str]]:
        """
        ★ WHICH RELATIONS ARE INVERSES OF WHICH, LEARNED FROM THE GRAPH.

        The guard used to drop only `rel == query_relation`. That catches a
        mirror stored under the SAME name -- (a, r, b) and (b, r, a) -- which is
        how YAGO happens to store most of them, so it worked in practice. It
        does NOT catch a mirror stored under a DIFFERENT name:

            query    (London, isLocatedIn, ?)          gold: England
            context   England hasPart London           <- names the answer

        Relying on the dataset's storage convention is not a guard, it is luck,
        and the paper claims we exclude "every edge on its inverse". So detect
        the pairs: r' is an inverse of r when most (h, r, t) edges have a
        matching (t, r', h) edge. One pass, then one lookup per relation.

        ⚠️ TWO CONDITIONS, BOTH NECESSARY — the first version had neither and
           over-fired on a four-edge test graph, deleting a legitimate
           neighbour. A guard that removes too much is not safe, it is quiet:
           it shrinks the block until P6 degenerates into P0, and the coverage
           guard in chapter1/data.py is what finally complains.

             SUPPORT    a rate cannot be estimated from a handful of edges, so
                        relations below `min_support` are never paired.
             MUTUALITY  r' must mirror r AND r must mirror r'. One-directional
                        co-occurrence is a different relation -- `isCitizenOf`
                        often accompanies `livesIn`, but it is not its inverse
                        and dropping it would cost real context for nothing.
        """
        pair_rels: dict[tuple[str, str], set[str]] = defaultdict(set)
        for t in triples:
            pair_rels[(t.head, t.tail)].add(t.relation)

        support: dict[str, int] = defaultdict(int)
        co: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for t in triples:
            support[t.relation] += 1
            for r2 in pair_rels.get((t.tail, t.head), ()):
                co[t.relation][r2] += 1

        def rate(a: str, b: str) -> float:
            return co.get(a, {}).get(b, 0) / (support.get(a) or 1)

        out: dict[str, set[str]] = {}
        for r, others in co.items():
            if support[r] < min_support:
                continue
            hits = {r2 for r2 in others
                    if support.get(r2, 0) >= min_support
                    and rate(r, r2) >= threshold
                    and rate(r2, r) >= threshold}
            if hits:
                out[r] = hits
        return out

    def inverses_of(self, relation: str) -> set[str]:
        """`relation` itself plus every relation that mirrors it."""
        return {relation} | self.inverse.get(relation, set())

    def degree(self, e: str) -> int:
        return len(self.nbrs.get(e, ()))


# =============================================================================
#  P6 — neighbours, with the three-part guard
# =============================================================================
def safe_neighbours(index: GraphIndex, kg: KG, head: str, query_relation: str,
                    k: int = 5, rng: random.Random | None = None,
                    exclude: set[str] | None = None) -> list[str]:
    """
    Up to `k` facts about `head`, with every route to the answer removed.

    Returns rendered strings like `Steve Wozniak`, matching KG-LLM's format
    (`Giving the neighbors of Steve Jobs: Steve Wozniak|USA|Bill Gates|...`).
    """
    rng = rng or random.Random(42)
    exclude = exclude or set()
    # ★ guard 4: any edge on the QUERY RELATION names a true answer, whichever
    #   direction it points -- and so does any edge on a relation that MIRRORS
    #   it under a different name. `inverses_of` is learned from the graph; the
    #   getattr keeps an index pickled by an older run usable.
    banned = (index.inverses_of(query_relation)
              if hasattr(index, "inverses_of") else {query_relation})
    out: list[str] = []
    for rel, other, _dir in index.nbrs.get(head, ()):
        if rel in banned:
            continue
        if other == head or other in exclude:       # guards 1 and 2
            continue
        out.append(kg.ent2txt.get(other, other))
    uniq = list(dict.fromkeys(out))
    rng.shuffle(uniq)
    return uniq[:k]


def neighbour_block(index: GraphIndex, kg: KG, head: str, query_relation: str,
                    k: int = 5, rng: random.Random | None = None,
                    gold: str | None = None) -> str:
    n = safe_neighbours(index, kg, head, query_relation, k, rng,
                        exclude={gold} if gold else None)
    if not n:
        return ""
    return (f"Giving the neighbors of {kg.ent2txt.get(head, head)}: "
            f"{'|'.join(n)}.")


# =============================================================================
#  P7 — paths from head to candidate, same guard
# =============================================================================
def safe_paths(index: GraphIndex, kg: KG, head: str, tail: str,
               query_relation: str, max_paths: int = 3) -> list[str]:
    """Two-hop chains head -> mid -> tail, never using the query relation
    or any relation that mirrors it (see `GraphIndex._find_inverses`)."""
    banned = (index.inverses_of(query_relation)
              if hasattr(index, "inverses_of") else {query_relation})
    tails = {}
    for rel, other, _d in index.nbrs.get(tail, ()):
        if rel not in banned and other != tail:
            tails.setdefault(other, rel)
    out = []
    for rel, mid, _d in index.nbrs.get(head, ()):
        if rel in banned or mid == head or mid == tail:
            continue
        if mid in tails:
            out.append(f"{kg.ent2txt.get(head, head)} "
                       f"{kg.rel2txt.get(rel, rel)} "
                       f"{kg.ent2txt.get(mid, mid)} "
                       f"{kg.rel2txt.get(tails[mid], tails[mid])} "
                       f"{kg.ent2txt.get(tail, tail)}")
        if len(out) >= max_paths:
            break
    return out


def path_block(index: GraphIndex, kg: KG, head: str, tail: str,
               query_relation: str, max_paths: int = 3) -> str:
    p = safe_paths(index, kg, head, tail, query_relation, max_paths)
    return f"Known connections: {'; '.join(p)}." if p else ""


# =============================================================================
#  The proof that the guard works
# =============================================================================
def assert_no_leak(kg: KG, index: GraphIndex, n: int = 300) -> str:
    """
    Two separate questions, and conflating them is a mistake I already made.

      SAFETY   on the index we will actually score with, does the gold answer
               ever appear in a neighbour block?  Must be zero.

      POWER    does the guard remove the gold when the gold IS reachable?
               Tested on an ADVERSARIAL index built from train + test, where
               the query triple is present by construction.

    The first can pass trivially: on a clean split like WN11 no test fact
    lives in the training graph, so there is nothing to remove and the guard
    correctly does nothing. Treating that as "the guard is vacuous" — which an
    earlier version of this function did — confuses a property of the DATASET
    with a defect in the CODE.
    """
    leaks = 0
    for t in kg.test[:n]:
        if t.label == -1:
            continue
        gold = kg.ent2txt.get(t.tail, t.tail)
        if gold in safe_neighbours(index, kg, t.head, t.relation, k=10**6,
                                   exclude={t.tail}):
            leaks += 1
    if leaks:
        raise AssertionError(
            f"CONTEXT LEAK: the gold answer appears in {leaks} neighbour "
            f"block(s) — the model is being shown the answer")

    # ── POWER: force the leak to exist, then check it is removed ────────────
    adv = GraphIndex(kg, include_test=True)
    reachable = fired = 0
    for t in kg.test[:n]:
        if t.label == -1:
            continue
        gold = kg.ent2txt.get(t.tail, t.tail)
        raw = [kg.ent2txt.get(o, o) for _r, o, _d in adv.nbrs.get(t.head, ())]
        if gold not in raw:
            continue
        reachable += 1
        if gold not in safe_neighbours(adv, kg, t.head, t.relation, k=10**6,
                                       exclude={t.tail}):
            fired += 1
    if reachable and fired < reachable:
        raise AssertionError(
            f"the guard failed to remove the gold answer in "
            f"{reachable - fired}/{reachable} adversarial cases")
    power = (f"guard removed the gold in {fired}/{reachable} forced cases"
             if reachable else
             "no forced case available on this split (train and test are "
             "disjoint), so POWER is untested here — the unit test covers it")
    # ★ report the detected mirrors. If this is 0 on a graph that obviously has
    #   them, the detector is mis-tuned and guard 4 is weaker than claimed —
    #   which is exactly the thing the paper asserts and must be able to show.
    inv = getattr(index, "inverse", {})
    npairs = sum(len(v) - (1 if r in v else 0) for r, v in inv.items())
    examples = "; ".join(f"{r}~{'/'.join(sorted(v - {r}))}"
                         for r, v in list(inv.items())[:3] if v - {r})
    inv_note = (f"{npairs} inverse relation pair(s) detected"
                + (f" ({examples})" if examples else ""))
    return f"0 leaks in {n} queries; {power}; {inv_note}"
