"""
★★ EXOGENOUS SEMANTIC TYPES — the fix for the induced-type problem.

WHY THIS MODULE EXISTS
----------------------
`routing/types.py::induced_types` labels an entity by the (relation, position)
pair it occupies most often:

    cat  ->  _type_of::tail        "usually the tail of a `type of` edge"

That is not a type. It is a summary of the entity's position in the very graph
whose edges we are asking the model to predict. When the query is

    Is this true: domestic cat _type_of  cat [_type_of::tail] ?

the tag restates the question. The circularity is measurable, and it was
measured: a one-line rule -- "if the tail's tag names the query relation, say
yes" -- scored **62.4 %** on YAGO3-10 with no model at all. Regenerating the
test negatives type-consistently pushed that to 51.3 %, which does not make the
tag meaningful; it makes it uninformative in both directions. So condition C
landing at chance may be a property of the construction rather than a finding
about types.

THE DISTINCTION THAT MATTERS
----------------------------
    ENDOGENOUS type   derived from the edges under test        (induced)
    EXOGENOUS  type   derived from a source outside the graph  (this module)

A lexicographer wrote `noun.animal` next to `cat` before your graph existed.
NELL's ontology wrote `concept:athlete:` in front of Michael Jordan. Neither
consulted your train split. That is the property that lets condition C ask the
question the paper wants to ask:

    does knowing WHAT a thing is substitute for knowing WHAT IT IS CALLED?

    python -m src.routing.semantic_types --dataset WN11
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..data.loaders import KG

# =============================================================================
#  WordNet  (WN11, WN18RR)  -- lexicographer supersenses
# =============================================================================
#  WN11    __east_indian_1     lemma `east_indian`, sense 1
#  WN18RR  stool_NN_2          lemma `stool`, POS noun, sense 2
#
#  The sense number is IN the identifier, so the mapping is exact: one id ->
#  one synset -> one supersense. No string matching, no guessing.
_WN11_ID = re.compile(r"^_*(?P<lemma>.+?)_(?P<sense>\d+)$")
_WN18_ID = re.compile(r"^(?P<lemma>.+?)_(?P<pos>NN|VB|JJ|RB)_?(?P<sense>\d+)$")
_POS_MAP = {"NN": "n", "VB": "v", "JJ": "a", "RB": "r"}


def parse_wordnet_id(entity_id: str) -> tuple[str, str | None, int] | None:
    """`__east_indian_1` -> ('east_indian', None, 1); `stool_NN_2` -> ('stool','n',2)."""
    m = _WN18_ID.match(entity_id)
    if m:
        return m["lemma"], _POS_MAP[m["pos"]], int(m["sense"])
    m = _WN11_ID.match(entity_id)
    if m:
        return m["lemma"], None, int(m["sense"])
    return None


def wordnet_types(kg: KG, strict: bool = False) -> dict[str, str]:
    """
    Entity -> WordNet supersense (`noun.animal`, `noun.location`, ...).

    45 noun supersenses + 15 verb ones, assigned by the WordNet lexicographers.
    Requires nltk with the `wordnet` corpus:

        pip install nltk && python -c "import nltk; nltk.download('wordnet')"
    """
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("cat")                      # force the corpus to load
    except Exception as exc:                   # noqa: BLE001
        raise RuntimeError(
            f"WordNet unavailable ({type(exc).__name__}: {exc}).\n"
            f"  pip install nltk\n"
            f'  python -c "import nltk; nltk.download(\'wordnet\'); '
            f"nltk.download('omw-1.4')\"") from exc

    out: dict[str, str] = {}
    exact = approx = 0
    for eid in kg.ent2txt:
        parsed = parse_wordnet_id(eid)
        if parsed is None:
            out[eid] = "OTHER"
            continue
        lemma, pos, sense = parsed
        syn, was_exact = None, False
        # exact: the sense index the identifier names, in the POS it names
        for p in ([pos] if pos else ["n", "v", "a", "r"]):
            cands = wn.synsets(lemma, pos=p)
            if len(cands) >= sense:
                syn, was_exact = cands[sense - 1], True
                break
            if cands and syn is None:
                syn = cands[0]                 # sense index out of range
        # ★ COUNT the approximations. `cands[0]` is a DIFFERENT synset from the
        #   one the identifier names and can carry a different supersense, so a
        #   silent fallback here is a silent accuracy loss. If it is common the
        #   type column is not what the caller thinks it is.
        if syn is None:
            out[eid] = "OTHER"
        else:
            out[eid] = syn.lexname()
            exact += was_exact
            approx += not was_exact
        if strict and syn is not None and not was_exact:
            out[eid] = "OTHER"
    total = exact + approx
    if approx:
        print(f"[wordnet] {exact:,} exact sense matches, {approx:,} approximated "
              f"({approx / max(1, total):.1%} — the identifier named a sense the "
              f"lemma does not have; --strict would drop these instead)")
    return out


# =============================================================================
#  NELL-995  -- the ontology type is inside the identifier
# =============================================================================
#      concept:athlete:michael_jordan   ->  athlete
#      concept:sportsteam:lakers        ->  sportsteam
#
#  NELL assigns these from its own ontology during extraction. They are not a
#  function of any single relation, which is exactly the property we need.
_NELL_ID = re.compile(r"^concept[:_](?P<type>[a-z0-9_]+)[:_]")


def nell_types(kg: KG) -> dict[str, str]:
    """
    Entity -> NELL ontology category, read from the IDENTIFIER prefix.

    ★ Deliberately ignores the surface text. Falling back to `ent2txt` looks
      generous but breaks the property conditions C and G depend on: after
      anonymise() the text is `entity4471`, so an entity whose type was only
      visible in its text resolves on the real graph and to OTHER on the
      anonymised one. The two arms would then carry different tag inventories
      and would no longer be a matched pair. Types must key on the ID, which
      anonymise() and shuffle_surface_forms() both leave untouched.
    """
    return {eid: (m["type"] if (m := _NELL_ID.match(eid)) else "OTHER")
            for eid in kg.ent2txt}


# =============================================================================
#  YAGO3-10  -- needs an EXTERNAL type file. There is no free lunch here.
# =============================================================================
#  ★★ THE CANONICAL SOURCE IS YAGO ITSELF.
#
#  YAGO was built by joining Wikipedia (which supplies the INSTANCES) to
#  WordNet (which supplies the CLASS HIERARCHY). Every YAGO entity therefore
#  already carries a WordNet class:
#
#      <Alastair_Sim>   rdf:type   <wordnet_actor_109765278>
#      <Littlefield,_Texas>        <wordnet_town_108665504>
#
#  YAGO3-10 is a subset of YAGO3 (entities with >= 10 relations), so EVERY one
#  of its 123,182 entities has such a class by construction. That gives near
#  total coverage, where the Wikidata route depends on each Wikipedia title
#  still resolving. It is also the same kind of label WN11 gets -- a WordNet
#  synset written by a lexicographer -- so the two graphs become comparable.
#
#  Download `yagoSimpleTypes.tsv` (YAGO3) from the YAGO site, then:
#      python -m scripts.fetch_yago_types --from-yago yagoSimpleTypes.tsv
#  ── HOW THE LABELS ARE ACTUALLY WRITTEN ──────────────────────────────────
#  YAGO ships several type files and the layout differs between them, so the
#  parser must not assume a fixed column count.
#
#  yagoSimpleTypes.tsv / yagoTypes.tsv   (YAGO 3) — leading fact-id column:
#      <id_1x8vfq_88c_1eeb0x8>  <Alastair_Sim>  rdf:type  <wordnet_actor_109765278>
#  yagoTransitiveType.tsv                (YAGO 3) — no fact id, many rows/entity:
#      <Alastair_Sim>  rdf:type  <wordnet_person_100007846>
#  YAGO 4 / 4.5                          — N-Triples with schema.org classes:
#      <http://yago-knowledge.org/resource/Alastair_Sim> <...#type> <http://schema.org/Person> .
#
#  THREE KINDS OF CLASS APPEAR, and they are NOT equally good:
#      wordnet_actor_109765278      a WordNet synset          ★ best
#      wikicat_1979_films           a Wikipedia category      usable head noun
#      schema.org/Person            a schema.org class        coarse but clean
_WORDNET = re.compile(r"^wordnet_(?P<name>[a-z][a-z_]*?)_\d+$")
# ★ BOTH SPELLINGS. yago-knowledge.org documents the middle taxonomy layer as
#   `<wikicategory_American_rock_singers>`, while YAGO3's shipped TSVs also use
#   the abbreviated `<wikicat_...>`. Matching only one form silently sent every
#   category-typed entity to OTHER on whichever dump you happened to download.
_WIKICAT = re.compile(r"^wikicat(?:egory)?_(?P<name>.+)$")
_SCHEMA = re.compile(r"^https?://schema\.org/(?P<name>\w+)$")
# ★ `/schema/` ONLY. Including `/resource/` here made every YAGO 4 SUBJECT
#   parse as a class, so the subject was consumed before it could be read and
#   every N-Triples line returned None.
_YAGO_SCHEMA = re.compile(r"^https?://yago-knowledge\.org/schema/(?P<name>\w+)$")

# Wikipedia category names are "<modifiers> <HEAD> <preposition> <qualifier>",
# e.g. `English_male_film_actors`, `Cities_in_Texas`, `1979_films`. The head
# noun is what we want; everything after the first preposition is a qualifier.
_PREP = re.compile(r"_(?:in|of|from|by|at|for|about|with|during)_")
_PLURAL = [("ies", "y"), ("sses", "ss"), ("shes", "sh"), ("ches", "ch"),
           ("xes", "x"), ("zes", "z"), ("ses", "s"), ("s", "")]


def _singular(w: str) -> str:
    if w.endswith("ss") or len(w) <= 3:
        return w
    for suf, rep in _PLURAL:
        if w.endswith(suf):
            return w[: -len(suf)] + rep
    return w


def wikicat_head(cat: str) -> str | None:
    """
    `1979_films` -> 'film'   ·   `Cities_in_Texas` -> 'city'
    `English_male_film_actors` -> 'actor'

    ⚠️ A HEURISTIC, unlike the WordNet route. Still EXOGENOUS -- a Wikipedia
    editor wrote the category, not our training edges -- so it does not
    reintroduce the circularity. But it is noisier, and the caller is told how
    many entities were typed this way so the mix can be reported.
    """
    head = _PREP.split(cat, maxsplit=1)[0]
    toks = [t for t in head.split("_") if t and not t.isdigit()]
    # drop trailing past participles: `Populated_places_established` -> places
    while len(toks) > 1 and (toks[-1].endswith("ed") or toks[-1].lower() in
                             {"and", "or", "the", "a", "an"}):
        toks.pop()
    return _singular(toks[-1].lower()) if toks else None


def parse_yago_class(value: str) -> tuple[str, str] | None:
    """Class URI/atom -> (kind, name), or None if it is not a usable class."""
    v = value.strip().strip(".").strip().strip("<>")
    for rx, kind in ((_WORDNET, "wordnet"), (_SCHEMA, "schema"),
                     (_YAGO_SCHEMA, "schema")):
        m = rx.match(v)
        if m:
            n = m["name"]
            return (kind, n.lower()) if kind == "schema" else (kind, n)
    m = _WIKICAT.match(v)
    if m:
        h = wikicat_head(m["name"])
        return ("wikicat", h) if h else None
    return None


def parse_yago_line(line: str) -> tuple[str, tuple[str, str]] | None:
    """
    One TSV/N-Triples row -> (entity_id, (kind, class)), tolerant of layout.

    ★ Does NOT index by column number. yagoSimpleTypes carries a leading
      fact-id, yagoTransitiveType does not, and YAGO 4 is N-Triples — a fixed
      `parts[-3]` silently reads the wrong field on one of the three.
    """
    fields = [f for f in line.rstrip("\n").replace("\t", " ").split(" ") if f]
    if len(fields) < 3:
        return None
    cls = subj = None
    for f in fields:
        c = parse_yago_class(f)
        if c and cls is None:
            cls = c
            continue
        bare = f.strip("<>")
        if (subj is None and cls is None and not bare.startswith("id_")
                and ":" not in bare[:8] and not bare.startswith("http")):
            subj = bare
        elif subj is None and bare.startswith("http") and "/resource/" in bare:
            subj = bare.rsplit("/", 1)[-1]
    return (subj, cls) if (subj and cls) else None

def yago_types(kg: KG, root: str = "data", dataset: str = "YAGO3-10") -> dict[str, str]:
    """
    Entity -> YAGO class, read from `data/{dataset}/entity2type.txt`.

    ⚠️ Unlike WordNet and NELL, YAGO3-10's identifiers carry no type. The names
    are Wikipedia titles (`Steve_Jobs`, `F.C._Alverca`), so an exogenous type
    MUST come from outside the archive. Two honest routes:

      1. YAGO's own `yagoSimpleTypes` dump, filtered to this entity set.
      2. Wikidata `P31` (instance of) via the Wikipedia title.

    Both produce a two-column TSV that this function reads:

        Steve_Jobs \\t Person
        F.C._Alverca \\t SportsTeam

    Deriving YAGO types from relation domain/range instead would put us straight
    back into the endogenous trap this module exists to escape -- so we refuse
    and ask for the file.
    """
    p = Path(root, dataset, "entity2type.txt")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. YAGO3-10 identifiers carry no type, so a semantic\n"
            f"  type must come from an external source. Build the file with\n"
            f"  scripts/fetch_yago_types.py, or fall back to induced types and\n"
            f"  SAY SO in the caption.")
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            e, t = line.split("\t", 1)
            out[e.strip()] = t.strip()
    return {e: out.get(e, "OTHER") for e in kg.ent2txt}


# =============================================================================
#  DISPATCH + COVERAGE
# =============================================================================
# ★ PREFIX rules, not an exact-name table. The old table matched the full
#   dataset name and then "fell back" to `dataset.split("-")[0]`, which for
#   NELL-995-ind yields "NELL" -- not a key. So every NELL variant that worked
#   did so by exact match, and the fallback never fired for anything; meanwhile
#   WN18RR-ind raised even though WordNet types apply to it perfectly well.
#   A prefix rule is what the intent always was.
PROVIDERS: dict[str, object] = {
    "WN11": wordnet_types, "WN18RR": wordnet_types,
    "NELL-995": nell_types, "NELL": nell_types,
    "YAGO3-10": yago_types,
}


def provider_for(dataset: str):
    """Longest matching prefix wins, so `WN18RR-ind` resolves like `WN18RR`."""
    for key in sorted(PROVIDERS, key=len, reverse=True):
        if dataset == key or dataset.startswith(key + "-"):
            return PROVIDERS[key]
    return None


def semantic_types(kg: KG, dataset: str, root: str = "data") -> dict[str, str]:
    """Exogenous types for `dataset`, or raise saying why none are available."""
    fn = provider_for(dataset)
    if fn is None:
        raise KeyError(
            f"no exogenous type source registered for {dataset!r}. "
            f"Known prefixes: {', '.join(sorted(PROVIDERS))}. Using induced "
            f"types instead is a DIFFERENT experiment -- report it as such.")
    # ★ `root` must be threaded through: build_condition takes --root and
    #   yago_types reads a file under it. Defaulting to "data" inside the
    #   provider meant a non-default root was silently ignored.
    return fn(kg, root=root, dataset=dataset) if fn is yago_types else fn(kg)


def coverage(types: dict[str, str]) -> dict:
    """How much of the entity set actually got a type, and how concentrated."""
    n = len(types) or 1
    c = Counter(types.values())
    other = c.get("OTHER", 0)
    named = [(k, v) for k, v in c.most_common() if k != "OTHER"]
    return {
        "n_entities": len(types),
        "n_distinct": len(named),
        "other_rate": other / n,
        "largest_share": (named[0][1] / n) if named else 0.0,
        "top5_share": sum(v for _, v in named[:5]) / n,
        "top10": named[:10],
    }


def main() -> None:
    import argparse

    from ..data.loaders import load_kg
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WN11")
    ap.add_argument("--root", default="data")
    ns = ap.parse_args()

    kg = load_kg(ns.dataset, ns.root)
    t = semantic_types(kg, ns.dataset, **({"root": ns.root} if ns.dataset == "YAGO3-10" else {}))
    r = coverage(t)
    print(f"\n{ns.dataset}: {r['n_entities']:,} entities")
    print(f"  distinct types  {r['n_distinct']}")
    print(f"  OTHER           {r['other_rate']:.1%}"
          + ("   ✋ too high — the condition would be near-vacuous"
             if r["other_rate"] > 0.5 else "   ✓"))
    print(f"  largest type    {r['largest_share']:.1%}")
    print(f"  top 5 together  {r['top5_share']:.1%}")
    print("\n  most common:")
    for k, v in r["top10"]:
        print(f"    {k:28s} {v:7,d}  ({v / max(1, r['n_entities']):5.1%})")
    print("\n  examples:")
    for e in list(kg.ent2txt)[:8]:
        print(f"    {kg.ent2txt[e]!r:34} -> {t.get(e)}")


if __name__ == "__main__":
    main()
