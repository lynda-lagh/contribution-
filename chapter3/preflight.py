
"""
★ PREFLIGHT — does the code on this machine match the code the tests expect?

    python -m chapter3.preflight

WHY THIS EXISTS
---------------
`chapter3/` and `src/` are committed separately. If one is pushed and the other
is not, the notebook pulls a repository in which new tests run against old
library code, and the first symptom is a bare traceback:

    AttributeError: 'KG' object has no attribute 'valid'

That message names a symptom, not a cause. The cause is always the same: a file
that exists locally was never committed. This module checks every cross-module
assumption the pipeline makes and reports the missing commit by name.

⚠️ Run this BEFORE the smoke tests. A failed capability check means the tests
   below it are testing the wrong code, and their results are meaningless.
"""
from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path

# capability -> (how to test it, which file supplies it, why it matters)
CHECKS = []


def cap(name: str, path: str, why: str):
    def deco(fn):
        CHECKS.append((name, fn, path, why))
        return fn
    return deco


@cap("KG.valid field", "src/data/loaders.py",
     "filtered ranking must exclude true triples living in the valid split")
def _kg_valid():
    from src.data.loaders import KG
    return "valid" in getattr(KG, "__dataclass_fields__", {})


@cap("KG.all_true()", "src/data/loaders.py",
     "candidates.py filters negatives against train u valid u test")
def _kg_alltrue():
    from src.data.loaders import KG
    return hasattr(KG, "all_true")


@cap("GraphIndex (inductive neighbours)", "chapter3/sources.py",
     "without it unseen test entities get NO neighbour block and the ladder collapses")
def _graphindex():
    from chapter3 import sources
    return hasattr(sources, "GraphIndex") and hasattr(sources, "assert_no_leak")


@cap("frozen candidate sets", "chapter3/candidates.py",
     "policies must rank against identical negatives by construction")
def _candidates():
    from chapter3 import candidates
    return all(hasattr(candidates, x) for x in ("build", "load", "fingerprint", "qid"))


@cap("paired bootstrap", "chapter3/stats.py",
     "verdicts at +/-0.005 MRR sit under the noise floor without it")
def _stats():
    from chapter3 import stats
    return all(hasattr(stats, x) for x in ("paired_bootstrap", "align", "bootstrap_ci"))


@cap("direction-aware prompts", "chapter3/data.py",
     "head prediction needs prefix/suffix straddling the predicted slot")
def _direction():
    from chapter3 import data
    return "direction" in inspect.signature(data.build_one).parameters


@cap("relation task + confusion matrix", "chapter3/evaluate.py",
     "the only place a genuine F1 exists; link-task F1 equals hits@1")
def _confusion():
    from chapter3 import evaluate
    return hasattr(evaluate, "confusion") and hasattr(evaluate, "per_relation_hits1")


@cap("live display helpers", "chapter3/live.py",
     "the notebook imports step/run/leaderboard/panel")
def _live():
    from chapter3 import live
    return all(hasattr(live, x) for x in ("step", "run", "leaderboard", "panel"))


@cap("qualitative side-by-side", "chapter3/qualitative.py",
     "the case-study table for the paper")
def _qual():
    import chapter3.qualitative as q
    return hasattr(q, "load_cell")


@cap("training CLI", "src/train/sft_cli.py",
     "the notebook trains the shared model through a subprocess")
def _sftcli():
    return importlib.util.find_spec("src.train.sft_cli") is not None


@cap("CATS split fetcher", "scripts/fetch_cats_splits.py",
     "there is no inductive dataset without it")
def _fetch():
    return importlib.util.find_spec("scripts.fetch_cats_splits") is not None


def main() -> int:
    print("=" * 76)
    print("  PREFLIGHT — is the checked-out code consistent?")
    print("=" * 76)

    try:
        head = subprocess.run(["git", "log", "-1", "--pretty=%h %ad %s",
                               "--date=short"], capture_output=True,
                              text=True).stdout.strip()
        print(f"  HEAD: {head}\n")
    except Exception:
        pass

    missing = {}
    for name, fn, path, why in CHECKS:
        try:
            ok = bool(fn())
            err = ""
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"
        print(f"  {'✅' if ok else '❌'}  {name:34s} {path}")
        if not ok:
            missing.setdefault(path, []).append((name, why, err))

    if not missing:
        print(f"\n  all {len(CHECKS)} capabilities present — the repository is consistent\n")
        return 0

    print("\n" + "=" * 76)
    print("  ✋ STALE OR MISSING FILES")
    print("=" * 76)
    for path, items in missing.items():
        exists_here = Path(path).exists()
        print(f"\n  {path}   ({'present but OLD' if exists_here else 'ABSENT'})")
        for name, why, err in items:
            print(f"      missing: {name}")
            print(f"      needed:  {why}")
            if err:
                print(f"      error:   {err[:90]}")

    print("\n" + "-" * 76)
    print("  This is a COMMIT problem, not a code problem. On your computer:\n")
    print("    cd <repo>")
    print("    git add " + " ".join(sorted(missing)))
    print("    git commit -m 'sync library code with chapter3 pipeline'")
    print("    git push")
    print("\n  Then re-run the setup cell. Nothing below this point is meaningful")
    print("  until every capability above is green: the tests would be exercising")
    print("  old library code and reporting results that do not describe your run.")
    print("-" * 76)
    return 1


if __name__ == "__main__":
    sys.exit(main())
