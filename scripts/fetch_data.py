"""
Fetch the four KGs directly from KG-LLM. No manual upload, no Kaggle Dataset.

    python -m scripts.fetch_data                      # all four
    python -m scripts.fetch_data --datasets WN11 FB13 # just what you need now

Source: github.com/yao8839836/kg-llm  (Yao, Peng, Mao & Luo, ICASSP 2025)
The entity/relation descriptions there are KG-BERT's -- the field standard, and
the same text APE and others use. Taking them from the origin repo rather than
re-deriving them is what keeps our numbers commensurable with the published ones.

Per dataset we need exactly four files:
    entity2text.txt    entity_id   \\t surface text
    relation2text.txt  relation_id \\t surface text
    train.tsv          head \\t relation \\t tail
    test.tsv           head \\t relation \\t tail \\t label      label in {1,-1}

⚠️ On Kaggle this needs Internet ON (Settings > Internet). It is also worth
running ONCE and then saving /kaggle/working, because re-downloading every
session wastes quota -- but unlike a manual upload, nothing breaks if you forget.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KGLLM_REPO = "https://github.com/yao8839836/kg-llm.git"
ALL_DATASETS = ("WN11", "FB13", "WN18RR", "YAGO3-10")
REQUIRED = ("entity2text.txt", "relation2text.txt", "train.tsv", "test.tsv")


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return r.stdout


def _count_lines(p: Path) -> int:
    with p.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def fetch(datasets: tuple[str, ...], dest_root: str = "data",
          keep_clone: bool = False) -> dict:
    dest_root_p = Path(dest_root)
    dest_root_p.mkdir(parents=True, exist_ok=True)

    todo = [d for d in datasets
            if not all((dest_root_p / d / f).exists() for f in REQUIRED)]
    if not todo:
        print(f"[fetch] all {len(datasets)} datasets already present in {dest_root_p}/")
        return verify(datasets, dest_root)

    tmp = Path(tempfile.mkdtemp(prefix="kgllm_"))
    try:
        print(f"[fetch] cloning {KGLLM_REPO} (shallow) ...")
        # --depth 1: we want the files, not 12 commits of history
        _run(["git", "clone", "--depth", "1", KGLLM_REPO, str(tmp / "kg-llm")])

        src_root = tmp / "kg-llm" / "data"
        if not src_root.is_dir():
            raise SystemExit(f"[fetch] no data/ directory in the clone at {src_root}")

        available = sorted(p.name for p in src_root.iterdir() if p.is_dir())
        print(f"[fetch] datasets available upstream: {available}")

        for name in todo:
            src = src_root / name
            if not src.is_dir():
                print(f"  [SKIP] {name}: not found upstream "
                      f"(upstream has {available})")
                continue
            dst = dest_root_p / name
            dst.mkdir(parents=True, exist_ok=True)
            for fname in REQUIRED:
                s = src / fname
                if not s.exists():
                    print(f"  [WARN] {name}/{fname} missing upstream")
                    continue
                shutil.copy2(s, dst / fname)
            print(f"  [ok]   {name} -> {dst}")
    finally:
        if keep_clone:
            print(f"[fetch] clone kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    return verify(datasets, dest_root)


def verify(datasets: tuple[str, ...], dest_root: str = "data") -> dict:
    """
    Confirm the files are present AND non-trivial. A silently truncated or
    empty file is worse than a missing one -- it trains without complaint.
    """
    root = Path(dest_root)
    report: dict = {}
    print("\n" + "=" * 66)
    print(f"{'dataset':12s} {'entities':>10s} {'relations':>10s} "
          f"{'train':>10s} {'test':>10s}")
    print("=" * 66)

    problems: list[str] = []
    for name in datasets:
        d = root / name
        missing = [f for f in REQUIRED if not (d / f).exists()]
        if missing:
            print(f"{name:12s}  MISSING {missing}")
            problems.append(f"{name}: missing {missing}")
            continue
        counts = {f: _count_lines(d / f) for f in REQUIRED}
        report[name] = counts
        print(f"{name:12s} {counts['entity2text.txt']:>10,d} "
              f"{counts['relation2text.txt']:>10,d} "
              f"{counts['train.tsv']:>10,d} {counts['test.tsv']:>10,d}")
        for f, n in counts.items():
            if n == 0:
                problems.append(f"{name}/{f} is EMPTY")

        # test.tsv must carry the +1/-1 label in column 4 for WN11/FB13,
        # or triple classification has no negatives to score.
        first = (d / "test.tsv").open(encoding="utf-8").readline().rstrip("\n")
        ncol = len(first.split("\t"))
        if name in ("WN11", "FB13") and ncol < 4:
            problems.append(
                f"{name}/test.tsv has {ncol} columns, expected 4 (label in col 4)")

    print("=" * 66)
    if problems:
        print("\n⚠️ PROBLEMS:")
        for p in problems:
            print(f"   - {p}")
        print("\nDo not train on this. Re-run the fetch, or check upstream.")
    else:
        print("\n✅ all datasets present and well-formed")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS),
                    help=f"subset of {ALL_DATASETS}")
    ap.add_argument("--dest", default="data")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--keep-clone", action="store_true",
                    help="leave the temporary kg-llm clone on disk for inspection")
    ns = ap.parse_args()

    ds = tuple(ns.datasets)
    if ns.verify_only:
        verify(ds, ns.dest)
    else:
        fetch(ds, ns.dest, keep_clone=ns.keep_clone)


if __name__ == "__main__":
    main()
