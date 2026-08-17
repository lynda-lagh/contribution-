
"""
★ Fetch the CATS inductive splits — the ones RealKGC also adopts.

WHY THESE SPLITS
----------------
RealKGC, §4: it uses *"the specific dataset versions and splits as processed in
CATS"*. Adopting them makes both of the closest papers directly comparable, and
means split construction is not something a reviewer can question. CATS also
ablates **Qwen2-1.5B**, so our model size is size-matched by their choice.

    github.com/IDEA-FinAI/CATS      (AAAI 2025)

WHAT THIS DOES
--------------
Clones the repo, **inventories what is actually there**, and converts anything
that looks like an inductive split into this project's four-file format:

    train.tsv  valid.tsv  test.tsv        head \\t relation \\t tail
    entity2text.txt  relation2text.txt    id \\t text

⚠️ It does NOT assume a directory layout. Released repos move files between
   revisions, and a fetch script that silently writes to the wrong path produces
   a dataset that validates fine and answers a different question. This one
   prints what it found and refuses to guess.

    python -m scripts.fetch_cats_splits --list              # inventory only
    python -m scripts.fetch_cats_splits --dataset WN18RR --out data/WN18RR-ind
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/IDEA-FinAI/CATS.git"
CACHE = Path(".cache/CATS")
TRIPLE_NAMES = ("train", "valid", "dev", "test", "train_ind", "test_ind",
                "inductive", "ind")


def clone(dest: Path, depth: int = 1) -> None:
    if (dest / ".git").exists():
        print(f"[cats] already cloned at {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cats] cloning {REPO} -> {dest}")
    r = subprocess.run(["git", "clone", "--depth", str(depth), REPO, str(dest)],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(
            f"clone failed:\n{r.stdout}{r.stderr}\n"
            "  If there is no network, download the repo manually and pass\n"
            "  --cache <path-to-extracted-repo>.")


def inventory(root: Path) -> dict[str, list[Path]]:
    """Everything that could plausibly be a split, grouped by parent directory."""
    out: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".txt", ".tsv", ".csv"):
            continue
        if ".git" in p.parts:
            continue
        stem = p.stem.lower()
        if any(k in stem for k in TRIPLE_NAMES) or "entity" in stem or "relation" in stem:
            out.setdefault(str(p.parent.relative_to(root)), []).append(p)
    return out


def sniff(p: Path, n: int = 3) -> tuple[str, list[str]]:
    """Guess the delimiter and show the first lines, so a human can confirm."""
    lines = []
    with p.open(encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line.rstrip("\n"))
    delim = "\t" if any("\t" in l for l in lines) else ","
    return delim, lines


def convert(src: Path, dst: Path, delim: str) -> int:
    """Normalise to head\\trelation\\ttail. Returns rows written."""
    rows = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8", errors="replace") as f, \
         dst.open("w", encoding="utf-8") as o:
        for line in f:
            parts = [x.strip() for x in line.rstrip("\n").split(delim)]
            if len(parts) < 3:
                continue
            o.write("\t".join(parts[:3]) + "\n")
            rows += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--dataset", default=None,
                    help="substring to match, e.g. WN18RR or fb15k")
    ap.add_argument("--out", default=None, help="e.g. data/WN18RR-ind")
    ap.add_argument("--list", action="store_true", help="inventory only, write nothing")
    ap.add_argument("--depth", type=int, default=1)
    ns = ap.parse_args()

    cache = Path(ns.cache)
    clone(cache, ns.depth)

    inv = inventory(cache)
    if not inv:
        raise SystemExit(f"no split-like files found under {cache}")

    print(f"\n{'='*76}\nCATS REPO INVENTORY\n{'='*76}")
    for d, files in inv.items():
        print(f"\n{d}/")
        for p in files:
            delim, lines = sniff(p)
            n = sum(1 for _ in p.open(encoding='utf-8', errors='replace'))
            print(f"   {p.name:32s} {n:>9,d} lines  delim={delim!r}")
            for l in lines[:1]:
                print(f"        {l[:88]}")

    if ns.list or not ns.dataset:
        print("\n★ Pick the directory that matches your dataset, then re-run with")
        print("  --dataset <substring> --out data/<name>-ind")
        print("\n⚠️ Confirm by eye that the files really are an INDUCTIVE split:")
        print("   test entities must NOT appear in train. `chapter3.validate`")
        print("   checks this and exits non-zero — run it immediately after.")
        return

    if not ns.out:
        raise SystemExit("--out is required when --dataset is given")

    matches = {d: fs for d, fs in inv.items() if ns.dataset.lower() in d.lower()}
    if not matches:
        matches = {d: [p for p in fs if ns.dataset.lower() in p.name.lower()]
                   for d, fs in inv.items()}
        matches = {d: fs for d, fs in matches.items() if fs}
    if not matches:
        raise SystemExit(f"nothing matched {ns.dataset!r}. Run with --list.")
    if len(matches) > 1:
        print(f"\n⚠️ {len(matches)} directories matched {ns.dataset!r}:")
        for d in matches:
            print(f"     {d}")
        raise SystemExit("be more specific — refusing to guess which split you meant")

    src_dir, files = next(iter(matches.items()))
    out = Path(ns.out)
    print(f"\n[cats] {src_dir} -> {out}")

    written = {}
    for p in files:
        stem = p.stem.lower()
        delim, _ = sniff(p)
        if "entity" in stem and "text" in stem or stem in ("entity2text",):
            shutil.copy2(p, out / "entity2text.txt"); written["entity2text.txt"] = "copied"
        elif "relation" in stem and "text" in stem or stem in ("relation2text",):
            shutil.copy2(p, out / "relation2text.txt"); written["relation2text.txt"] = "copied"
        else:
            for want, name in (("train", "train.tsv"), ("valid", "valid.tsv"),
                               ("dev", "valid.tsv"), ("test", "test.tsv")):
                if want in stem:
                    n = convert(p, out / name, delim)
                    written[name] = f"{n:,} rows"
                    break

    print("\nwritten:")
    for k, v in sorted(written.items()):
        print(f"   {k:20s} {v}")

    for need in ("train.tsv", "test.tsv"):
        if need not in written:
            print(f"\n✋ {need} was not produced — check the inventory above and "
                  f"map it by hand.")
            sys.exit(1)

    print(f"\n★ NOW RUN, before anything else:")
    print(f"    python -m chapter3.validate --dataset {out.name} --root {out.parent}")
    print("  Chapter 3's premise is that test entities are unseen. A leaky split")
    print("  does not fail loudly — it just produces better numbers and an")
    print("  indefensible chapter.")


if __name__ == "__main__":
    main()
