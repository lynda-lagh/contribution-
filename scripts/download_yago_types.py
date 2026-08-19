"""
Download YAGO 3's taxonomy themes and extract the type files.

    python -m scripts.download_yago_types --out /kaggle/working/yago3

WHAT THIS FETCHES, AND HOW BIG IT IS
------------------------------------
yago-knowledge.org does not publish `yagoSimpleTypes.tsv` on its own. The TSV
release is ONE 7-Zip archive containing every "theme" — taxonomy, labels,
dates, sources — and the type files live inside it:

    https://yago-knowledge.org/data/yago3/yago-3.0.2-native.7z      (official)
    https://yago-knowledge.org/data/yago3/yago-3.0.3-native.7z      (2022 revival)

⚠️ The archive is several GB. Kaggle allows it, but budget the time and disk.
   We extract ONLY the members whose names look like type themes and delete
   the archive afterwards unless --keep is passed.

FORMAT, as documented by the YAGO project:
    5 columns — fact id, subject, predicate, object, numeric value
    <id_42>  <Elvis_Presley>  rdf:type  <wikicategory_American_rock_singers>
Class names are `<wordnet_XXX_synsetid>` (the WordNet layer) or
`<wikicategory_...>` / `<wikicat_...>` (the Wikipedia-category layer).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://yago-knowledge.org/data/yago3/yago-3.0.2-native.7z",
    "https://yago-knowledge.org/data/yago3/yago-3.0.3-native.7z",
]
# ★ ONE member, not eight. In PREFERENCE order — the first that exists wins.
#
#   Kaggle gives 20 GB of /kaggle/working and the archive alone is 10.9 GB, so
#   there is ~9 GB of headroom. Extracting all eight type themes blew past it:
#       OSError: [Errno 28] No space left on device
#
#   yagoSimpleTypes is the one we want anyway — ONE leaf class per entity.
#   The others are far larger and useless here:
#       yagoTransitiveType   every entity x every ANCESTOR class (huge)
#       yagoTaxonomy         class -> class, no entities at all
#       *Sources             provenance, not types
PREFERRED = ("yagoSimpleTypes.tsv", "yagoTypes.tsv", "yagoTransitiveType.tsv")
EXCLUDE = ("source", "taxonomy", "geonames")
UA = "kgc-adaptation-thesis/1.0 (academic)"


def download(url: str, dest: Path) -> bool:
    """Stream to disk with a progress line. Returns False on any HTTP failure."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  already have {dest} ({dest.stat().st_size / 1e9:.2f} GB) — skipping")
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r, dest.open("wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {got / 1e9:5.2f} / {total / 1e9:5.2f} GB "
                          f"({got / total:5.1%})", end="", flush=True)
                elif got % (1 << 26) < (1 << 20):
                    print(f"\r  {got / 1e9:5.2f} GB", end="", flush=True)
        print()
        return True
    except Exception as exc:                                    # noqa: BLE001
        print(f"\n  ✗ {type(exc).__name__}: {exc}")
        dest.unlink(missing_ok=True)
        return False


def _pick(members: list[str], override: str | None = None) -> list[str]:
    """Exactly ONE member: the first PREFERRED name present, or --member."""
    if override:
        hit = [m for m in members if m.endswith(override)]
        return hit[:1]
    for name in PREFERRED:
        hit = [m for m in members
               if m.endswith(name) and not any(x in m.lower() for x in EXCLUDE)]
        if hit:
            return hit[:1]
    return []


def _check_space(out: Path, member: str, need_gb: float = 3.0) -> None:
    """Refuse to start an extraction that will obviously fill the disk."""
    free = shutil.disk_usage(out).free / 1e9
    print(f"  free space: {free:.1f} GB")
    if free < need_gb:
        raise SystemExit(
            f"\n✋ only {free:.1f} GB free — extracting {member} needs ~{need_gb} GB.\n"
            f"   Delete the archive first, or free space:\n"
            f"     rm /kaggle/working/yago3/*.7z\n"
            f"   ...but then the download has to run again. Better: keep the\n"
            f"   archive on a Kaggle Dataset and extract straight from there.")


def extract_types(archive: Path, out: Path,
                  member: str | None = None) -> list[Path]:
    """Pull ONE type-bearing member out of the 7z."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        import py7zr
    except ImportError:
        if shutil.which("7z"):
            names = subprocess.run(["7z", "l", "-ba", "-slt", str(archive)],
                                   capture_output=True, text=True).stdout
            have = [ln.split("= ", 1)[1] for ln in names.splitlines()
                    if ln.startswith("Path = ")]
            want = _pick(have, member)
            if not want:
                raise SystemExit("no type-like members found in the archive")
            print(f"  extracting {len(want)} member(s) with 7z")
            subprocess.run(["7z", "x", str(archive), f"-o{out}", "-y", *want],
                           check=True)
            return sorted(p for p in out.rglob("*") if p.is_file())
        raise SystemExit(
            "need py7zr or the 7z binary to open a .7z archive:\n"
            "    pip install py7zr        # or:  apt-get install -y p7zip-full")

    with py7zr.SevenZipFile(archive, "r") as z:
        members = z.getnames()
        want = _pick(members, member)
        if not want:
            print("  members found:", ", ".join(members[:20]))
            raise SystemExit("no type-like members in the archive — inspect the "
                             "list above and pass one with --member")
        _check_space(out, want[0])
        print(f"  extracting 1 of {len(members)} members: {want[0]}")
        z.extract(path=out, targets=want)
    return sorted(p for p in out.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/yago3",
                    help="where to put the archive and the extracted themes")
    ap.add_argument("--url", default=None, help="override the mirror list")
    ap.add_argument("--keep", action="store_true", help="keep the .7z afterwards")
    ap.add_argument("--member", default=None,
                    help="extract this member instead of yagoSimpleTypes.tsv")
    ns = ap.parse_args()

    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    urls = [ns.url] if ns.url else MIRRORS

    archive = None
    for url in urls:
        dest = out / url.rsplit("/", 1)[-1]
        print(f"→ {url}")
        if download(url, dest):
            archive = dest
            break
    if archive is None:
        raise SystemExit(
            "\n✋ every mirror failed. Download by hand from\n"
            "     https://yago-knowledge.org/downloads/yago-3\n"
            "   upload the .7z (or the extracted type file) as a Kaggle Dataset,\n"
            "   and point --from-yago at it.")

    files = extract_types(archive, out / "themes", ns.member)
    if not ns.keep:
        archive.unlink(missing_ok=True)
        print(f"  removed {archive.name} (pass --keep to retain it)")

    print("\nextracted:")
    best = None
    for p in files:
        n = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
        print(f"  {p}   {p.stat().st_size / 1e6:,.0f} MB · {n:,} lines")
        if best is None or "simple" in p.name.lower():
            best = p

    # ★ VERIFY, don't assume. Parse a few lines with the real parser: a file
    #   that downloads cleanly but parses to nothing is the failure mode that
    #   looks like "YAGO doesn't have types for these entities".
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.routing.semantic_types import parse_yago_line
    ok = 0
    with best.open(encoding="utf-8", errors="replace") as f:
        sample = [next(f, "") for _ in range(200)]
    for ln in sample:
        if parse_yago_line(ln):
            ok += 1
    print(f"\nparse check on {best.name}: {ok}/200 sample lines yielded "
          f"(entity, class)")
    if ok < 20:
        print("  ✗ the parser does not understand this file. First 3 lines:")
        for ln in sample[:3]:
            print(f"      {ln.rstrip()[:150]}")
        raise SystemExit("refusing to continue with a file we cannot read")

    print(f"\n✓ next:\n"
          f"    python -m scripts.fetch_yago_types --dataset YAGO3-10 \\\n"
          f"        --from-yago {best} --min-coverage 0.90")


if __name__ == "__main__":
    main()
