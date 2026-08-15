# ── UPDATE CODE FROM GITHUB, THEN PROVE THE FIX LANDED ─────────────────────
# Safe to re-run at any point.
#
# ⚠️ This uses `git reset --hard`, which discards local EDITS to tracked files.
#    It does NOT touch data/*/built/, checkpoints/ or results/*.json — those are
#    gitignored, so your downloaded YAGO3-10 and any trained adapters survive.
#    ✋ NEVER run `git clean -fdx` here. That WOULD delete both.

DEST, BRANCH = "/kaggle/working/repo", "main"

import os, socket, subprocess, sys, random, time

try:
    socket.create_connection(("github.com", 443), timeout=10).close()
except OSError:
    raise SystemExit("No network. Settings → Internet → ON, then re-run.")

def sh(*cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return r.stdout.strip()

before = sh("git", "-C", DEST, "rev-parse", "--short", "HEAD", check=False) or "(none)"
sh("git", "-C", DEST, "fetch", "--depth", "1", "origin", BRANCH)
sh("git", "-C", DEST, "reset", "--hard", "FETCH_HEAD")
after = sh("git", "-C", DEST, "rev-parse", "--short", "HEAD")

print(f"repo  {before} → {after}")
print("     ", sh("git", "-C", DEST, "log", "-1", "--pretty=%h  %ad  %s", "--date=short"))

for p in ("data", "checkpoints", "results"):
    full = os.path.join(DEST, p)
    n = sum(len(f) for _, _, f in os.walk(full)) if os.path.isdir(full) else 0
    print(f"      kept {p+'/':14s} {n} files")

# ---- is the perf fix actually in this checkout? ---------------------------
neg = open(f"{DEST}/src/data/negatives.py").read()
dat = open(f"{DEST}/chapter1/data.py").read()
ok = ('kw.get("type_index")' in neg
      and "_sorted_ranges" in neg
      and "neg_index = (build_relation_type_index(kg)" in dat)

if not ok:
    raise SystemExit(
        "\n✋ The negative-sampling fix is NOT in this checkout.\n"
        "   Conditions D and E will take ~1.5 h and ~9 h instead of seconds.\n\n"
        "   Push it from your laptop first:\n"
        "     git add src/data/negatives.py chapter1/data.py\n"
        '     git commit -m "perf: hoist relation type index out of the negative loop"\n'
        "     git push\n"
        "   then re-run this cell.")

print("\nfix present ✓  type index hoisted, ranges pre-sorted")

# ---- and prove it, here, in this environment ------------------------------
sys.path.insert(0, DEST)
for m in [k for k in sys.modules if k.startswith(("src.", "chapter1."))]:
    del sys.modules[m]                       # drop any pre-fix import
from src.data.loaders import KG, Triple
from src.data.negatives import build_relation_type_index, make_negatives

r = random.Random(0)
rels = [f"r{i}" for i in range(37)]
ents = [f"e{i}" for i in range(8000)]
train = [Triple(r.choice(ents), r.choice(rels), r.choice(ents), 1) for _ in range(60_000)]
kg = KG(name="bench", ent2txt={e: e for e in ents}, rel2txt={x: x for x in rels},
        train=train, test=[])

t0 = time.time()
for p in train[:10]:
    make_negatives([p], kg, strategy="type_consistent", seed=1)
old = (time.time() - t0) / 10

idx = build_relation_type_index(kg)
t0 = time.time()
for p in train[:200]:
    make_negatives([p], kg, strategy="type_consistent", seed=1, type_index=idx)
new = (time.time() - t0) / 200

print(f"\nbenchmark on a graph 18x smaller than YAGO3-10:")
print(f"  before  {old*1000:8.2f} ms/negative")
print(f"  after   {new*1000:8.4f} ms/negative      {old/new:,.0f}x faster")
print(f"\n  condition D (10,000 negatives):  {old*10_000/3600:5.2f} h  →  {new*10_000:6.1f} s")
print(f"  condition E (60,000 negatives):  {old*60_000/3600:5.2f} h  →  {new*60_000:6.1f} s")
print("\nD and E are now safe to run. Rebuild them:")
print("  !python -m chapter1.data --condition D E --dataset YAGO3-10")
