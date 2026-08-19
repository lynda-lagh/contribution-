# Cells to run next — Chapter 1

Ranking is where your signal is clean (chance = MRR 0.090, not 0.5). Everything
below extends the ranking table, because that is what the chapter now rests on.

Run **1 → 2 → 3 → 4**. Cell 5 is only needed if you want typed conditions on WN11.

---

## Cell 1 · pull the repo (no data touched)

```python
import subprocess, os, sys
DEST = '/kaggle/working/repo'
if os.path.isdir(f'{DEST}/.git'):
    subprocess.run(['git','-C',DEST,'fetch','--depth','1','origin','main'], check=True)
    subprocess.run(['git','-C',DEST,'reset','--hard','FETCH_HEAD'], check=True)
os.chdir(DEST); sys.path.insert(0, DEST)
print(subprocess.run(['git','-C',DEST,'log','-1','--pretty=%h %s'],
                     capture_output=True, text=True).stdout)
```

---

## Cell 2 · ★ the missing rank runs — D, G, E, and the untuned anchor

**This is the highest-value cell.** D and G have no rank file, and D is the
condition whose classification number (0.7465 → 0.6410 under hard negatives) is
most worth confirming on a metric where chance is 0.09 instead of 0.5.

`--limit 500` × 50-way = 25k forward passes ≈ **20 min each** on a T4.
Budget ~80 minutes for all four.

```python
import subprocess, time
from pathlib import Path
DS = 'YAGO3-10'

def rank(cond, adapter=True, limit=500):
    out = Path('results', f'ch1rank-{DS}-{cond}-P0.json')
    if out.exists():
        print(f'[skip] {cond} — {out.name} exists'); return
    cmd = ['python','-m','chapter1.rank','--dataset',DS,'--condition',cond,
           '--limit',str(limit)]
    if adapter:
        cmd += ['--adapter', f'checkpoints/ch1-{DS}-{cond}']
    print('$', ' '.join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    print(f'   rc={rc}   {(time.time()-t0)/60:.1f} min', flush=True)

# ★ D — hard negatives. The one that matters most.
rank('D')
# ★ G — "do types help when names ARE available?" Its classification number
#        (TPR 0.952 / TNR 0.395) is too skewed to interpret; ranking is not.
rank('G')
# E — expected to sit at chance. Run it to CONFIRM the collapse rather than
#     assume it: a collapsed classifier can still rank, and if E ranks above
#     chance that changes what you write about it.
rank('E')
```

⚠️ If an adapter path is wrong, check `!ls checkpoints/` — the naming has bitten
this project before (`ch1-WN11-lora` vs `ch1-WN11-A`).

---

## Cell 3 · ★ the untuned anchor, and WN11 as a second dataset

The decomposition currently rests on **one** graph. WN11 has A and B adapters
already, so a second replication costs two runs.

The untuned row is the other anchor: it separates *what pretraining already
knew* from *what fine-tuning added*.

```python
# the untuned baseline — no adapter, same protocol
rank('A', adapter=False)      # writes ch1rank-YAGO3-10-A-P0.json ONLY if absent,
                              # so rename first if you want both:
                              #   !mv results/ch1rank-YAGO3-10-A-P0.json \
                              #      results/ch1rank-YAGO3-10-A-tuned.json

# ★ WN11 — replicate the headline on a second graph
DS = 'WN11'
rank('A')     # real names
rank('B')     # anonymised
```

★ **Why WN11 matters here.** Its tuned classification accuracy is 0.9265 — the
strongest number in the chapter. If its ranking memorisation share lands near
YAGO3-10's 94%, the claim stops being a property of one graph.

⚠️ WN11 has no **S** (permuted-names) adapter. Without S you get the two-way
split, not the three-way one. If you have GPU time left, training S on WN11 is
the single most valuable *training* run remaining — it is the control that
answers the reviewer's fatal objection, and right now it exists on one dataset.

---

## Cell 4 · the decomposition table — run this after every new rank file

```python
import json, glob
from pathlib import Path

CH = sum(1/k for k in range(1, 51)) / 50      # 50-way random ranking = 0.0900

def table(ds):
    R = {}
    for f in glob.glob(f'results/ch1rank-{ds}-*.json'):
        d = json.load(open(f))
        R[d['condition']] = d['metrics']
    if not R:
        print(f'{ds}: no rank files yet'); return R
    print(f'\n{"="*74}\n{ds} — 50-way filtered link prediction   '
          f'(chance MRR = {CH:.4f})\n{"="*74}')
    lbl = {'A':'real names (baseline)', 'S':'names PERMUTED (control)',
           'B':'anonymised', 'C':'anonymised + types', 'D':'hard negatives',
           'E':'many negatives', 'G':'real names + types'}
    print(f'{"cond":5s} {"what":26s} {"MRR":>7s} {"H@1":>7s} {"H@10":>7s} {"vs chance":>10s}')
    for c in ['A','S','B','C','D','E','G']:
        if c not in R: continue
        m = R[c]
        flag = '  ← AT CHANCE' if abs(m['MRR']-CH) < 0.02 else ''
        print(f'{c:5s} {lbl[c]:26s} {m["MRR"]:>7.4f} {m["hits@1"]:>7.4f} '
              f'{m["hits@10"]:>7.3f} {m["MRR"]-CH:>+10.4f}{flag}')

    if 'A' in R and 'B' in R:
        A, B = R['A']['MRR'], R['B']['MRR']
        tot = A - CH
        print(f'\n  total skill above chance        {tot:.4f}')
        if 'S' in R:
            S = R['S']['MRR']
            print(f'  A→S  name↔entity binding        {A-S:.4f}   {(A-S)/tot:5.1%}')
            print(f'  S→B  readable names in general  {S-B:.4f}   {(S-B)/tot:5.1%}')
        print(f'  B−chance  ★ real generalisation  {B-CH:.4f}   {(B-CH)/tot:5.1%}')
        print(f'  ★ memorisation share            {(A-B)/tot:.1%}')
    return R

for ds in ['YAGO3-10', 'WN11']:
    table(ds)

print('\n⚠️ Every caption must say "50-way". 50-way Hits@1 is NOT full-ranking Hits@1.')
```

---

## Cell 5 · ⚠️ ONLY if you want typed conditions on WN11

`check_type_leak` came back **MATERIAL LEAK on WN11: tag-only rule = 0.568**
(YAGO3-10 after its fix: 0.513; separation 0.136 vs 0.026).

Nothing is contaminated — only A and B have run on WN11 and neither shows types.
`conditions.check_type_gate()` now **refuses** to start C/D/E/G on WN11 until
this is fixed.

```python
# rewinds to test.original.tsv first, so it can only rewind, never destroy labels
!python -m chapter1.make_test_negatives --dataset WN11 --strategy type_consistent --regenerate
!python -m chapter1.check_type_leak --dataset WN11        # expect < 0.55
```

Then update the measured floor in `chapter1/conditions.py`:

```python
TYPE_TAG_FLOOR = {'YAGO3-10': 0.513, 'WN11': <the new number>}
```

⚠️ Numbers from the two versions of the test set are **not comparable**. Say
which one produced each result.

---

# What NOT to run

| | why |
|---|---|
| more **E** classification | it collapsed — `p_yes` never exceeds 0.5, TPR 0.002. Re-running the same recipe reproduces the collapse |
| **C** on WN11 | blocked by the type gate until Cell 5 |
| more seeds on **classification** | three conditions are degenerate there; seeds on a collapsed model buy nothing. Put seeds on **ranking** instead |

# Priority if GPU time is short

1. **Cell 2, D and G** — fills the two real holes in the ranking table
2. **Cell 3, WN11 A and B** — makes the headline a two-dataset result
3. **train S on WN11** — the control that closes the reviewer's main objection,
   currently on one graph only
