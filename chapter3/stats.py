"""
★★ THE NOISE FLOOR — what a difference has to be before it is a result.

THE PROBLEM THIS FILE FIXES
---------------------------
`report.py` originally printed a verdict when two policies differed by 0.005 MRR:

    if abs(dr) < 0.005:  "the DECISIONS add nothing"
    elif d0 > 0.005:     "★★ ALLOCATION PAYS"

At `--limit 300`, the standard error of MRR is roughly 0.02. So **every verdict
in that table could flip on the seed**, and the first reviewer question — "is
0.007 anything?" — has the answer "no, and you had no way to know".

    n = 300 queries      SE(MRR) ≈ 0.02      a 0.005 threshold is NOISE

★ THE FIX IS THE PAIRED BOOTSTRAP, and the pairing is what makes it powerful.
  Every policy is evaluated on the SAME queries against the SAME frozen
  candidates (see candidates.py). So we do not compare two independent means —
  we resample the per-query DIFFERENCE. Query difficulty, which dominates the
  raw variance, cancels out entirely.

      unpaired:  var(A) + var(B)                  large
      paired:    var(A - B)                       small, because hard queries
                                                  are hard for both policies

  In practice this cuts the interval enough to detect differences an unpaired
  test would call noise — which is the honest way to rescue small effects,
  rather than lowering the threshold and hoping.

WHAT TO REPORT
--------------
    MRR = 0.412 [0.389, 0.436]                    a cell
    S4 - S0 = +0.021 [+0.004, +0.038], p = 0.014  ★ the claim

⚠️ A difference whose CI contains 0 is NOT a small effect. It is an effect the
   experiment could not measure, and the honest sentence is "we cannot
   distinguish these", not "S4 is slightly better".

    python -m chapter3.stats --demo        # sanity-check the estimator itself
"""
from __future__ import annotations

import argparse
import math
import random


# --------------------------------------------------------------- basic pieces
def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def sem(xs) -> float:
    """Standard error of the mean — the number that sets the noise floor."""
    xs = list(xs)
    return stdev(xs) / math.sqrt(len(xs)) if xs else 0.0


def rr(ranks) -> list[float]:
    """Reciprocal ranks. MRR is their mean, so bootstrapping these IS MRR."""
    return [1.0 / r for r in ranks]


# ------------------------------------------------------------ single-cell CI
def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """
    Percentile bootstrap CI for the mean of `values`.

    Used for a single cell: pass reciprocal ranks and it returns the MRR with an
    interval. `hits@k` works too — pass the 0/1 indicators.
    """
    v = list(values)
    n = len(v)
    if n == 0:
        return {}
    if n == 1:
        return {"mean": v[0], "lo": v[0], "hi": v[0], "n": 1, "sem": 0.0}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(mean(v[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"mean": mean(v), "lo": lo, "hi": hi, "n": n, "sem": sem(v),
            "n_boot": n_boot}


# ------------------------------------------------------- ★ the paired version
def paired_bootstrap(a, b, n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 0) -> dict:
    """
    ★★ Bootstrap the per-query difference `a - b`.

    `a` and `b` must be ALIGNED: element i of each is the same query. That
    alignment is guaranteed by candidates.py freezing the query set, and
    `align()` below enforces it rather than trusting it.

    Returns the observed difference, its CI, and a two-sided bootstrap p-value.

    ⚠️ The p-value here is the fraction of resampled differences that fall on
       the opposite side of zero, doubled. It is a bootstrap p-value, not a
       t-test — report it as such. With n_boot=2000 the smallest reportable
       value is p < 0.001; do not print "p = 0.0000".
    """
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError(f"paired test needs aligned inputs: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return {}
    d = [x - y for x, y in zip(a, b)]
    obs = mean(d)

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(mean(d[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]

    # two-sided bootstrap p: how often does a resample land past zero?
    n_le = sum(1 for x in boots if x <= 0)
    n_ge = sum(1 for x in boots if x >= 0)
    p = min(1.0, 2.0 * min(n_le, n_ge) / n_boot)

    return {"diff": obs, "lo": lo, "hi": hi, "p": p, "n": n,
            "significant": (lo > 0) or (hi < 0),
            "sem_diff": sem(d), "n_boot": n_boot,
            "n_better": sum(1 for x in d if x > 0),
            "n_worse": sum(1 for x in d if x < 0),
            "n_tied": sum(1 for x in d if x == 0)}


def align(rows_a: list[dict], rows_b: list[dict], key: str = "qid"):
    """
    ★ Pair two cells' per-query rows on query identity, not on list position.

    Returns (rr_a, rr_b, n_common). Refuses silently-wrong comparisons: if the
    two cells share fewer than 90% of their queries, something is misconfigured
    (different --limit, different direction, a stale file) and pairing them
    would compare partly different test sets.
    """
    A = {r[key]: r for r in rows_a}
    B = {r[key]: r for r in rows_b}
    common = sorted(set(A) & set(B))
    if not common:
        raise ValueError("the two cells share no queries — cannot pair them")
    frac = len(common) / max(len(A), len(B))
    if frac < 0.90:
        raise ValueError(
            f"only {len(common)} of {max(len(A), len(B))} queries are shared "
            f"({frac:.0%}). These cells were not evaluated on the same test set; "
            f"pairing them would compare different data. Re-run both with the "
            f"same --limit, --direction, --n-way and --seed.")
    return ([1.0 / A[k]["rank"] for k in common],
            [1.0 / B[k]["rank"] for k in common],
            len(common))


# -------------------------------------------------- ★ how big must it be?
def min_detectable_effect(values, power: float = 0.80, alpha: float = 0.05) -> float:
    """
    ★ The smallest difference this many queries could detect — computed BEFORE
      looking at results, so the verdict thresholds are set honestly.

    For a paired design at 80% power and α=0.05 the rule of thumb is

        MDE ≈ 2.8 × SE(difference)

    ⚠️ Pass the per-query DIFFERENCES if you have a pilot pair; passing a single
       cell's reciprocal ranks gives the UNPAIRED bound, which is conservative
       (larger). Both are useful: the unpaired figure tells you what n you would
       need with no pairing at all.
    """
    z_a = 1.959964                                    # two-sided 0.05
    z_b = 0.8416212 if abs(power - 0.80) < 1e-9 else 1.2815516   # 0.80 / 0.90
    return (z_a + z_b) * sem(values)


def n_for_effect(values, effect: float, power: float = 0.80) -> int:
    """How many queries would be needed to detect `effect`? Answers 'run more'."""
    s = stdev(values)
    if effect <= 0 or s == 0:
        return 0
    z_a, z_b = 1.959964, (0.8416212 if abs(power - 0.80) < 1e-9 else 1.2815516)
    return int(math.ceil(((z_a + z_b) * s / effect) ** 2))


# --------------------------------------------------------------- formatting
def fmt_ci(d: dict, digits: int = 4) -> str:
    if not d:
        return "—"
    return f"{d['mean']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"


def fmt_diff(d: dict, digits: int = 4) -> str:
    if not d:
        return "—"
    p = d["p"]
    ps = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    star = " ★" if d["significant"] else ""
    return (f"{d['diff']:+.{digits}f} [{d['lo']:+.{digits}f}, "
            f"{d['hi']:+.{digits}f}] {ps}{star}")


def verdict(d: dict, label_a: str = "A", label_b: str = "B") -> str:
    """★ The sentence that goes in the paper, chosen by the interval, not the point."""
    if not d:
        return "no data"
    if not d["significant"]:
        return (f"⚠️ {label_a} and {label_b} are INDISTINGUISHABLE at n={d['n']} "
                f"(CI contains 0). Not 'a small gain' — an unmeasurable one.")
    if d["diff"] > 0:
        return (f"★ {label_a} > {label_b} by {d['diff']:+.4f} MRR, CI excludes 0 "
                f"({d['n_better']}/{d['n']} queries improved)")
    return (f"★ {label_a} < {label_b} by {d['diff']:+.4f} MRR, CI excludes 0 "
            f"({d['n_worse']}/{d['n']} queries got worse)")


# -------------------------------------------------------------------- demo
def _demo() -> None:
    """
    Sanity-check the estimator on data with a KNOWN answer.

    ⚠️ A statistical tool nobody has validated is worse than none: it produces
       confident numbers either way. These three cases have known truth.
    """
    rng = random.Random(7)
    print("=" * 76)
    print("STATS SELF-CHECK — three cases with a known answer")
    print("=" * 76)

    # 1. no real difference: the CI must contain 0 most of the time
    hits = 0
    for t in range(200):
        base = [rng.random() for _ in range(300)]
        a = [x + rng.gauss(0, 0.05) for x in base]
        b = [x + rng.gauss(0, 0.05) for x in base]
        d = paired_bootstrap(a, b, n_boot=400, seed=t)
        hits += d["significant"]
    print(f"\n1 · NULL CASE (no true difference)")
    print(f"    false positives {hits}/200 = {hits/200:.1%}   expected ≈ 5%")
    print(f"    {'✓ calibrated' if hits/200 < 0.10 else '✗ MISCALIBRATED'}")

    # 2. a real, small difference: pairing should find what unpaired cannot
    base = [rng.random() for _ in range(300)]
    a = [x + 0.02 + rng.gauss(0, 0.30) for x in base]
    b = [x + rng.gauss(0, 0.30) for x in base]
    dp = paired_bootstrap(a, b, seed=1)
    print(f"\n2 · SMALL TRUE EFFECT (+0.02, heavy per-query noise)")
    print(f"    paired      {fmt_diff(dp)}")
    ca, cb = bootstrap_ci(a, seed=1), bootstrap_ci(b, seed=2)
    unp_lo = ca["lo"] - cb["hi"]
    unp_hi = ca["hi"] - cb["lo"]
    print(f"    unpaired    {ca['mean']-cb['mean']:+.4f} "
          f"[{unp_lo:+.4f}, {unp_hi:+.4f}]  (wider — query difficulty not cancelled)")
    print(f"    ★ pairing narrows the interval by "
          f"{1 - (dp['hi']-dp['lo'])/(unp_hi-unp_lo):.0%}")

    # 3. the noise floor at the sizes we actually run
    print(f"\n3 · NOISE FLOOR at the limits we run")
    print(f"    {'n queries':>10s} {'SE(MRR)':>10s} {'min detectable diff':>22s}")
    for n in (100, 300, 500, 1000, 2000):
        v = [1.0 / rng.randint(1, 50) for _ in range(n)]
        print(f"    {n:>10d} {sem(v):>10.4f} {min_detectable_effect(v):>22.4f}")
    print("\n    ⚠️ Compare these to report.py's old 0.005 verdict threshold.")
    print("       At n=300 an UNPAIRED difference under ~0.05 MRR is not measurable.")
    print("       Pairing is what makes the experiment affordable.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true",
                    help="validate the estimator on data with a known answer")
    ns = ap.parse_args()
    if ns.demo:
        _demo()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
