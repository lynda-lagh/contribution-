"""
★ THE CROSS-ENVIRONMENT CONTROL.

    python -m scripts.verify_env_control

Chapter 2 compares LoRA, MoRA and BOFT. MoRA exists only in the `kongds/MoRA`
fork (pinned to peft 0.9.0); BOFT is in official peft and did not exist in 0.9.0.
They share the import name `peft`, so installing one overwrites the other and the
three arms CANNOT be produced by a single environment.

That is a confound. A reviewer is entitled to ask: is MoRA's margin over LoRA a
property of MoRA, or of peft 0.9.0?

The answer is to run **LoRA in both environments** and compare. If the two LoRA
numbers agree within seed noise, the library version is not driving anything and
the three-way comparison stands. If they disagree, the comparison is invalid as
it stands and this script says so.

This is cheap -- one extra training run -- and it converts an unavoidable
technical obstacle into a reported control. Most papers comparing PEFT methods do
not even state which version they used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from src.utils.config import load_results


def collect(results_dir: str = "results") -> dict:
    """Group every ch2 LoRA run by the environment that produced it."""
    by_env: dict[str, list[dict]] = defaultdict(list)
    unstamped = []
    for r in load_results("ch2-*.json", results_dir):
        rid = r.get("run_id", r.get("_file", ""))
        if "-lora-" not in rid:
            continue
        env = (r.get("env") or {}).get("peft_env")
        acc = r.get("accuracy")
        if env is None:
            unstamped.append(rid)
            continue
        by_env[env].append({"run_id": rid, "accuracy": acc,
                            "peft": (r.get("env") or {}).get("peft"),
                            "entities": r.get("entities"),
                            "seed": r.get("seed")})
    return {"by_env": dict(by_env), "unstamped": unstamped}


def compare(by_env: dict, tolerance: float = 0.01) -> dict:
    envs = sorted(by_env)
    if len(envs) < 2:
        return {"status": "incomplete",
                "message": (f"LoRA runs found in {len(envs)} environment(s): {envs or 'none'}. "
                            f"The control needs LoRA from BOTH 'mora-fork' and 'official'.")}

    # match on the same (entities, seed) so we compare like with like
    keyed: dict[tuple, dict] = defaultdict(dict)
    for env, runs in by_env.items():
        for r in runs:
            if r["accuracy"] is not None:
                keyed[(r["entities"], r["seed"])][env] = r["accuracy"]

    pairs = [{"entities": k[0], "seed": k[1], **v,
              "delta": abs(v[envs[0]] - v[envs[1]])}
             for k, v in sorted(keyed.items())
             if len(v) >= 2]

    if not pairs:
        return {"status": "incomplete",
                "message": ("LoRA runs exist in both environments but never at the "
                            "SAME (entities, seed). Re-run one matching config.")}

    worst = max(p["delta"] for p in pairs)
    ok = worst <= tolerance
    return {
        "status": "pass" if ok else "FAIL",
        "environments": envs,
        "pairs": pairs,
        "max_delta": worst,
        "tolerance": tolerance,
        "message": (
            f"LoRA agrees across environments (max delta {worst:.4f} <= {tolerance}). "
            f"peft version is NOT a confound; the LoRA/MoRA/BOFT comparison stands."
            if ok else
            f"LoRA DISAGREES across environments (max delta {worst:.4f} > {tolerance}). "
            f"The three-way comparison is NOT valid as it stands: any MoRA-vs-BOFT "
            f"difference may be the library, not the method. Either report LoRA-vs-MoRA "
            f"and LoRA-vs-BOFT as two separate within-environment comparisons, or "
            f"raise the tolerance only if seed variance justifies it."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="max acceptable |accuracy difference|; set from your seed variance")
    ns = ap.parse_args()

    data = collect(ns.results_dir)
    rep = compare(data["by_env"], ns.tolerance)

    print("=" * 66)
    print("CROSS-ENVIRONMENT CONTROL -- is peft version a confound?")
    print("=" * 66)

    for env, runs in sorted(data["by_env"].items()):
        print(f"\n  {env}  ({len(runs)} LoRA run(s))")
        for r in runs:
            acc = "     -" if r["accuracy"] is None else f"{r['accuracy']:.4f}"
            print(f"     {r['run_id']:38s} acc={acc}  peft={r['peft']}")

    if data["unstamped"]:
        print(f"\n  ⚠️ {len(data['unstamped'])} run(s) have NO env stamp -- produced "
              f"before result stamping was added:")
        for rid in data["unstamped"][:5]:
            print(f"     {rid}")
        print("     These cannot take part in the control. Re-run or exclude them.")

    if rep.get("pairs"):
        print(f"\n  {'entities':>10s} {'seed':>6s} " +
              " ".join(f"{e:>12s}" for e in rep["environments"]) + f" {'delta':>9s}")
        for p in rep["pairs"]:
            print(f"  {p['entities']:>10,d} {p['seed']:>6d} " +
                  " ".join(f"{p[e]:12.4f}" for e in rep["environments"]) +
                  f" {p['delta']:+9.4f}")

    print(f"\n  STATUS: {rep['status'].upper()}")
    print(f"  {rep['message']}")
    print("\n  ★ Report this in the paper either way. An unavoidable technical")
    print("    obstacle that you measured is a strength, not a weakness.")


if __name__ == "__main__":
    main()
