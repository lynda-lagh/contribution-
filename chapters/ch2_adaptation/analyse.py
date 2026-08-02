"""
CHAPTER 2 ANALYSIS -- the decisive experiment.

    Does MoRA's margin over LoRA GROW with |E| ?

        grows -> the low-rank update is a bottleneck for entity memorisation
        flat  -> refuted, AND this is exactly what Chapter 1 predicts if tuning
                 installs FORMAT. MoRA's own paper says it is "comparable on other
                 tasks", so a null cannot be read as our method failing.

Also produces, from checkpoints that already exist:
    * the data-size axis        does the requirement scale with |E|?
    * catastrophic forgetting   what makes BOFT meaningful (asserted by BiGTex,
                                measured by nobody in 188 papers)
    * significance              3 seeds, paired tests, Bonferroni (P04's protocol)

No training. Run after the Chapter 2 grid.

    python -m chapters.ch2_adaptation.analyse
    python -m chapters.ch2_adaptation.analyse --forgetting
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.config import load_config, load_results, save_result


def _parse(rid: str) -> dict | None:
    """ch2-lora-E123182-T10000-s42 -> {peft, entities, triples, seed}"""
    p = rid.split("-")
    if len(p) < 5 or p[0] != "ch2":
        return None
    try:
        return {"peft": p[1],
                "entities": int(p[2][1:]), "triples": int(p[3][1:]), "seed": int(p[4][1:])}
    except (ValueError, IndexError):
        return None


def collect(results_dir: str = "results") -> list[dict]:
    rows = []
    for r in load_results("ch2-*.json", results_dir):
        meta = _parse(r.get("run_id", r["_file"].replace(".json", "")))
        if not meta:
            continue
        acc = r.get("accuracy")
        if acc is None:                       # pull from a Ch4 summary if present
            s = Path(results_dir, f"ch4_{r.get('run_id','')}", "summary.json")
            if s.exists():
                acc = json.loads(s.read_text()).get("metrics", {}).get("hits_at_1")
        rows.append(meta | {"accuracy": acc,
                            "runtime_s": r.get("train_runtime_s"),
                            "peak_vram_gb": r.get("peak_vram_gb"),
                            "run_id": r.get("run_id", r["_file"])})
    return [r for r in rows if r["accuracy"] is not None]


# ------------------------------------------------------------------ the sweep
def vocabulary_sweep(rows: list[dict], triples: int = 10_000) -> dict:
    """MoRA − LoRA margin as a function of |E|. THE experiment."""
    by = defaultdict(dict)
    for r in rows:
        if r["triples"] == triples and r["seed"] == 42:
            by[r["entities"]][r["peft"]] = r["accuracy"]

    pts = []
    for e in sorted(by):
        d = by[e]
        if "lora" in d and "mora" in d:
            pts.append({"entities": e, "lora": d["lora"], "mora": d["mora"],
                        "margin": d["mora"] - d["lora"],
                        "boft": d.get("boft"), "probe": d.get("probe")})

    out = {"points": pts, "n_points": len(pts)}
    if len(pts) >= 3:
        x = np.log10([p["entities"] for p in pts])
        y = np.array([p["margin"] for p in pts])
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out |= {
            "slope_per_decade": float(slope),
            "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else 0.0,
            "margin_at_smallest": pts[0]["margin"],
            "margin_at_largest": pts[-1]["margin"],
            "verdict": (
                "BOTTLENECK CONFIRMED -- MoRA's advantage grows with |E|"
                if slope > 0.01 else
                "BOTTLENECK REFUTED -- margin flat/negative across |E|. "
                "This is what Chapter 1 predicts if tuning installs FORMAT, and "
                "MoRA's own paper reports it is 'comparable on other tasks'."
            ),
        }
    else:
        out["verdict"] = "need >= 3 |E| points for a slope"
    return out


def frozen_probe_check(rows: list[dict]) -> dict:
    """
    ★ The control that makes a flat MoRA result INTERPRETABLE.
    If every adaptation method merely matches a linear probe on frozen
    representations, no method installed knowledge (FLAME's finding, reproduced).
    """
    probe = [r for r in rows if r["peft"] == "probe"]
    if not probe:
        return {"available": False,
                "note": "run `--peft probe` at the largest |E| -- one cheap run makes "
                        "the whole chapter interpretable"}
    best = max(probe, key=lambda r: r["accuracy"])
    tuned = {p: max((r["accuracy"] for r in rows if r["peft"] == p), default=None)
             for p in ("lora", "mora", "boft")}
    gaps = {p: (a - best["accuracy"]) for p, a in tuned.items() if a is not None}
    return {
        "available": True,
        "probe_accuracy": best["accuracy"],
        "tuned_accuracy": tuned,
        "gap_over_probe": gaps,
        "verdict": (
            "adaptation adds little over a frozen probe -> FLAME's finding reproduced: "
            "'fine-tuning primarily aligns representations rather than injecting knowledge'"
            if gaps and max(gaps.values()) < 0.02 else
            "adaptation clearly beats the frozen probe -> tuning does more than align"
        ),
    }


def data_axis(rows: list[dict]) -> dict:
    """
    Does the DATA requirement scale with |E|?
      flat   -> format (a fixed number of examples suffices however many entities)
      rising -> knowledge (more entities means more facts to memorise)

    The plain question is already answered (FLAME: 0.6% -> 97%; COSIGN: ~40%).
    Crossing it with |E| is what nobody has done.
    """
    by = defaultdict(dict)
    for r in rows:
        if r["seed"] == 42 and r["peft"] in ("lora", "mora"):
            by[(r["peft"], r["entities"])][r["triples"]] = r["accuracy"]

    out = {}
    for (peft, ents), d in sorted(by.items()):
        if len(d) < 2:
            continue
        ts = sorted(d)
        full = d[ts[-1]]
        need = next((t for t in ts if full and d[t] >= 0.95 * full), ts[-1])
        out[f"{peft}_E{ents}"] = {"curve": d, "triples_for_95pct": need,
                                  "fraction_of_max": need / ts[-1]}
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--triples", type=int, default=10_000)
    ap.add_argument("--forgetting", action="store_true",
                    help="measure catastrophic forgetting over the checkpoints")
    ns = ap.parse_args()

    cfg = load_config(ns.config)
    rows = collect(cfg["output"]["results_dir"])
    if not rows:
        raise SystemExit("No ch2-*.json results found. Run the Chapter 2 grid first.")
    print(f"[ch2] {len(rows)} runs collected")

    sweep = vocabulary_sweep(rows, ns.triples)
    probe = frozen_probe_check(rows)
    data = data_axis(rows)

    # significance across seeds, where the headline claim lives
    sig = {}
    seeds = defaultdict(list)
    for r in rows:
        if r["triples"] == ns.triples:
            seeds[(r["peft"], r["entities"])].append(r["accuracy"])
    from src.eval.significance import seed_variance
    for (peft, ents), accs in seeds.items():
        if len(accs) >= 2:
            sig[f"{peft}_E{ents}"] = seed_variance(accs)

    forget = None
    if ns.forgetting:
        from src.eval.forgetting import measure_forgetting
        adapters = {r["peft"]: str(Path(cfg["output"]["adapter_dir"], r["run_id"]))
                    for r in rows if r["peft"] in ("lora", "mora", "boft")
                    and r["entities"] == max(x["entities"] for x in rows)}
        forget = measure_forgetting(cfg["model"]["name"], adapters,
                                    out_path=str(Path(cfg["output"]["results_dir"],
                                                      "ch2_forgetting.json")))

    out = {"n_runs": len(rows), "vocabulary_sweep": sweep, "frozen_probe": probe,
           "data_axis": data, "seed_variance": sig, "forgetting": forget}
    save_result(cfg, "ch2_analysis", out)

    print("\n" + "=" * 68)
    print("CHAPTER 2 -- is low-rank a bottleneck for entity memorisation?")
    print("=" * 68)
    print(f"\n{'|E|':>10s} {'LoRA':>8s} {'MoRA':>8s} {'margin':>9s} {'BOFT':>8s} {'probe':>8s}")
    for p in sweep["points"]:
        f = lambda v: f"{v:8.4f}" if v is not None else "       -"
        print(f"{p['entities']:>10,d} {f(p['lora'])} {f(p['mora'])} "
              f"{p['margin']:+9.4f} {f(p.get('boft'))} {f(p.get('probe'))}")
    if "slope_per_decade" in sweep:
        print(f"\n  slope {sweep['slope_per_decade']:+.5f} per decade of |E| "
              f"(R² {sweep['r_squared']:.3f})")
        print(f"  {sweep['verdict']}")

    print(f"\n  frozen probe: {probe.get('verdict', probe.get('note'))}")

    if sig:
        print("\n  seed variance (a difference below 2*CI is not distinguishable "
              "from noise):")
        for k, v in sig.items():
            print(f"    {k:22s} mean {v['mean']:.4f} ± {v['ci95_halfwidth']:.4f} "
                  f"(n={v['n_seeds']})")

    if data:
        print("\n  data axis -- triples needed for 95% of full-budget accuracy:")
        for k, v in data.items():
            print(f"    {k:22s} {v['triples_for_95pct']:,} "
                  f"({v['fraction_of_max']:.0%} of max)")

    print(f"\nsaved -> {cfg['output']['results_dir']}/ch2_analysis.json")


if __name__ == "__main__":
    main()
