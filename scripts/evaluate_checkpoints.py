"""
★ BACKFILL EVALUATION — score adapters you have ALREADY trained.

    python -m scripts.evaluate_checkpoints                 # everything found
    python -m scripts.evaluate_checkpoints --only ch3      # just the ladder
    python -m scripts.evaluate_checkpoints --dry-run       # check data, train nothing

Chapters 2 and 3 wrote training summaries with NO accuracy field, so
`ch2_adaptation.analyse` has nothing to compare and the conditioning ladder has
no outcome. Every adapter is on disk, so this is pure inference -- no retraining.

Writes `accuracy` and the four-parser breakdown back into results/<run_id>.json.

⚠️ READ THE LABEL GUARD BELOW BEFORE TRUSTING ANY NUMBER IT PRINTS.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.prompts import ALPACA_NO_INPUT
from src.eval.parse import (ConstrainedParser, LenientParser, LogitParser,
                            StrictParser, response_breakdown, score)
from src.infer.scoring import constrained_choice, yes_no_probabilities
from src.utils.config import load_config, peft_env


# --------------------------------------------------------------- data mapping
def data_dir_for(run_id: str, root: str) -> Path:
    """
    Which built dataset did this run train on?

        ch1-WN11-lora              -> data/WN11/built
        ch1-WN11-anon-lora         -> data/WN11-anon/built
        ch2-lora-E123182-T10000-s42-> data/YAGO3-10-E123182/built
        ch3-YAGO3-10-L2-lora       -> data/YAGO3-10-L2/built
    """
    p = run_id.split("-")
    if run_id.startswith("ch1-"):
        return Path(root, "WN11-anon" if "anon" in run_id else "WN11", "built")
    if run_id.startswith("ch2-"):
        ent = next((x[1:] for x in p if x.startswith("E") and x[1:].isdigit()), None)
        return Path(root, f"YAGO3-10-E{ent}", "built")
    if run_id.startswith("ch3-"):
        lvl = next((x for x in p if x.startswith("L") and x[1:].isdigit()), None)
        return Path(root, f"YAGO3-10-{lvl}", "built")
    raise ValueError(f"cannot infer dataset for {run_id}")


def rebuild_command(run_id: str, cfg: dict) -> str:
    """
    The command that regenerates this run's built data.

    ⚠️ Only Chapter 1 can be rebuilt with plain `build_instructions`. Chapter 2's
    data depends on the |E| induced subgraph and Chapter 3's on the routing level,
    both of which are produced inside their own runners -- and those RETRAIN.
    Say so rather than printing a command that will not work.
    """
    n, seed = cfg["data"]["train_triples"], cfg["seed"]
    if run_id.startswith("ch1-"):
        anon = " --anonymise" if "anon" in run_id else ""
        return (f"python -m src.data.build_instructions --dataset WN11 "
                f"--n_triples {n} --seed {seed}{anon}")
    if run_id.startswith("ch2-"):
        ent = next((x[1:] for x in run_id.split("-")
                    if x.startswith("E") and x[1:].isdigit()), "?")
        return (f"⚠️ produced inside ch2_adaptation.run (needs the |E|={ent} induced "
                f"subgraph).\n            Regenerating it currently RETRAINS. "
                f"Fetch YAGO3-10 first:\n            "
                f"python -m scripts.fetch_data --datasets YAGO3-10")
    if run_id.startswith("ch3-"):
        lvl = next((x for x in run_id.split("-")
                    if x.startswith("L") and x[1:].isdigit()), "?")
        return (f"⚠️ produced inside ch3_conditioning.run (needs the {lvl} routing "
                f"plan).\n            Regenerating it currently RETRAINS. "
                f"Fetch YAGO3-10 first:\n            "
                f"python -m scripts.fetch_data --datasets YAGO3-10")
    return "unknown run type"


def check_labels(records: list[dict], run_id: str) -> dict:
    """
    ★★ THE GUARD THAT MATTERS MOST.

    `build_instructions` writes the target as:

        YES if t.label == 1 else NO

    and `load_kg` sets `label = None` when test.tsv has no 4th column. WN11 and
    FB13 carry ±1 labels. **YAGO3-10 in the KG-LLM repo does not** -- it was used
    there for RELATION prediction, not triple classification.

    So on YAGO3-10 every test row gets `label=None`, `None == 1` is False, and
    the gold answer becomes "No, this is not true." for ALL of them.

    A model can then score 100% by always answering No, and 0% by always
    answering Yes. Either way the number is meaningless. Refuse to report it.
    """
    labels = [r.get("label") for r in records]
    dist = Counter(labels)
    n = len(labels)
    pos = dist.get(1, 0)
    neg = dist.get(-1, 0)
    none = dist.get(None, 0)

    ok = pos > 0 and neg > 0
    return {
        "n": n, "positive": pos, "negative": neg, "unlabelled": none,
        "positive_rate": pos / n if n else 0.0,
        "usable": ok,
        "verdict": (
            f"ok — {pos} positive / {neg} negative"
            if ok else
            f"⚠️ DEGENERATE: {pos} pos / {neg} neg / {none} unlabelled. "
            f"Triple classification needs BOTH classes in the test set. "
            f"If this is YAGO3-10, its test.tsv has no ±1 label column, so every "
            f"gold answer became 'No'. Accuracy here measures nothing. "
            f"Fix: evaluate {run_id} on WN11/FB13, or generate negatives for the "
            f"YAGO3-10 test split the same way training negatives are generated."
        ),
    }


# --------------------------------------------------------------- model
def load(base: str, adapter: str):
    tok = AutoTokenizer.from_pretrained(adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float32, attn_implementation="sdpa").cuda()
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, adapter)
    return m.eval(), tok


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=16, batch_size=8):
    prev, tok.padding_side = tok.padding_side, "left"
    outs = []
    try:
        for i in range(0, len(prompts), batch_size):
            b = prompts[i:i + batch_size]
            enc = tok(b, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to("cuda")
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            for j in range(len(b)):
                outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True))
    finally:
        tok.padding_side = prev
    return outs


def evaluate_one(base: str, adapter_dir: Path, records: list[dict],
                 limit: int) -> dict:
    recs = records[:limit]
    prompts = [ALPACA_NO_INPUT.format(instruction=r["instruction"]) for r in recs]
    labels = [r["label"] for r in recs]

    model, tok = load(base, str(adapter_dir))
    texts = generate(model, tok, prompts)
    probs = yes_no_probabilities(model, tok, prompts)
    choices = constrained_choice(model, tok, prompts)
    del model
    torch.cuda.empty_cache()

    strict, lenient, logit_p, constr = (StrictParser(), LenientParser(),
                                        LogitParser(), ConstrainedParser())
    out = {
        "strict":      score([strict(t).verdict for t in texts], labels),
        "lenient":     score([lenient(t).verdict for t in texts], labels),
        "logit":       score([logit_p(py, pn).verdict for py, pn in probs], labels),
        "constrained": score([constr(c).verdict for c in choices], labels),
        "breakdown":   response_breakdown(texts),
        "n_evaluated": len(recs),
    }
    # the headline number the analysis scripts look for
    out["accuracy"] = out["logit"]["accuracy"]
    out["accuracy_source"] = "logit (format-independent)"

    # ★ A model that always says the same thing scores well on a one-class test
    # set. Report the prediction distribution so that is visible, not hidden.
    verdicts = [logit_p(py, pn).verdict for py, pn in probs]
    d = Counter(verdicts)
    out["prediction_distribution"] = dict(d)
    out["always_same_answer"] = len(d) == 1
    return out


# --------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--only", default="", help="substring filter, e.g. ch3")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate data and labels only; load no model")
    ns = ap.parse_args()

    cfg = load_config(ns.config)
    base = cfg["model"]["name"]
    root = cfg["data"]["root"]
    res_dir = Path(cfg["output"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    adapters = sorted(p.parent for p in Path(ns.checkpoints).glob("*/adapter_model.safetensors")
                      if ns.only in p.parent.name)
    if not adapters:
        raise SystemExit(f"no adapters under {ns.checkpoints}/ matching '{ns.only}'")

    print("=" * 72)
    print(f"BACKFILL EVALUATION — {len(adapters)} adapter(s)")
    print(f"environment: {peft_env()['peft_env']}")
    print("=" * 72)

    # ---- phase 1: validate every dataset BEFORE loading a 3 GB model ----
    plan, missing, degenerate = [], [], []
    print("\nPHASE 1 — data and label check\n")
    for a in adapters:
        rid = a.name
        try:
            d = data_dir_for(rid, root)
        except ValueError as e:
            print(f"  [skip] {rid}: {e}")
            continue
        f = d / "test_instructions.json"
        if not f.exists():
            missing.append(rid)
            print(f"  [no data] {rid}\n"
                  f"            {f} does not exist. Rebuild with:\n"
                  f"            {rebuild_command(rid, cfg)}")
            continue
        recs = json.loads(f.read_text(encoding="utf-8"))
        chk = check_labels(recs, rid)
        flag = "ok  " if chk["usable"] else "STOP"
        print(f"  [{flag}] {rid:34s} {chk['verdict'][:90]}")
        if chk["usable"]:
            plan.append((a, recs, chk))
        else:
            degenerate.append(rid)

    print(f"\n  {len(plan)} evaluable · {len(missing)} no data · "
          f"{len(degenerate)} degenerate labels  (of {len(adapters)} adapters)")

    if not plan:
        # Distinguish the two very different reasons, because the fixes differ.
        if degenerate and not missing:
            raise SystemExit(
                "\nNothing evaluable — the test sets exist but have only ONE class.\n"
                "That is the real problem: see the verdicts above.")
        raise SystemExit(
            "\nNothing evaluable — the built data is simply absent (not a label bug).\n"
            "Run the rebuild command shown under each adapter, then try again.\n"
            "Start with Chapter 1: it needs only WN11 and rebuilds in ~1 minute.")

    if ns.dry_run:
        print("\n--dry-run: stopping before model load.")
        return

    # ---- phase 2: evaluate ----
    print("\nPHASE 2 — evaluation\n")
    summary = []
    for a, recs, chk in plan:
        rid = a.name
        print(f"  [{rid}] evaluating {min(ns.limit, len(recs))} items …")
        r = evaluate_one(base, a, recs, ns.limit)

        # merge into the existing result JSON rather than overwrite it
        out_f = res_dir / f"{rid}.json"
        existing = {}
        if out_f.exists():
            try:
                existing = json.loads(out_f.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing |= {"label_check": chk, **r}
        out_f.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")

        warn = "  ⚠️ ALWAYS SAME ANSWER" if r["always_same_answer"] else ""
        print(f"      logit {r['logit']['accuracy']:.4f} | strict "
              f"{r['strict']['accuracy']:.4f} | constrained "
              f"{r['constrained']['accuracy']:.4f}{warn}")
        summary.append((rid, r))

    # ---- phase 3: the table you actually want ----
    print("\n" + "=" * 72)
    print(f"{'run':36s} {'logit':>8s} {'strict':>8s} {'constr':>8s} {'pos rate':>9s}")
    print("=" * 72)
    for rid, r in summary:
        pr = r["prediction_distribution"].get("yes", 0) / max(1, r["n_evaluated"])
        print(f"{rid:36s} {r['logit']['accuracy']:8.4f} "
              f"{r['strict']['accuracy']:8.4f} {r['constrained']['accuracy']:8.4f} "
              f"{pr:9.3f}")
    print("=" * 72)
    print("\n★ 'pos rate' is the fraction predicted Yes. If it is ~0 or ~1 the model")
    print("  is answering constantly and the accuracy is an artefact of the class")
    print("  balance, not a measurement of anything.")
    print(f"\nwritten to {res_dir}/*.json — now run:")
    print("  python -m chapters.ch2_adaptation.analyse")


if __name__ == "__main__":
    main()
