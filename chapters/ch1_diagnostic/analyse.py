"""
CHAPTER 1 ANALYSIS -- the four-parser decomposition.

No training here. We generate once per model, then read the SAME outputs four
different ways, and additionally score P("Yes") vs P("No") without generating
at all.

    python -m chapters.ch1_diagnostic.analyse --dataset WN11

Produces results/ch1_{dataset}.json:

    strict                     what the field reports
    lenient - strict           cost of formatting alone
    logit   - lenient          FORMAT CEILING: knew it, would not say it
    logit(anon)                how much of that "knowledge" is memorisation
    buggy_no_rate              how much free credit KG-LLM's lenient parser grants
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.prompts import ALPACA_NO_INPUT
from src.eval.parse import (ConstrainedParser, LenientParser, LogitParser,
                            StrictParser, decompose, response_breakdown, score)
from src.infer.scoring import constrained_choice, yes_no_probabilities


def load_model(base: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(adapter or base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                 # correct for next-token scoring
    model = AutoModelForCausalLM.from_pretrained(
        # fp32: this is inference, so the memory is affordable, and Chapter 1's
        # whole argument rests on these logits being real numbers. Qwen2.5 in
        # fp16 returns NaN.
        base, dtype=torch.float32, attn_implementation="sdpa").cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int = 16,
             batch_size: int = 8) -> list[str]:
    outs = []
    for i in range(0, len(prompts), batch_size):
        b = prompts[i:i + batch_size]
        enc = tok(b, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(b)):
            outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
    return outs


def evaluate(model, tok, prompts, labels, tag: str) -> dict:
    strict, lenient, logit_p, constr = StrictParser(), LenientParser(), LogitParser(), ConstrainedParser()

    print(f"  [{tag}] generating ...")
    texts = generate(model, tok, prompts)

    print(f"  [{tag}] logit scoring (no generation) ...")
    probs = yes_no_probabilities(model, tok, prompts)

    print(f"  [{tag}] constrained decoding ...")
    choices = constrained_choice(model, tok, prompts)

    res = {
        "strict":      score([strict(t).verdict for t in texts], labels),
        "lenient":     score([lenient(t).verdict for t in texts], labels),
        "logit":       score([logit_p(py, pn).verdict for py, pn in probs], labels),
        "constrained": score([constr(c).verdict for c in choices], labels),
    }

    # WHY each response got its verdict:
    #   refusal_scored_as_no  -> abstention converted into a negative prediction
    #                            (KG-LLM Table VI; Knit Fig. 1 calls it a hallucination)
    #   spurious_substring    -> parser artefact ("no" inside "know", "not" in "cannot")
    res["breakdown"] = response_breakdown(texts)
    res["buggy_no_rate"] = res["breakdown"]["rates"]["spurious_substring"]
    res["refusal_rate"] = res["breakdown"]["rates"]["refusal_scored_as_no"]
    res["not_a_real_answer_rate"] = res["breakdown"]["not_a_real_answer_rate"]

    # ★★ LOG THE DIRECTIONAL SCORES, not only the margin.
    #
    #    `confidences` is LogitParser.confidence = |p_yes - p_no| / (p_yes + p_no).
    #    That is a CERTAINTY MAGNITUDE: it says how sure the model was, not which
    #    way it answered. Downstream code (seen_unseen, calibration, ECE, Brier,
    #    risk-coverage) needs DIRECTION, and thresholding a magnitude at 0.5
    #    silently yields chance-level predictions that still look like a result.
    #
    #    p_yes and p_no are the raw quantities; everything else derives from them.
    res["p_yes"] = [float(py) for py, _ in probs]
    res["p_no"] = [float(pn) for _, pn in probs]
    res["confidences"] = [LogitParser.confidence(py, pn) for py, pn in probs]
    res["_raw"] = {"texts": texts[:50], "probs": probs[:50]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WN11", choices=["WN11", "FB13"])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--limit", type=int, default=2000,
                    help="fixed test subset, identical across conditions -> paired tests")
    ap.add_argument("--smi", action="store_true",
                    help="also compute sliced mutual information (FLAME's instrument); "
                         "slower, uses 600 items")
    ns = ap.parse_args()

    from src.utils.config import load_config
    cfg = load_config(ns.config)              # ★ seeds python/numpy/torch/HF
    base = cfg["model"]["name"]
    root = cfg["data"]["root"]

    # ★ chapter1.data writes to data/{dataset}-A/built/. The old
    #   data/{dataset}/built/ path never existed -- same bug as in
    #   chapter1/evaluate.py and chapter1/seen_unseen.py.
    _cands = [Path(root, f"{ns.dataset}-A", "built", "test_instructions.json"),
              Path(root, ns.dataset, "built", "test_instructions.json")]
    _test_path = next((p for p in _cands if p.exists()), None)
    if _test_path is None:
        raise SystemExit(
            "no built test set. Looked in:\n  "
            + "\n  ".join(str(p.parent) for p in _cands)
            + f"\n\n  build it:  python -m chapter1.data --condition A "
              f"--dataset {ns.dataset}")
    print(f"[ch1] test set <- {_test_path}")
    test = json.loads(_test_path.read_text(encoding="utf-8"))[: ns.limit]
    prompts = [ALPACA_NO_INPUT.format(instruction=r["instruction"]) for r in test]
    labels = [r["label"] for r in test]
    print(f"[ch1] {ns.dataset}: {len(prompts)} test items "
          f"(+1: {labels.count(1)}, -1: {labels.count(-1)}) | chance = 0.5")

    out: dict = {"dataset": ns.dataset, "n_test": len(prompts), "model": base}

    print("[ch1] UNTUNED")
    m, t = load_model(base, None)
    out["untuned"] = evaluate(m, t, prompts, labels, "untuned")
    del m; torch.cuda.empty_cache()

    # ★ TWO NAMING SCHEMES EXIST. This script was written against
    #     checkpoints/ch1-{dataset}-{method}          e.g. ch1-WN11-lora
    #   but chapter1/run.py (the current trainer) writes
    #     checkpoints/ch1-{dataset}-{condition}       e.g. ch1-WN11-A
    #   so a perfectly good adapter produced "skip tuned -- not found" and the
    #   result file silently lost its tuned arm. Accept both.
    _ad = Path(cfg["output"]["adapter_dir"])
    _method = cfg["peft"]["method"]

    def _find(*names):
        for n in names:
            p = _ad / n
            if p.exists():
                return p
        return _ad / names[0]          # for the error message

    tuned_dir = _find(f"ch1-{ns.dataset}-A", f"ch1-{ns.dataset}-{_method}")
    if tuned_dir.exists():
        print("[ch1] TUNED")
        m, t = load_model(base, str(tuned_dir))
        out["tuned"] = evaluate(m, t, prompts, labels, "tuned")
        del m; torch.cuda.empty_cache()
    else:
        print(f"[ch1] skip tuned -- {tuned_dir} not found")

    anon_dir = _find(f"ch1-{ns.dataset}-B", f"ch1-{ns.dataset}-anon-{_method}")
    if anon_dir.exists():
        # ★★ THE ANONYMISED ARM MUST BE EVALUATED ON ANONYMISED PROMPTS.
        #
        # This adapter was trained on `entity1234`-style identifiers. Scoring it
        # against real entity names measures DISTRIBUTION SHIFT (train/test
        # mismatch), not memorisation -- accuracy would collapse for the wrong
        # reason and `memorisation` in the decomposition would be inflated by
        # however much the mismatch costs.
        #
        # The contamination control (KG-CF) asks a single question: how much
        # accuracy survives when surface forms carry no pretraining signal?
        # Both training AND evaluation must therefore be anonymised, and the
        # only thing differing from the `tuned` arm is the surface forms.
        anon_path = Path(root, f"{ns.dataset}-anon", "built", "test_instructions.json")
        if not anon_path.exists():
            raise FileNotFoundError(
                f"{anon_path} not found. Build it first:\n"
                f"  python -m src.data.build_instructions --dataset {ns.dataset} "
                f"--n_triples {cfg['data']['train_triples']} --seed {cfg['seed']} --anonymise"
            )
        test_a = json.loads(anon_path.read_text(encoding="utf-8"))[: ns.limit]
        prompts_a = [ALPACA_NO_INPUT.format(instruction=r["instruction"]) for r in test_a]
        labels_a = [r["label"] for r in test_a]

        # Both files are built from kg.test in order, so the two arms must be
        # item-for-item paired. If they are not, the comparison is not paired and
        # McNemar does not apply.
        if labels_a != labels:
            raise RuntimeError(
                "Anonymised test labels differ from the plain ones -- the two "
                "arms are not item-for-item paired, so the memorisation estimate "
                "and the paired significance tests would both be invalid. "
                "Rebuild both with the SAME --seed."
            )

        print(f"[ch1] TUNED (anonymised)  [{len(prompts_a)} anonymised prompts]")
        m, t = load_model(base, str(anon_dir))
        out["tuned_anon"] = evaluate(m, t, prompts_a, labels_a, "anon")
        del m; torch.cuda.empty_cache()

    for cond in ("untuned", "tuned"):
        if cond in out:
            out[f"decomposition_{cond}"] = decompose(
                out[cond]["strict"], out[cond]["lenient"], out[cond]["logit"],
                out.get("tuned_anon", {}).get("logit") if cond == "tuned" else None)

    # ★ SECOND INSTRUMENT -- sliced mutual information (FLAME).
    # FLAME reports that frozen representations "reach fine-tuned-level SMI values,
    # indicating that fine-tuning primarily aligns representations rather than
    # injecting knowledge". If SMI barely moves while the logit gap is large, the
    # two instruments agree from opposite directions: tuning installed FORMAT.
    if ns.smi and "tuned" in out:
        from src.eval.smi import compare, smi_across_layers
        y = [1 if l == 1 else 0 for l in labels]
        sub = prompts[: min(600, len(prompts))]          # SMI needs samples, not all of them
        print("[ch1] SMI (untuned) ...")
        m, t = load_model(base, None)
        smi_u = smi_across_layers(m, t, sub, y[: len(sub)])
        del m; torch.cuda.empty_cache()
        print("[ch1] SMI (tuned) ...")
        m, t = load_model(base, str(tuned_dir))
        smi_t = smi_across_layers(m, t, sub, y[: len(sub)])
        del m; torch.cuda.empty_cache()
        out["smi"] = {"untuned": smi_u, "tuned": smi_t, "comparison": compare(smi_u, smi_t)}
        print(f"[ch1] SMI {smi_u['best_smi']:.5f} -> {smi_t['best_smi']:.5f} "
              f"| {out['smi']['comparison']['interpretation']}")

    res_dir = Path(cfg["output"]["results_dir"]); res_dir.mkdir(parents=True, exist_ok=True)
    slim = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                   if kk not in ("_raw", "confidences")})
            for k, v in out.items()}
    (res_dir / f"ch1_{ns.dataset}.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 62)
    print(f"CHAPTER 1 -- {ns.dataset}   (binary task, chance = 0.500)")
    print("=" * 62)
    for cond in ("untuned", "tuned", "tuned_anon"):
        if cond not in out:
            continue
        r = out[cond]
        print(f"\n{cond.upper()}")
        for p in ("strict", "lenient", "constrained", "logit"):
            print(f"  {p:12s} acc={r[p]['accuracy']:.4f}  "
                  f"unparseable={r[p]['unparseable_rate']:.1%}")
        b = r["breakdown"]["rates"]
        print("  response breakdown:")
        print(f"     answered yes/no        {b['answered_yes']+b['answered_no']:6.1%}")
        print(f"     REFUSAL scored as 'No' {b['refusal_scored_as_no']:6.1%}  "
              f"<- abstention penalised")
        print(f"     spurious substring     {b['spurious_substring']:6.1%}  "
              f"<- parser artefact")
        print(f"     unparseable            {b['unparseable']:6.1%}")
        print(f"     NOT A REAL ANSWER      {r['not_a_real_answer_rate']:6.1%}")
        d = out.get(f"decomposition_{cond}")
        if d:
            print(f"  -> format cost      {d['format_cost']:+.4f}")
            print(f"  -> FORMAT CEILING   {d['format_ceiling']:+.4f}")
            print(f"  -> logit vs chance  {d['logit_above_chance']:+.4f}")
            if "memorisation" in d:
                print(f"  -> memorisation     {d['memorisation']:+.4f}")
                print(f"  -> residual knowl.  {d['residual_knowledge']:+.4f}")
    print(f"\nsaved -> {res_dir / f'ch1_{ns.dataset}.json'}")


if __name__ == "__main__":
    main()
