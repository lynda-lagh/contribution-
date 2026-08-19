# ── the CODE the Kaggle notebook needs. Nothing else. ──────────────────────
git add \
  chapter1/analysis.py chapter1/conditions.py chapter1/data.py \
  chapter1/evaluate.py chapter1/rank.py chapter1/run.py \
  chapter1/seen_unseen.py chapter1/test_chapter1.py \
  chapter1/audit_data.py chapter1/compare.py chapter1/context.py \
  chapter1/preflight.py chapter1/rescore.py chapter1/showcase.py \
  src/routing/semantic_types.py src/eval/calibration.py src/infer/scoring.py \
  scripts/fetch_yago_types.py scripts/download_yago_types.py \
  notebooks/chapter1.ipynb CHANGES.md

git commit -m "chapter1: fix S permutation, prompt identity, type source, filtering

- rank.py: apply cond.shuffle (S was ranked on the REAL graph)
- rank.py: filter against train u valid u test, not train alone
- rank.py: save all 500 per-query ranks + top5, not 200
- run.py: --prompt is part of the run identity (was reading P0 data and
  OVERWRITING the P0 adapter)
- evaluate.py: prompt-aware test-set pairing; raise on a missing adapter
  instead of silently scoring the base model; memorisation share divided by
  the measured floor, not a hardcoded 0.5
- data.py: refuse an unlabelled test set; context blocks for P5/P6/P7;
  --require-semantic; --min-context coverage guard
- semantic_types.py: exogenous types (WordNet / YAGO / NELL), anon-invariant
- context.py: P5/P6/P7 with a four-part leak guard
- preflight.py, audit_data.py: 8 + 7 checks, non-zero exit, no fallbacks
- compare.py: our row in KG-LLM Table II, AUC / macro-F1 / McNemar
- showcase.py: real model outputs + a plain-English report"

git push origin main
