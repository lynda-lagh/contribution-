"""
★ Runs BEFORE anything else under `src.*` -- Python must import this package
first to reach any submodule (`src.train.sft`, `src.utils.config`, ...), so
this is the one place a fix here is guaranteed to land before the first
`import transformers` anywhere in the process.

Why this exists:

`transformers.set_seed()` (called by `src.utils.config.set_all_seeds`, so that
sampling/negatives/LoRA init are reproducible) also seeds every ML framework
IT CAN DETECT INSTALLED -- including TensorFlow -- not just the one you use.
Kaggle images ship a full TensorFlow install that this project never touches.

The first `import tensorflow` triggers TF's own CPU-feature self-check
(`tensorflow.python.platform.self_check.preload_check`), which is slow and, in
`chapters/*/run.py` launched as a subprocess pinned to one GPU (see
`DEPLOY.md`'s `pair()` helper -- TWO such subprocesses run at once, one per
T4), can stall long enough that stopping/interrupting the cell prints a giant
traceback that looks like a training crash. It never reaches your code: the
stack trace bottoms out inside `tensorflow/python/pywrap_tensorflow.py`.

`USE_TF=0` (read by `transformers.utils.import_utils` at import time) makes
transformers skip TensorFlow/Flax detection entirely -- it must be set before
`transformers` is first imported, hence here rather than inside
`set_all_seeds`, which runs too late (after `src.train.sft` has already
imported `transformers` at module level).
"""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
