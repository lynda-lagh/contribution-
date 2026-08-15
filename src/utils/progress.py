"""
Progress you can actually follow — bars, ETAs and phase banners.

A cell that prints nothing for forty minutes is indistinguishable from a hung
one, and on a 12-hour Kaggle budget "is it working?" is a real question.

    from src.utils.progress import track, phase, done

    with phase("build", "10,000 triples -> instances"):
        for t in track(triples, "rendering"):
            ...

Falls back to plain prints if tqdm is unavailable, so nothing here can break a
run. Works in both notebooks and terminals.
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager

try:
    from tqdm.auto import tqdm as _tqdm       # notebook-aware
    HAVE_TQDM = True
except Exception:                              # pragma: no cover
    HAVE_TQDM = False


def track(iterable, desc: str = "", total: int | None = None, unit: str = "it"):
    """
    A progress bar with a live ETA.

    tqdm's postfix shows rate and remaining time, which is the number you
    actually want mid-run: "how long until I can close the laptop?"
    """
    if HAVE_TQDM:
        return _tqdm(iterable, desc=desc, total=total, unit=unit,
                     dynamic_ncols=True, leave=True,
                     bar_format="  {desc:<26} {percentage:3.0f}%|{bar:28}| "
                                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    return _fallback(iterable, desc, total)


def _fallback(iterable, desc, total, min_interval: float = 3.0):
    """
    No tqdm: print at most every `min_interval` seconds.

    ⚠️ Throttle by TIME, not by count. A percentage-based fallback prints 20
    lines for a loop that finishes in 300 ms, which buries the output that
    matters. Fast loops should stay silent; slow ones should report.
    """
    try:
        total = total if total is not None else len(iterable)
    except TypeError:
        total = None
    t0 = last = time.time()
    i = 0
    for i, x in enumerate(iterable, 1):
        now = time.time()
        if now - last >= min_interval:
            el = now - t0
            if total:
                eta = el / i * (total - i)
                print(f"  {desc:<26} {i:,}/{total:,} ({i/total:5.1%}) "
                      f"elapsed {_hms(el)} · eta {_hms(eta)}", flush=True)
            else:
                print(f"  {desc:<26} {i:,} · elapsed {_hms(el)}", flush=True)
            last = now
        yield x
    el = time.time() - t0
    if el >= min_interval:                      # only summarise if it was slow
        print(f"  {desc:<26} {i:,} done in {_hms(el)}", flush=True)


def _hms(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m"


@contextmanager
def phase(name: str, detail: str = ""):
    """
    A banner around a stage, with the wall clock at the end.

    Printing the duration is not decoration: it is what makes the next session's
    schedule predictable, and GPU-hours are a reported metric in this thesis.
    """
    print(f"\n{'─' * 72}")
    print(f"▶ {name}" + (f"   {detail}" if detail else ""))
    print("─" * 72, flush=True)
    t0 = time.time()
    try:
        yield
    finally:
        print(f"✓ {name} finished in {_hms(time.time() - t0)}", flush=True)


def done(msg: str) -> None:
    print(f"  ✓ {msg}", flush=True)


def step(i: int, n: int, msg: str) -> None:
    print(f"\n[{i}/{n}] {msg}", flush=True)


def eta_note(n_items: int, per_item_s: float, label: str) -> None:
    """Say up front how long something should take, before it starts."""
    print(f"  {label}: {n_items:,} items × ~{per_item_s:.3f}s "
          f"≈ {_hms(n_items * per_item_s)}", flush=True)
