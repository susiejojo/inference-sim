"""Rung decision rules (§3.5): time series -> one verdict per rung.

Two reductions:
  1. Run -> per-seed verdict: discard a warm-up (one burst-lull cycle), then
     FIRED iff the detector is in a fired level for >= FIRED_FRACTION of the
     remaining run, measured as a TIME fraction (§3.5), not a sample count.
  2. Seeds -> rung verdict: majority vote (>= SEED_MAJORITY of the seeds FIRED).

"Fired level" maps the 3-level bank output to yes/no via config.FIRED_LEVELS
(OVERLOADED, BACKLOGGED). Declared on every scorecard.
"""

from __future__ import annotations

import config
from parse import TraceRecord


def _is_fired(level: str) -> bool:
    return level in config.FIRED_LEVELS


def fired_time_fraction(records: list[TraceRecord], warmup_us: int) -> float:
    """Fraction of the post-warm-up run TIME spent in a fired level.

    Each record's verdict is held until the next record (piecewise-constant),
    so we weight each verdict by the interval to the following event. The last
    record is weighted by the mean inter-event gap (it has no successor).
    """
    if not records:
        return 0.0
    recs = [r for r in records if r.timestamp >= records[0].timestamp + warmup_us]
    if len(recs) < 2:
        # Not enough post-warm-up signal to time-weight; fall back to the last
        # verdict's firedness as a degenerate 0/1.
        return 1.0 if recs and _is_fired(recs[-1].level) else 0.0

    total = recs[-1].timestamp - recs[0].timestamp
    if total <= 0:
        return 1.0 if _is_fired(recs[-1].level) else 0.0
    mean_gap = total / (len(recs) - 1)

    fired = 0.0
    for i, r in enumerate(recs):
        if i < len(recs) - 1:
            dt = recs[i + 1].timestamp - r.timestamp
        else:
            dt = mean_gap
        if _is_fired(r.level):
            fired += dt
    return fired / (total + mean_gap)


def run_verdict(records: list[TraceRecord]) -> bool:
    """Per-seed FIRED verdict (§3.5)."""
    warmup_us = int(config.WARMUP_CYCLES * config.NOMINAL_CYCLE_S * 1e6)
    return fired_time_fraction(records, warmup_us) >= config.FIRED_FRACTION


def rung_verdict(seed_records: list[list[TraceRecord]]) -> tuple[bool, list[bool]]:
    """Seeds -> rung verdict via majority vote. Returns (fired, per_seed)."""
    per_seed = [run_verdict(recs) for recs in seed_records]
    fired = sum(per_seed) >= config.SEED_MAJORITY
    return fired, per_seed
