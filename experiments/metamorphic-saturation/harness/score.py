"""Scoring rules for the response tests (§4.3–§4.4) and T4 (§5.3).

Response tests operate on per-rung majority verdicts (from reduce.py):
  - fired_at_all: some rung FIRED, else FAIL.
  - monotone_from_first_firing: no NOT-FIRED rung above the first FIRED rung.
  - sharpness: the first-firing rung's load multiple (§4.4).

T4 operates on a single overloaded trace's verdict series:
  - flip count = fired->not-fired transitions after first firing (§5.3).
  - lead time = t_div - t_first_fire, t_div = first time the backlog proxy
    exceeds the sub-capacity envelope and never returns below (§5.3).
  - dwell sweep: wrap the raw verdict in a hysteresis (fire on k consecutive,
    clear after w sustained not-fired); report flip count at matched lead time.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from parse import TraceRecord


# --- Response tests ----------------------------------------------------------

@dataclass
class ResponseResult:
    ladder: str
    fired_any: bool
    monotone: bool
    first_fire_index: int | None      # rung index of first firing
    first_fire_mult: float | None     # load multiple at first firing (sharpness)
    passed: bool
    rung_verdicts: list[bool]
    per_seed: list[list[bool]]        # [rung][seed]
    gray_zone_firings: list[int]      # rung indices in the gray band that fired


def score_response(ladder: str, rung_mults: list[float],
                   rung_fired: list[bool],
                   per_seed: list[list[bool]]) -> ResponseResult:
    fired_any = any(rung_fired)
    first_idx = next((i for i, f in enumerate(rung_fired) if f), None)

    monotone = True
    if first_idx is not None:
        # No NOT-FIRED rung strictly above the first firing (§4.3).
        monotone = all(rung_fired[i] for i in range(first_idx, len(rung_fired)))

    first_mult = rung_mults[first_idx] if first_idx is not None else None
    passed = fired_any and monotone

    gz_lo, gz_hi = config.GRAY_ZONE
    gray = [i for i, m in enumerate(rung_mults)
            if gz_lo <= m <= gz_hi and rung_fired[i]]

    return ResponseResult(
        ladder=ladder, fired_any=fired_any, monotone=monotone,
        first_fire_index=first_idx, first_fire_mult=first_mult,
        passed=passed, rung_verdicts=rung_fired, per_seed=per_seed,
        gray_zone_firings=gray,
    )


# --- T4 temporal -------------------------------------------------------------

def _fired_series(records: list[TraceRecord]) -> list[tuple[int, bool]]:
    """(timestamp, fired) piecewise-constant verdict series."""
    return [(r.timestamp, r.level in config.FIRED_LEVELS) for r in records]


def apply_dwell(series: list[tuple[int, bool]], k: int, w_us: int) -> list[tuple[int, bool]]:
    """Hysteresis wrapper (§5.3): fire after k consecutive fired samples; clear
    only after the raw signal is not-fired continuously for w_us.

    Returns the wrapped (timestamp, fired) series.
    """
    out = []
    state = False
    consec_fire = 0
    clear_start = None      # timestamp when the current not-fired stretch began
    for ts, f in series:
        if f:
            consec_fire += 1
            clear_start = None
            if not state and consec_fire >= k:
                state = True
        else:
            consec_fire = 0
            if state:
                if clear_start is None:
                    clear_start = ts
                elif ts - clear_start >= w_us:
                    state = False
        out.append((ts, state))
    return out


def flip_count(series: list[tuple[int, bool]]) -> tuple[int, int | None]:
    """Count fired->not-fired transitions AFTER the first firing (§5.3).

    Returns (flips, t_first_fire_us). t_first_fire None if never fired.
    """
    first_fire = None
    flips = 0
    prev = False
    for ts, f in series:
        if first_fire is None and f:
            first_fire = ts
        if first_fire is not None and prev and not f:
            flips += 1
        prev = f
    return flips, first_fire


def backlog_divergence_time(records: list[TraceRecord], envelope: float) -> int | None:
    """t_div: first time the in-flight backlog proxy exceeds `envelope` and never
    returns below it for the rest of the run (§5.3).

    backlog-drift's signals carry `in_flight` directly; for detectors without it
    we approximate from arrivals/completions if present, else return None.

    IMPORTANT: the backlog is a property of the RUN, shared by all detectors.
    Callers should compute t_div ONCE from the run's in-flight trajectory (see
    run_backlog_trajectory / divergence_from_trajectory) and pass the same value
    into every detector's curve, rather than reading each detector's own signals
    (only backlog-drift carries in_flight).
    """
    traj = [(r.timestamp, r.signals["in_flight"]) for r in records if "in_flight" in r.signals]
    if not traj:
        traj = [(r.timestamp, r.signals["arrivals"] - r.signals["completions"])
                for r in records if "arrivals" in r.signals and "completions" in r.signals]
    return divergence_from_trajectory(traj, envelope)


def divergence_from_trajectory(traj: list[tuple[int, float]], envelope: float) -> int | None:
    """t_div from a shared in-flight trajectory (§5.3): the first instant backlog
    exceeds `envelope` and never returns below it — EXCEPT we ignore the terminal
    drain tail (a finite run stops accepting arrivals, so in-flight always decays
    to 0 at the end; that decay is not a recovery under held load). We anchor the
    "never returns below" check to the trajectory's global-max timestamp, past
    which the tail is drain, not signal."""
    if not traj:
        return None
    peak_ts = max(traj, key=lambda p: p[1])[0]
    # Consider only up to the peak for the "sustained above" test.
    head = [(ts, v) for ts, v in traj if ts <= peak_ts]
    t_div = None
    for ts, v in head:
        if v > envelope:
            if t_div is None:
                t_div = ts
        else:
            t_div = None      # dipped back below before diverging; reset
    return t_div


@dataclass
class T4Point:
    w_us: int
    k: int
    flips: int
    lead_time_us: int | None       # t_div - t_first_fire; None if either missing


def t4_curve(records: list[TraceRecord], envelope: float,
             k: int = 2, w_grid_us: list[int] | None = None,
             t_div: int | None = None) -> list[T4Point]:
    """Sweep the dwell window w; return the (lead time, flip count) curve.

    t_div: the run's backlog-divergence timestamp, computed ONCE from the shared
    in-flight trajectory and passed in. If None, we fall back to this detector's
    own signals (only meaningful for backlog-drift)."""
    if w_grid_us is None:
        # 0 (no dwell) up to ~4 cycles.
        cyc = config.NOMINAL_CYCLE_S * 1e6
        w_grid_us = [0, int(0.25 * cyc), int(0.5 * cyc), int(cyc),
                     int(2 * cyc), int(4 * cyc)]
    raw = _fired_series(records)
    if t_div is None:
        t_div = backlog_divergence_time(records, envelope)
    curve = []
    for w in w_grid_us:
        wrapped = apply_dwell(raw, k, w) if w > 0 else raw
        flips, t_first = flip_count(wrapped)
        lead = (t_div - t_first) if (t_div is not None and t_first is not None) else None
        curve.append(T4Point(w_us=w, k=k, flips=flips, lead_time_us=lead))
    return curve


def zero_flip_lead(curve: list[T4Point]) -> int | None:
    """Largest lead time this detector achieves at flip count 0. None if it
    never reaches 0 flips (=> fails T4 outright)."""
    zeros = [p.lead_time_us for p in curve if p.flips == 0 and p.lead_time_us is not None]
    return max(zeros) if zeros else None
