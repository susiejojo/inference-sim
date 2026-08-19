"""Workload-spec (v2) generator for the metamorphic suite.

Emits a strict-parseable BLIS v2 workload YAML per (ladder, rung, seed). Every
spec shares ONE burstiness shape (gamma, fixed CV) so only the named axis varies
across a ladder's rungs (§3.2). Token sizes are `constant` distributions so the
size ladders isolate their axis exactly on the simulator (§3.2 hardware caveat
does not apply here).

The inline `blis run --rate` path forces `constant` arrivals, so bursty runs
MUST go through `--workload-spec`; that is the whole reason this module exists.
"""

from __future__ import annotations

import os

import yaml

import config
from ladder import Rung


def _arrival_block() -> dict:
    """The arrival spec for the current arm. Poisson has an intrinsic CV of 1
    (no tunable knob), so we omit `cv` there; gamma carries the explicit CV."""
    if config.ARRIVAL_PROCESS == "poisson":
        return {"process": "poisson"}
    return {"process": config.ARRIVAL_PROCESS, "cv": config.ARRIVAL_CV}


def _client_spec(rung: Rung) -> dict:
    """One rate-based client carrying all traffic; arrivals per the current arm."""
    return {
        "id": "metamorphic",
        "slo_class": "standard",     # one priority class (S2)
        "rate_fraction": 1.0,
        "streaming": True,
        "arrival": _arrival_block(),
        "input_distribution": {
            "type": "constant",
            "params": {"value": float(rung.input_tokens)},
        },
        "output_distribution": {
            "type": "constant",
            "params": {"value": float(rung.output_tokens)},
        },
    }


def build_spec(rung: Rung, seed: int) -> dict:
    """A complete v2 WorkloadSpec dict for this rung+seed.

    aggregate_rate carries the arrival rate; for L-CAP the rate is identical
    across rungs (capacity is the CLI axis, not the workload), so the produced
    YAML is byte-identical across L-CAP rungs at a fixed seed — exactly T3's
    "workload byte-identical across rungs" requirement (§T3).
    """
    n_requests = config.rung_run_requests(rung.rate_rps)
    return {
        "version": "2",
        "seed": int(seed),
        "category": "language",
        "aggregate_rate": float(rung.rate_rps),
        "num_requests": int(n_requests),
        "clients": [_client_spec(rung)],
    }


def spec_path(rung: Rung, seed: int) -> str:
    fname = f"{rung.ladder}_r{rung.index:02d}_seed{seed}.yaml"
    return os.path.join(config.SPECS_DIR, fname)


def write_spec(rung: Rung, seed: int) -> str:
    os.makedirs(config.SPECS_DIR, exist_ok=True)
    path = spec_path(rung, seed)
    with open(path, "w") as f:
        yaml.safe_dump(build_spec(rung, seed), f, sort_keys=False)
    return path


# --- T4 temporal traces ------------------------------------------------------
# Held-mean overloaded traces (§5.2): a fixed super-capacity mean with
# pronounced bursts so a level detector is whipsawed. We use an explicit
# lifecycle square-wave (burst window / lull window) to make the burst-lull
# cycle exact, which the flip-count and lead-time scoring key on (§5.3).

def build_t4_spec(mean_mult: float, seed: int,
                  burst_amp: float = 3.0, lull_frac: float = 0.5,
                  cycle_s: float = None, n_cycles: int = 24) -> dict:
    """A held-mean overloaded trace with a square-wave burst/lull schedule.

    mean_mult: overloaded mean as a multiple of mu* (e.g. 1.3).
    burst_amp: peak rate multiple applied during the burst half of a cycle.
    lull_frac: fraction of the cycle spent in the lull (low rate).
    The lull rate is solved so the time-average equals mean_mult * mu*.
    """
    cycle_s = cycle_s or config.NOMINAL_CYCLE_S
    mean_rate = mean_mult * config.MU_STAR
    burst_rate = burst_amp * config.MU_STAR
    # mean = burst_rate*(1-lull_frac) + lull_rate*lull_frac  =>  solve lull_rate.
    lull_rate = (mean_rate - burst_rate * (1 - lull_frac)) / lull_frac
    lull_rate = max(0.0, lull_rate)

    cycle_us = int(cycle_s * 1e6)
    burst_us = int(cycle_us * (1 - lull_frac))
    windows = []
    for k in range(n_cycles):
        start = k * cycle_us
        windows.append({          # burst half
            "start_us": start,
            "end_us": start + burst_us,
            "trace_rate": round(burst_rate, 4),
        })
        windows.append({          # lull half
            "start_us": start + burst_us,
            "end_us": start + cycle_us,
            "trace_rate": round(lull_rate, 4),
        })

    horizon_us = n_cycles * cycle_us
    client = {
        "id": "t4",
        "slo_class": "standard",
        "rate_fraction": 1.0,
        "streaming": True,
        "arrival": _arrival_block(),
        "input_distribution": {"type": "constant", "params": {"value": float(config.E_IN)}},
        "output_distribution": {"type": "constant", "params": {"value": float(config.E_OUT)}},
        "lifecycle": {"windows": windows},
    }
    return {
        "version": "2",
        "seed": int(seed),
        "category": "language",
        # Absolute-rate mode: per-window trace_rate drives arrivals.
        "aggregate_rate": 0.0,
        "horizon": int(horizon_us),
        "clients": [client],
    }


def t4_spec_path(mean_mult: float, seed: int) -> str:
    tag = str(mean_mult).replace(".", "p")
    return os.path.join(config.SPECS_DIR, f"T4_mean{tag}_seed{seed}.yaml")


def write_t4_spec(mean_mult: float, seed: int) -> str:
    os.makedirs(config.SPECS_DIR, exist_ok=True)
    path = t4_spec_path(mean_mult, seed)
    with open(path, "w") as f:
        yaml.safe_dump(build_t4_spec(mean_mult, seed), f, sort_keys=False)
    return path


if __name__ == "__main__":
    from ladder import build_ladder
    r = build_ladder("L-RATE")[8]
    print(yaml.safe_dump(build_spec(r, 42), sort_keys=False))
    print("--- T4 ---")
    print(yaml.safe_dump(build_t4_spec(1.3, 42), sort_keys=False)[:600])
