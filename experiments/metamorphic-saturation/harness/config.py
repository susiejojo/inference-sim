"""Frozen apparatus constants for the metamorphic saturation-detector suite.

Everything the suite treats as "chosen once and held fixed" lives here so a
single edit re-parameterizes the whole run. Values trace directly to
metamorphic_tests.md (§3.2–§3.5) and to the on-machine cliff probe at vLLM
defaults (qwen/qwen3-14b, trained-physics, B=256 => mu* ~= 28.7 req/s).

NOTE on mu*: it is a *scaffold for rung spacing only* (§3.3), never a pass/fail
input. Being ~20% off just widens the ladder.
"""

from __future__ import annotations

import os

# --- Paths -------------------------------------------------------------------
HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HARNESS_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
BLIS_BIN = os.path.join(REPO_ROOT, "blis")

# --- Arrival arm (experiment dimension) --------------------------------------
# The suite runs one arrival "arm" at a time, selected by the ARRIVAL_ARM env
# var. Each arm gets its OWN specs/runs/configs/results subdir so the arms never
# collide and can be compared side by side. Two arms are defined:
#   gamma   (CV=3.0) — bursty, the default; the instrument T4 needs (§3.1/§5.2)
#   poisson (CV=1.0) — the burstiness CONTROL; isolates whether a failure is
#                      burst-driven or intrinsic (composite FPR, backlog-drift T2)
# Usage:  ARRIVAL_ARM=poisson python3 run_suite.py all
_ARMS = {
    "gamma":   {"process": "gamma",   "cv": 3.0},
    "poisson": {"process": "poisson", "cv": 1.0},
}
ARRIVAL_ARM = os.environ.get("ARRIVAL_ARM", "gamma").strip().lower()
if ARRIVAL_ARM not in _ARMS:
    raise ValueError(f"ARRIVAL_ARM={ARRIVAL_ARM!r} invalid; choose one of {sorted(_ARMS)}")

# Per-arm subdirectories keep the two experiments fully separate.
SPECS_DIR = os.path.join(EXP_DIR, "specs", ARRIVAL_ARM)
RUNS_DIR = os.path.join(EXP_DIR, "runs", ARRIVAL_ARM)
CONFIGS_DIR = os.path.join(EXP_DIR, "configs", ARRIVAL_ARM)
RESULTS_DIR = os.path.join(EXP_DIR, "results", ARRIVAL_ARM)

# --- Model / engine (frozen apparatus, §3.2) --------------------------------
MODEL = "qwen/qwen3-14b"
LATENCY_MODEL = "trained-physics"
# S1 (non-shedding): disable the request deadline so overloaded runs do not drop
# requests, which would silently reintroduce shedding. -1 == disabled.
TIMEOUT = -1

# Healthy capacity anchor: vLLM DEFAULTS (--max-num-seqs 256, --max-num-batched-
# tokens 2048, --gpu-memory-utilization 0.9). B (max-num-seqs) is the T3 knob;
# T3 shrinks B downward from this anchor (throughput scales cleanly with B at
# these defaults: B=256->7176, 128->5456, 64->3644 tok/s at a fixed overload).
HEALTHY_B = 256                  # vLLM default --max-num-seqs
MAX_NUM_BATCHED_TOKENS = 2048    # vLLM default --max-num-batched-tokens
GPU_MEMORY_UTILIZATION = 0.9     # vLLM default
# Default KV blocks left implicit (large); capacity is controlled via B for T3.

# --- Workload mix (constant token sizes => exact axis isolation, §3.2) -------
E_IN = 512   # mean input (prompt) tokens per request
E_OUT = 256  # mean output tokens per request

# --- Nominal cliff (scaffold only, §3.3) -------------------------------------
# mu* = r_dec / E[O]; probed r_dec ~= 7350 tok/s at vLLM defaults (B=256) =>
# ~28.7 req/s. Throughput plateaus for rate>=45; the knee is ~rate 33-40.
MU_STAR = 28.7  # req/s, the healthy-capacity cliff at vLLM defaults

# --- Burstiness shape: chosen ONCE, held fixed across ALL rungs of ALL ladders
# (§3.2). Gamma arrivals with CV>1 give statistical bursts. The suite requires a
# burst-lull cycle to size the decision window and warm-up; with gamma arrivals
# the "cycle" is a statistical timescale rather than a square wave, so we define
# a nominal cycle length for the run-length / warm-up accounting (§3.2, §3.5).
ARRIVAL_PROCESS = _ARMS[ARRIVAL_ARM]["process"]   # gamma | poisson (per arm)
ARRIVAL_CV = _ARMS[ARRIVAL_ARM]["cv"]             # 3.0 (gamma) | 1.0 (poisson)
NOMINAL_CYCLE_S = 2.0            # nominal burst-lull timescale (s) for accounting

# T4 backlog-divergence envelope as a multiple of HEALTHY_B (§3.1/§5.3): the
# in-flight level a sub-capacity burst stays under but a super-capacity mean
# sustains above. Probed: 1.3x-overload in-flight peaks ~540 at B=256; ~1.0xB
# cleanly separates transient from sustained.
T4_ENVELOPE_MULT = 1.0

# --- Ladder layout (§3.2 defaults) -------------------------------------------
RUNG_COUNT = 12
LADDER_LOW = 0.3                 # x mu*
LADDER_HIGH = 1.6                # x mu*
DENSE_CENTER = 1.0               # denser spacing near here

# Run length per rung: larger of 2000 requests or 20 burst-lull cycles (§3.2).
MIN_REQUESTS = 2000
MIN_CYCLES = 20

# --- Seeds (§3.2 / §3.5) -----------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]     # 5 seeds; stochastic (gamma) arrivals
SEED_MAJORITY = 3                # rung FIRED iff >=3/5 seed-runs FIRED

# --- Rung decision rule (§3.5) -----------------------------------------------
WARMUP_CYCLES = 1                # discard one burst-lull cycle
FIRED_FRACTION = 0.50            # FIRED iff fired-level for >=50% of remainder

# Fired-level mapping: the bank emits a 3-level verdict; the suite is yes/no.
# BACKLOGGED means backlog is growing => counts as "saturated". Declared on
# every scorecard so it is not a hidden judgment call.
FIRED_LEVELS = {"OVERLOADED", "BACKLOGGED"}
NOTFIRED_LEVELS = {"STABLE"}

# --- FPR operating point (§3.4) ----------------------------------------------
CALIB_RUNGS = [0.3, 0.4, 0.5, 0.6]   # x mu*, sub-capacity calibration band
TARGET_FPR = 0.05                    # <=5% => <=1 firing on the 20-run set
GRAY_ZONE = (0.7, 0.95)              # x mu*, governed by neither FPR nor T4

# --- Final-window for the stdout plurality label (§3.5 / --saturation-final-window)
# Set >= one burst-lull cycle. Used for the CLI final label; the harness computes
# its own per-run verdict from the trace (§3.5), so this is mainly for parity.
FINAL_WINDOW = "10s"

# --- Detectors under test (canonical roster order, sim/saturation/bank.go) ---
DETECTORS = ["composite", "threshold", "backlog-drift"]


def rung_run_requests(rate_rps: float) -> int:
    """Requests for a rung: larger of MIN_REQUESTS or MIN_CYCLES burst-lull
    cycles' worth of arrivals at this rate (§3.2)."""
    cycle_reqs = int(rate_rps * NOMINAL_CYCLE_S * MIN_CYCLES)
    return max(MIN_REQUESTS, cycle_reqs)
