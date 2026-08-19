"""Probe the nominal cliff mu* by sweeping arrival rate until the response rate
flatlines. See docs/CLIFF-PROBE.md for the full method and interpretation.

mu* is a rung-spacing scaffold only (config.MU_STAR), never a pass/fail input —
a rough bracket of the knee is sufficient (metamorphic_tests.md §3.3).

Usage:
  python3 probe_cliff.py                 # default rate grid + vLLM-default capacity
  python3 probe_cliff.py 8 20 35 45 90   # custom arrival-rate grid (req/s)
"""

from __future__ import annotations

import json
import subprocess
import sys

import config

# Default arrival-rate grid (req/s). Widen if the plateau is not yet visible.
DEFAULT_RATES = [8, 20, 35, 40, 45, 55, 90]


def probe_one(rate: float, num_requests: int = 800) -> dict:
    cmd = [
        config.BLIS_BIN, "run",
        "--model", config.MODEL,
        "--latency-model", config.LATENCY_MODEL,
        "--rate", str(rate),
        "--num-requests", str(num_requests),
        "--output-tokens", str(config.E_OUT), "--output-tokens-stdev", "0",
        "--prompt-tokens", str(config.E_IN), "--prompt-tokens-stdev", "0",
        "--max-num-seqs", str(config.HEALTHY_B),
        "--max-num-batched-tokens", str(config.MAX_NUM_BATCHED_TOKENS),
        "--gpu-memory-utilization", str(config.GPU_MEMORY_UTILIZATION),
        "--timeout", str(config.TIMEOUT),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=config.REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"blis run failed at rate={rate}:\n{proc.stderr[-1500:]}")
    lines = proc.stdout.splitlines()
    if lines and lines[0].startswith("==="):
        lines = lines[1:]
    return json.loads("\n".join(lines))


def main():
    rates = [float(a) for a in sys.argv[1:]] or DEFAULT_RATES
    print(f"model={config.MODEL} B={config.HEALTHY_B} E[I]={config.E_IN} "
          f"E[O]={config.E_OUT}  (vLLM defaults, --timeout -1)")
    print(f"{'arrival':>8} {'tok/s':>8} {'resp/s':>8} {'e2e_ms':>8} {'queued':>7} {'running':>7}")
    prev_resp = None
    ceiling = None
    for r in rates:
        d = probe_one(r)
        resp = d["responses_per_sec"]
        print(f"{r:>8.1f} {d['tokens_per_sec']:>8.0f} {resp:>8.2f} "
              f"{d['e2e_mean_ms']:>8.0f} {d['still_queued']:>7d} {d['still_running']:>7d}")
        # Track the plateau: when response stops rising materially, record it.
        if prev_resp is not None and resp <= prev_resp * 1.02:
            ceiling = resp if ceiling is None else max(ceiling, resp)
        prev_resp = resp
    if ceiling is None:
        ceiling = prev_resp   # never plateaued in this grid — widen it
        print("\nWARNING: response rate never plateaued in this grid — widen the "
              "rates (add higher values) to bracket the knee.")
    mu_from_resp = ceiling
    mu_from_tokens = None
    # r_dec plateau / E[O] as a cross-check (uses the last, saturated tok/s).
    try:
        mu_from_tokens = probe_one(max(rates))["tokens_per_sec"] / config.E_OUT
    except Exception:
        pass
    print(f"\nCapacity (response-rate plateau) ~= {mu_from_resp:.1f} req/s")
    if mu_from_tokens:
        print(f"Cross-check r_dec/E[O]            ~= {mu_from_tokens:.1f} req/s")
    print(f"config.MU_STAR is currently set to    {config.MU_STAR} req/s")
    print("If these disagree materially, update config.MU_STAR and re-run "
          "`python3 run_suite.py all`.")


if __name__ == "__main__":
    main()
