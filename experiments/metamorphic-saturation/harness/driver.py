"""Invoke `blis run` per (ladder, rung, seed) and collect its outputs.

Each invocation runs the FULL detector bank in one deterministic replay
(`--detectors all`), writing:
  - the metrics JSON (stdout, after a `=== Simulation Metrics ===` header line)
  - the saturation report ({"final":..,"trace":..}) to --saturation-report

The driver never re-runs a (rung,seed) whose report already exists unless
force=True, so a long sweep is resumable.
"""

from __future__ import annotations

import os
import subprocess

import config
from ladder import Rung


def metrics_path(tag: str) -> str:
    return os.path.join(config.RUNS_DIR, f"{tag}.metrics.json")


def report_path(tag: str) -> str:
    return os.path.join(config.RUNS_DIR, f"{tag}.sat.json")


def run_tag(rung: Rung, seed: int) -> str:
    return f"{rung.ladder}_r{rung.index:02d}_seed{seed}"


def run_blis(spec_file: str, max_num_seqs: int, tag: str,
             sat_config: str | None = None, force: bool = False) -> tuple[str, str]:
    """Run one blis simulation. Returns (metrics_path, report_path).

    sat_config: path to a --saturation-config YAML (threshold/backlog_drift
    tuning). None => library defaults. Composite ignores it either way.
    """
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    mpath = metrics_path(tag)
    rpath = report_path(tag)
    if not force and os.path.exists(mpath) and os.path.exists(rpath):
        return mpath, rpath

    cmd = [
        config.BLIS_BIN, "run",
        "--model", config.MODEL,
        "--latency-model", config.LATENCY_MODEL,
        "--workload-spec", spec_file,
        "--max-num-seqs", str(max_num_seqs),
        # Pin the other vLLM defaults explicitly so the apparatus is stable even
        # if BLIS's built-in defaults ever change.
        "--max-num-batched-tokens", str(config.MAX_NUM_BATCHED_TOKENS),
        "--gpu-memory-utilization", str(config.GPU_MEMORY_UTILIZATION),
        "--timeout", str(config.TIMEOUT),
        "--detectors", "all",
        "--saturation-report", rpath,
        "--saturation-final-window", config.FINAL_WINDOW,
    ]
    if sat_config:
        cmd += ["--saturation-config", sat_config]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=config.REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"blis run failed for {tag} (exit {proc.returncode}):\n"
                           f"{proc.stderr[-2000:]}")
    # Strip the `=== Simulation Metrics ===` header line before the JSON body.
    lines = proc.stdout.splitlines()
    if lines and lines[0].startswith("==="):
        lines = lines[1:]
    with open(mpath, "w") as f:
        f.write("\n".join(lines))
    return mpath, rpath
