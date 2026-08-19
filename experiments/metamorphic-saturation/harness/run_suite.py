"""Top-level orchestrator for the metamorphic saturation suite.

Runs, in order (metamorphic_tests.md §7):
  STEP 3  FPR calibration -> freeze threshold_ms & window_size_sec configs.
  STEP 5  Response tests T1, T2-OUT, T2-IN, T3 over the 4 ladders.
  STEP 6  T4 temporal on held-mean overloaded traces.
  STEP 7  Emit one §9 scorecard per detector + ladder-verdicts.csv.

Usage:
  python3 run_suite.py calibrate      # Step 3 only (writes configs/)
  python3 run_suite.py response       # Steps 5 (needs frozen configs)
  python3 run_suite.py t4             # Step 6
  python3 run_suite.py scorecards     # Step 7 (from cached runs)
  python3 run_suite.py all            # everything
"""

from __future__ import annotations

import csv
import os
import sys

import yaml

import config
import driver
import parse
import reduce as reduce_mod
import score
import scorecard
from ladder import LADDERS, build_ladder, rung_multiples

THRESHOLD_CFG = os.path.join(config.CONFIGS_DIR, "sat-threshold.yaml")
BACKLOGDRIFT_CFG = os.path.join(config.CONFIGS_DIR, "sat-merged.yaml")
# Merged config carrying both tunable blocks; valid under --detectors all.
MERGED_CFG = os.path.join(config.CONFIGS_DIR, "sat-merged.yaml")


# ---------------------------------------------------------------------------
# STEP 3 — FPR calibration
# ---------------------------------------------------------------------------

def _calib_rungs():
    """Build calibration rungs (L-RATE at the sub-capacity band, §3.4)."""
    import specs
    from ladder import Rung
    rungs = []
    for m in config.CALIB_RUNGS:
        rate = round(m * config.MU_STAR, 4)
        rungs.append(Rung("CALIB", int(m * 100), m, rate_rps=rate,
                          input_tokens=config.E_IN, output_tokens=config.E_OUT,
                          max_num_seqs=config.HEALTHY_B,
                          label=f"CALIB x{m:.2f}"))
    return rungs


def _write_merged_config(threshold_ms: float, window_size_sec: int):
    os.makedirs(config.CONFIGS_DIR, exist_ok=True)
    cfg = {
        "threshold": {"threshold_ms": float(threshold_ms)},
        "backlog_drift": {"window_size_sec": int(window_size_sec)},
    }
    with open(MERGED_CFG, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _calib_reports(sat_config):
    """Run the calibration band; return {seed_rung_tag: Report} and metrics."""
    import specs
    reps = []
    for rung in _calib_rungs():
        for seed in config.SEEDS:
            spec = specs.write_spec(rung, seed)
            tag = f"CALIB_x{rung.load_mult:.2f}_seed{seed}".replace(".", "p")
            _, rpath = driver.run_blis(spec, rung.max_num_seqs, tag,
                                       sat_config=sat_config, force=True)
            reps.append((tag, parse.load_report(rpath)))
    return reps


def _fpr_for(reports, detector) -> int:
    """How many calibration runs FIRED for this detector (§3.5 run verdict)."""
    fired = 0
    for _tag, rep in reports:
        recs = rep.per_detector.get(detector, [])
        if reduce_mod.run_verdict(recs):
            fired += 1
    return fired


def calibrate():
    print("STEP 3: FPR calibration on the 0.3-0.6x band "
          f"({len(config.CALIB_RUNGS)} rungs x {len(config.SEEDS)} seeds = "
          f"{len(config.CALIB_RUNGS)*len(config.SEEDS)} runs)")
    n_runs = len(config.CALIB_RUNGS) * len(config.SEEDS)
    max_fire = int(config.TARGET_FPR * n_runs)  # <=5% => <=1 on 20 runs

    # threshold_ms sweep (ascending: higher threshold => fewer firings).
    threshold_grid = [2000, 4000, 6000, 8000, 12000, 20000, 40000, 80000]
    # window_size_sec sweep for backlog-drift (its only knob).
    window_grid = [10, 30, 60, 120, 300]

    # Coarse joint search: pick the smallest threshold_ms and window that each
    # hit <= max_fire. Because the bank runs all three off one replay, we sweep
    # by re-running with candidate merged configs and reading per-detector FPR.
    chosen_threshold = threshold_grid[-1]
    chosen_window = window_grid[0]

    # Sweep threshold first (window fixed at default 60 during this leg).
    thr_results = []
    for thr in threshold_grid:
        _write_merged_config(thr, 60)
        reps = _calib_reports(MERGED_CFG)
        fired = _fpr_for(reps, "threshold")
        thr_results.append((thr, fired))
        print(f"  threshold_ms={thr:6d} -> {fired}/{n_runs} fired")
        if fired <= max_fire:
            chosen_threshold = thr
            break
    else:
        chosen_threshold = threshold_grid[-1]

    # Sweep window for backlog-drift (threshold fixed at chosen).
    win_results = []
    for win in window_grid:
        _write_merged_config(chosen_threshold, win)
        reps = _calib_reports(MERGED_CFG)
        fired = _fpr_for(reps, "backlog-drift")
        win_results.append((win, fired))
        print(f"  window_size_sec={win:4d} -> backlog-drift {fired}/{n_runs} fired")
        if fired <= max_fire:
            chosen_window = win
            break
    else:
        chosen_window = window_grid[-1]

    _write_merged_config(chosen_threshold, chosen_window)
    # Composite FPR at defaults (measured, not tuned).
    reps = _calib_reports(MERGED_CFG)
    comp_fired = _fpr_for(reps, "composite")
    thr_fired = _fpr_for(reps, "threshold")
    bd_fired = _fpr_for(reps, "backlog-drift")

    fpr = {
        "n_runs": n_runs, "max_fire": max_fire,
        "threshold": {"threshold_ms": chosen_threshold, "fired": thr_fired},
        "backlog-drift": {"window_size_sec": chosen_window, "fired": bd_fired},
        "composite": {"params": "none", "fired": comp_fired},
    }
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "fpr.yaml"), "w") as f:
        yaml.safe_dump(fpr, f, sort_keys=False)
    print(f"FROZEN: threshold_ms={chosen_threshold}, window_size_sec={chosen_window}")
    print(f"Achieved FPR (fired/{n_runs}): threshold={thr_fired} "
          f"backlog-drift={bd_fired} composite={comp_fired} (composite untunable)")
    return fpr


# ---------------------------------------------------------------------------
# STEP 5 — response tests
# ---------------------------------------------------------------------------

def _run_ladder(ladder_name, sat_config):
    """Run every rung x seed of a ladder; return per-detector ResponseResult."""
    import specs
    rungs = build_ladder(ladder_name)
    mults = [r.load_mult for r in rungs]
    # reports[detector][rung_idx] = [records_per_seed]
    reports = {d: [[] for _ in rungs] for d in config.DETECTORS}
    per_run_meta = {d: [[None] * len(config.SEEDS) for _ in rungs] for d in config.DETECTORS}
    for ri, rung in enumerate(rungs):
        for si, seed in enumerate(config.SEEDS):
            spec = specs.write_spec(rung, seed)
            tag = driver.run_tag(rung, seed)
            mpath, rpath = driver.run_blis(spec, rung.max_num_seqs, tag,
                                           sat_config=sat_config)
            rep = parse.load_report(rpath)
            for d in config.DETECTORS:
                reports[d][ri].append(rep.per_detector.get(d, []))

    results = {}
    verdict_rows = []
    for d in config.DETECTORS:
        rung_fired = []
        per_seed = []
        for ri in range(len(rungs)):
            fired, seeds_v = reduce_mod.rung_verdict(reports[d][ri])
            rung_fired.append(fired)
            per_seed.append(seeds_v)
            verdict_rows.append({
                "detector": d, "ladder": ladder_name, "rung": ri,
                "load_mult": mults[ri], "fired": fired,
                "per_seed": "".join("1" if s else "0" for s in seeds_v),
            })
        results[d] = score.score_response(ladder_name, mults, rung_fired, per_seed)
    return results, verdict_rows, mults


def response():
    sat_config = MERGED_CFG if os.path.exists(MERGED_CFG) else None
    if sat_config is None:
        print("WARNING: no frozen config found; running response tests at defaults.")
    all_results = {d: {} for d in config.DETECTORS}
    all_rows = []
    for lad in LADDERS:
        print(f"STEP 5: response ladder {lad}")
        res, rows, _mults = _run_ladder(lad, sat_config)
        for d in config.DETECTORS:
            all_results[d][lad] = res[d]
        all_rows.extend(rows)
    # Persist verdict audit trail.
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "ladder-verdicts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["detector", "ladder", "rung", "load_mult", "fired", "per_seed"])
        w.writeheader()
        w.writerows(all_rows)
    return all_results


# ---------------------------------------------------------------------------
# STEP 6 — T4 temporal
# ---------------------------------------------------------------------------

def t4():
    import specs
    sat_config = MERGED_CFG if os.path.exists(MERGED_CFG) else None
    mean_mult = 1.3
    # backlog envelope (§3.1/§5.3): a sub-capacity burst can transiently push the
    # in-flight backlog proxy up; the envelope is the level above which only a
    # genuinely super-capacity mean sustains. At vLLM defaults the 1.3x-overload
    # in-flight climbs into the several-hundreds (probed max ~540) while a
    # sub-capacity run settles near ~B; envelope = ENVELOPE_MULT x HEALTHY_B.
    envelope = config.T4_ENVELOPE_MULT * config.HEALTHY_B

    curves = {d: [] for d in config.DETECTORS}
    for seed in config.SEEDS:
        spec = specs.write_t4_spec(mean_mult, seed)
        tag = f"T4_mean{str(mean_mult).replace('.', 'p')}_seed{seed}"
        _, rpath = driver.run_blis(spec, config.HEALTHY_B, tag, sat_config=sat_config)
        rep = parse.load_report(rpath)
        # The backlog is a property of the RUN — compute t_div ONCE from the
        # shared in-flight trajectory (backlog-drift carries in_flight) and pass
        # the same value to every detector's curve (§5.3).
        bd_recs = rep.per_detector.get("backlog-drift", [])
        traj = [(r.timestamp, r.signals["in_flight"]) for r in bd_recs if "in_flight" in r.signals]
        t_div = score.divergence_from_trajectory(traj, envelope)
        for d in config.DETECTORS:
            recs = rep.per_detector.get(d, [])
            curves[d].append(score.t4_curve(recs, envelope, t_div=t_div))

    # Aggregate: for each detector, the WORST (max) flip count across seeds at
    # each w (conservative), and the zero-flip lead per seed.
    agg = {}
    for d in config.DETECTORS:
        # Average the curve across seeds by w index.
        n_w = len(curves[d][0]) if curves[d] else 0
        pts = []
        for wi in range(n_w):
            flips = max(c[wi].flips for c in curves[d])   # worst-case
            leads = [c[wi].lead_time_us for c in curves[d] if c[wi].lead_time_us is not None]
            lead = min(leads) if leads else None          # worst-case lead
            w_us = curves[d][0][wi].w_us
            k = curves[d][0][wi].k
            pts.append(score.T4Point(w_us=w_us, k=k, flips=flips, lead_time_us=lead))
        agg[d] = pts

    # Matched lead time: largest lead every detector reaches at 0 flips (§5.3).
    zero_leads = {d: score.zero_flip_lead(agg[d]) for d in config.DETECTORS}
    reachable = [v for v in zero_leads.values() if v is not None]
    matched = min(reachable) if len(reachable) == len(config.DETECTORS) else (
        min(reachable) if reachable else None)

    # Flip count at matched lead per detector: the flip count at the smallest w
    # whose lead >= matched (or the raw w=0 point if the detector never reached 0).
    t4_out = {}
    for d in config.DETECTORS:
        raw_flips = agg[d][0].flips          # w=0: the undwelled severity measure (§5.3)
        chosen = None
        for p in agg[d]:
            if p.flips == 0 and p.lead_time_us is not None and (matched is None or p.lead_time_us >= matched):
                if chosen is None or p.w_us < chosen.w_us:
                    chosen = p
        if chosen is None:
            # Never reaches 0 flips at any w => fails T4 outright.
            raw = agg[d][0]
            t4_out[d] = {"flips": raw.flips, "raw_flips": raw_flips,
                         "w_us": raw.w_us, "k": raw.k, "matched_lead_us": matched,
                         "curve_str": _curve_str(agg[d]),
                         "cadence_note": f"per-event; raw(no-dwell) flips={raw_flips}"}
        else:
            t4_out[d] = {"flips": 0, "raw_flips": raw_flips,
                         "w_us": chosen.w_us, "k": chosen.k, "matched_lead_us": matched,
                         "curve_str": _curve_str(agg[d]),
                         "cadence_note": f"per-event; raw(no-dwell) flips={raw_flips} "
                                         f"=> {'holds through bursts' if raw_flips == 0 else 'needs dwell w='+str(chosen.w_us//1000)+'ms to stop flip-flopping'}"}
    # Persist curve
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "t4-curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "w_us", "k", "flips", "lead_us"])
        for d in config.DETECTORS:
            for p in agg[d]:
                w.writerow([d, p.w_us, p.k, p.flips, p.lead_time_us])
    return t4_out


def _curve_str(pts):
    return "; ".join(f"w={p.w_us//1000}ms:flips={p.flips},lead={'%.0f' % (p.lead_time_us/1000) if p.lead_time_us is not None else 'na'}ms"
                     for p in pts)


# ---------------------------------------------------------------------------
# STEP 7 — scorecards
# ---------------------------------------------------------------------------

def scorecards(all_results=None, t4_out=None):
    if all_results is None:
        all_results = response()
    if t4_out is None:
        t4_out = t4()

    # Load frozen FPR summary.
    fpr_path = os.path.join(config.RESULTS_DIR, "fpr.yaml")
    fpr = yaml.safe_load(open(fpr_path)) if os.path.exists(fpr_path) else {}
    n_runs = fpr.get("n_runs", len(config.CALIB_RUNGS) * len(config.SEEDS))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    for d in config.DETECTORS:
        fired = fpr.get(d, {}).get("fired", "?")
        frozen = {
            "threshold": f"threshold_ms={fpr.get('threshold', {}).get('threshold_ms', 'default')}",
            "backlog-drift": f"window_size_sec={fpr.get('backlog-drift', {}).get('window_size_sec', 'default')}, K=3.0 (hardcoded)",
            "composite": "none (parameterless)",
        }[d]
        ctx = {
            "fpr_target": config.TARGET_FPR,
            "fpr_fired": fired,
            "fpr_runs": n_runs,
            "mu_star": config.MU_STAR,
            "frozen_params": frozen,
            "ladders": all_results[d],
            "rung_mults": rung_multiples(),
            "substrate": "simulator",
            "t4": t4_out.get(d, {}),
        }
        text = scorecard.render(d, ctx)
        out = os.path.join(config.RESULTS_DIR, f"scorecard-{d}.md")
        with open(out, "w") as f:
            f.write(f"# Metamorphic scorecard — `{d}`\n\n")
            f.write(text)
        print(f"wrote {out}")


def _run_one_arm():
    """Full pipeline for the CURRENTLY-selected arm (config.ARRIVAL_ARM)."""
    calibrate()
    res = response()
    t4_out = t4()
    scorecards(res, t4_out)


def run_both_arms():
    """Run the full suite for EVERY arrival arm, then write the comparison.

    Arm selection is fixed at config import time (env var), so each arm runs in
    its own subprocess with ARRIVAL_ARM set. Results land in per-arm dirs
    (results/<arm>/), and compare.py stitches them into results/SUMMARY.md.
    """
    import compare
    arms = list(config._ARMS.keys())   # e.g. ["gamma", "poisson"]
    for arm in arms:
        print(f"\n===== ARM: {arm} =====")
        env = dict(os.environ, ARRIVAL_ARM=arm)
        proc = subprocess.run(
            [sys.executable, "-u", __file__, "all-one"],
            env=env, cwd=config.HARNESS_DIR)
        if proc.returncode != 0:
            raise RuntimeError(f"arm {arm} failed (exit {proc.returncode})")
    compare.write_comparison(arms)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "calibrate":
        calibrate()
    elif what == "response":
        response()
    elif what == "t4":
        t4()
    elif what == "scorecards":
        scorecards()
    elif what == "all-one":
        # Single arm (the one selected by ARRIVAL_ARM). Used internally by
        # run_both_arms, and available directly for a fast one-arm run.
        _run_one_arm()
    elif what in ("all", "both"):
        # DEFAULT: run every arrival arm (gamma + poisson) and emit the
        # side-by-side comparison. Use `all-one` for a single arm.
        run_both_arms()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
