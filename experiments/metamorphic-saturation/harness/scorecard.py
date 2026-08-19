"""Fill in the §9 scorecard template, one per detector."""

from __future__ import annotations

import config

DETECTOR_META = {
    "composite": {
        "signal": "max(rate-deficit, quartile-filtered latency-trend) vs 1/sqrt(arrivals) noise floor",
        "side": "client", "cadence": "per-event (per completion/arrival)",
        "cost": "O(window) — buffers arrivals + completions",
        "params": "none (parameterless; §3.4 threshold-tuning N/A)",
        "trend_or_level": "trend (rate deficit + latency slope)",
    },
    "threshold": {
        "signal": "mean E2E latency vs threshold_ms (binary STABLE/OVERLOADED)",
        "side": "client", "cadence": "per-event",
        "cost": "O(1) running mean",
        "params": "threshold_ms",
        "trend_or_level": "level (absolute mean latency)",
    },
    "backlog-drift": {
        "signal": "online OLS slope of in-flight (arrivals-completions) over trailing window, banded vs noise floor (K=3)",
        "side": "client", "cadence": "per-event",
        "cost": "O(window) — bucketed trailing window",
        "params": "window_size_sec (K hardcoded at 3.0)",
        "trend_or_level": "trend (backlog slope accumulator)",
    },
}


def _mult(x):
    return f"{x:.2f}x" if x is not None else "----"


def _lead_ms(us):
    return f"{us/1000:.0f}ms" if us is not None else "n/a"


def render(detector: str, ctx: dict) -> str:
    """ctx keys: fpr_target, fpr_fired, fpr_runs, mu_star, frozen_params,
    ladders (dict ladder->ResponseResult), rung_mults, run_reqs, cycles,
    seeds, t4 (dict: matched_lead_us, per_detector_flips, curve_str, cadence_note),
    substrate."""
    m = DETECTOR_META[detector]
    lad = ctx["ladders"]

    def line(name, key):
        r = lad.get(key)
        if r is None:
            return f"  {name:16s} SKIP"
        verdict = "PASS" if r.passed else "FAIL"
        scale = "nominal" if key != "L-CAP" else "rung's own cliff"
        return (f"  {name:16s} {verdict}   first firing: {_mult(r.first_fire_mult)} "
                f"{scale}")

    t1 = line("T1 rate", "L-RATE")
    t2o = line("T2-OUT out-size", "L-SIZE-OUT")
    t2i = line("T2-IN  in-size", "L-SIZE-IN")
    t3 = line("T3 capacity", "L-CAP")

    mono = " ".join(
        f"{k}={'y' if (lad.get(k) and lad[k].monotone) else 'n'}"
        for k in ["L-RATE", "L-SIZE-OUT", "L-SIZE-IN", "L-CAP"] if lad.get(k))
    gray = " ".join(
        f"{k}:{lad[k].gray_zone_firings}"
        for k in lad if lad[k].gray_zone_firings)

    t4 = ctx.get("t4", {})
    t4_flips = t4.get("flips")
    t4_verdict = "PASS" if (t4_flips == 0) else ("FAIL" if t4_flips is not None else "n/a")

    # Verdict rule (§9): T1 & T2-OUT & T4 PASS, and (T2-IN pass or documented),
    # and (T3 pass or documented).
    def passed(key):
        return lad.get(key) is not None and lad[key].passed
    t2in_ok = passed("L-SIZE-IN")
    t3_ok = passed("L-CAP")
    overall = (passed("L-RATE") and passed("L-SIZE-OUT") and (t4_flips == 0))
    coverage_note = []
    if not t2in_ok:
        coverage_note.append("T2-IN failure documented as prefill coverage gap")
    if not t3_ok:
        coverage_note.append("T3 failure documented as thrash coverage gap")
    verdict = "PASS" if overall else "FAIL"

    fpr_note = ""
    if detector == "composite":
        fpr_note = ("  NOTE: composite is parameterless — §3.4 threshold tuning is "
                    "N/A. Achieved FPR is reported as-is; cross-detector FPR "
                    "commensurability is only partial for this detector.")

    return f"""```
DETECTOR: {detector}
  Signal computed:      {m['signal']}
  Stack layer:          post-hoc trace analysis (sim/saturation)
  Side:                 {m['side']}
  Decision cadence:     {m['cadence']}
  Online cost:          {m['cost']}
  Style:                {m['trend_or_level']}

SCOPE CHECK
  Single instance:      yes  (--num-instances 1)
  S1 non-shedding:      yes  (--timeout -1, no --flow-control, always-admit)
  S2 one class:         yes  (single 'standard' SLO class)
  S3 relaxed SLO:       yes  (no SLO gating; backlog growth is the signal)
  Substrate:            {ctx.get('substrate', 'simulator')}  (size ladders exactly isolated; §3.2)

OPERATING POINT  (frozen at Step 3)
  Target FPR:           {ctx['fpr_target']*100:.0f}%
  Achieved FPR:         {ctx['fpr_fired']} firings on {ctx['fpr_runs']} calibration runs
  Nominal cliff mu*:    {ctx['mu_star']:.2f} req/s  (scaffold only, not a label)
  Frozen parameters:    {ctx.get('frozen_params', m['params'])}
{fpr_note}

LADDERS
  Rungs / range:        {config.RUNG_COUNT}  /  {config.LADDER_LOW}x to {config.LADDER_HIGH}x nominal (denser near 1.0x)
  Run length per rung:  >= {config.MIN_REQUESTS} requests / >= {config.MIN_CYCLES} burst-lull cycles
  Seeds per rung:       {len(config.SEEDS)}  (majority >= {config.SEED_MAJORITY})
  Workload mix:         E[I] {config.E_IN} tok  E[O] {config.E_OUT} tok  (decode+prefill, aggregated)
  Arrival arm:          {config.ARRIVAL_ARM}
  Burstiness shape:     {config.ARRIVAL_PROCESS} arrivals, CV={config.ARRIVAL_CV} (fixed across ALL rungs)
  Fired-level mapping:  FIRED = {{OVERLOADED, BACKLOGGED}}, NOT-FIRED = {{STABLE}}
  Rung decision rule:   warm-up {config.WARMUP_CYCLES} cycle, fired-fraction >= {config.FIRED_FRACTION}, seed vote {config.SEED_MAJORITY}/{len(config.SEEDS)}

RESPONSE TESTS   (per-rung majority verdicts, §3.5)
{t1}
{t2o}
{t2i}
{t3}
                  monotone from first firing?  {mono}
                  gray-zone firings:  {gray or 'none'}

TEMPORAL TEST (T4)
  Divergence rule:      backlog-proxy envelope-exceedance (§5.3)
  Matched lead time:    {_lead_ms(t4.get('matched_lead_us'))}  (common across compared detectors)
  Raw flip count (no dwell): {t4.get('raw_flips', 'n/a')}   (§5.3 severity measure — 0 = holds through bursts undwelled)
  Flip count @ matched: {t4_flips if t4_flips is not None else 'n/a'}      T4  {t4_verdict}   (PASS iff 0)
  Dwell w at that point:{_lead_ms(t4.get('w_us'))}   Fire-count k: {t4.get('k', 'n/a')}
  (lead time, flip count) curve:  {t4.get('curve_str', 'see results/t4-curve.csv')}
  {t4.get('cadence_note', 'all detectors per-event cadence => raw flip counts comparable')}

COVERAGE
  Overload — decode axis (T2-OUT):   {'covered' if passed('L-SIZE-OUT') else 'NOT covered'}
  Overload — prefill axis (T2-IN):   {'covered' if t2in_ok else 'NOT covered (coverage gap)'}
  Thrash mechanism (T3):             {'covered' if t3_ok else 'NOT covered (coverage gap)'}
  Untested by scope:    multi-replica routing; shedding-regime T4; absolute mu* accuracy

VERDICT
  PASS iff: T1 & T2-OUT & T4(0 flips) PASS, and (T2-IN pass or documented gap),
            and (T3 pass or documented gap).
  Result:   {verdict}    Notes: {'; '.join(coverage_note) if coverage_note else 'all response axes covered'}
```
"""
