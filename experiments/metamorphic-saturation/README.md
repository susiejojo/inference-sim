# Metamorphic Saturation-Detector Test Suite (on BLIS)

A runnable implementation of the **metamorphic test suite** from
`~/Downloads/papers/saturation/metamorphic_tests.md` (companion:
`morphological_properties.md`), driven entirely through `blis run` against the
**Saturation Bank** (`--detectors all`) so all three post-hoc detectors —
`composite`, `threshold`, `backlog-drift` — are scored on **byte-identical
traffic in one deterministic replay**.

## Why a metamorphic suite

A saturation detector answers "is this system saturated right now?" continuously.
There is **no ground-truth label** on a real (or simulated) trace — saturation is
a property of the *unfinished-work process* (does backlog grow without bound?),
which a finite trace cannot reveal. So instead of scoring answers against a label,
the suite makes **one controlled change whose direction is known a-priori** and
checks the detector's answer moved the way it logically must.

| Test | Ladder | Controlled change | Catches |
|---|---|---|---|
| **T1** | L-RATE | ↑ arrivals/s | inert detectors |
| **T2-OUT** | L-SIZE-OUT | ↑ output tokens/req (rate fixed) | arrival-rate-keyed detectors |
| **T2-IN** | L-SIZE-IN | ↑ input tokens/req (rate fixed) | decode-only detectors (prefill-blind) |
| **T3** | L-CAP | ↓ server capacity (workload fixed) | workload-only / thrash-blind detectors |
| **T4** | — | held overloaded mean + bursts | level detectors that flip-flop |

## Apparatus (frozen; see `harness/config.py`)

- Model `qwen/qwen3-14b`, trained-physics, single instance, **non-shedding**
  (`--timeout -1`, no flow control), one SLO class.
- **vLLM defaults** for capacity: `--max-num-seqs 256`, `--max-num-batched-tokens
  2048`, `--gpu-memory-utilization 0.9` → probed nominal cliff **μ\* ≈ 28.7 req/s**
  (scaffold for rung spacing only, §3.3; not a pass/fail input). How μ\* was
  measured and how to reproduce it: **[`docs/CLIFF-PROBE.md`](docs/CLIFF-PROBE.md)**
  (or run `python3 harness/probe_cliff.py`). `--max-num-seqs` is the T3 (L-CAP)
  shrink knob: L-CAP sweeps B geometrically 256 → 40 (the fixed 0.7×μ\* workload
  is healthy at B≥96, overloaded by B≤64).
- Mix: constant `E[I]=512`, `E[O]=256` tokens (constant dists ⇒ exact axis
  isolation on the simulator).
- Burstiness: **gamma arrivals, CV=3.0, held fixed across ALL rungs of ALL
  ladders** (§3.2). Only the named axis varies per ladder.
- 12 rungs/ladder, geometric 0.3×–1.6× μ\*, denser near 1.0×. 5 seeds/rung,
  majority vote (≥3/5). Rung verdict (§3.5): drop 1 warm-up cycle, FIRED iff the
  detector is in a fired level for ≥50% of the remaining run **time**.
- **Fired-level mapping** (declared, not hidden): the bank emits 3 levels;
  FIRED = {OVERLOADED, BACKLOGGED}, NOT-FIRED = {STABLE}.

## FPR calibration is mandatory and comes first (§3.4)

All metrics are read at one fixed false-positive rate so detectors are
commensurable. Only **`threshold`** (`threshold_ms`) and **`backlog-drift`**
(`window_size_sec`) expose a tunable knob; the harness sweeps each to ≤5% firing
on the 0.3–0.6× sub-capacity band and **freezes** it (`configs/sat-merged.yaml`).
**`composite` is parameterless** — it runs at defaults and its achieved FPR is
reported honestly as a suite limitation (cross-detector commensurability is only
partial for it). This is the "calibrate what's tunable, report the rest" policy.

## Running

```bash
# from repo root: build once
go build -o blis main.go

cd experiments/metamorphic-saturation/harness
python3 run_suite.py all          # DEFAULT: runs BOTH arrival arms + comparison
python3 run_suite.py all-one      # a single arm (the one ARRIVAL_ARM selects)
python3 run_suite.py calibrate    # Step 3 — freezes configs/<arm>/sat-merged.yaml + fpr.yaml
python3 run_suite.py response     # Steps 5 — T1/T2-OUT/T2-IN/T3 over the 4 ladders
python3 run_suite.py t4           # Step 6 — temporal flip-flop test
python3 run_suite.py scorecards   # Step 7 — emit results/<arm>/scorecard-*.md
```

### Arrival arms (burstiness control)

The suite runs over two arrival "arms" so a failure can be attributed to
*burstiness* vs. *the detector itself*:

- **gamma** (CV=3.0) — bursty; the default single-arm and the shape T4 needs.
- **poisson** (CV=1.0) — the burstiness **control**; smoother arrivals.

`run_suite.py all` runs **both** arms back-to-back (each in its own subprocess
with `ARRIVAL_ARM` set) and writes **one consolidated `results/SUMMARY.md`**
spanning both arms — the overall verdict table plus the causal readings that
answer "is composite's FPR blowup / backlog-drift's failures caused by bursts,
or intrinsic?". For a single arm, use `all-one`:

```bash
python3 run_suite.py all                     # both arms + results/SUMMARY.md
ARRIVAL_ARM=gamma   python3 run_suite.py all-one   # bursty arm only
ARRIVAL_ARM=poisson python3 run_suite.py all-one   # Poisson arm only
```

Each arm has its OWN subdirs: `specs/<arm>/`, `runs/<arm>/`, `configs/<arm>/`,
`results/<arm>/`. Runs are cached (resumable); `specs/` and `runs/` are
git-ignored.

## Outputs

- **`results/SUMMARY.md`** — the single overall summary across all arms: verdict
  table (detector × arm), FPR, T4 flips+lead, per-ladder first-firing/monotonicity,
  auto-derived causal readings, bottom line. **Start here.**
- Per-arm detail in `results/<arm>/`:
  - `scorecard-{composite,threshold,backlog-drift}.md` — the filled §9 scorecards.
  - `ladder-verdicts.csv` — per (detector, ladder, rung, seed) FIRED audit trail.
  - `t4-curve.csv` — the (dwell w, flip count, lead time) curve per detector.
  - `fpr.yaml` — frozen operating point (chosen params + achieved firings).

## How to read a scorecard

- **Response tests** pass iff the detector fires at some rung and stays fired at
  every heavier rung (monotone-from-first-firing). **Sharpness** is the
  first-firing rung as a load multiple of μ\* — lower = earlier warning — and is
  only meaningful *relative to another detector on the same ladder* (§4.4), never
  as an absolute.
- **T4** passes iff flip count is 0 at the **matched lead time** (the largest
  lead time every detector reaches at 0 flips). A detector that reaches 0 flips
  only by an ever-larger dwell window has failed — matching prevents ranking
  dwell tuning instead of detector quality (§5.3).
- **Coverage** states explicitly whether the decode, prefill, and thrash
  mechanisms are covered. A T2-IN or T3 failure is reported as a documented
  *coverage gap*, not a defect, per the papers.

## Scope / caveats

Single instance, non-shedding, one class, relaxed SLO (§2 S1–S3). Multi-replica
routing, shedding-regime T4 (where T4 is *false*, not merely untested), and
absolute μ\* accuracy are out of scope. Detectors see a thin event stream
(arrival timestamps + E2E latency only; `sim/saturation/replay.go`
`buildSortedEvents`), so T2-IN/T2-OUT test size-sensitivity *through* the
backlog/latency mechanism — exactly as the papers' §T2 precision note
anticipates.
