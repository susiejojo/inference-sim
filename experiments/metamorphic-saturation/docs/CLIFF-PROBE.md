# How the nominal cliff μ\* was found (and how to reproduce it)

The metamorphic suite lays every ladder out relative to a **nominal cliff μ\***
— the arrival rate at which the server tips from "keeping up" to "falling
behind." This document records how μ\* was measured for the frozen apparatus
(`qwen/qwen3-14b`, trained-physics, vLLM defaults) and how to reproduce or
re-measure it.

> **μ\* is a scaffold, not a label.** It is used *only* to place ladder rungs
> (`config.MU_STAR`, consumed in `ladder.py`/`specs.py`). No pass/fail verdict
> reads it. Being ±20 % off just stretches the ladder; it does not invalidate a
> result (`metamorphic_tests.md` §3.3). So a rough probe is sufficient — do not
> over-engineer this.

---

## 1. What the cliff is

Two distinct quantities, easy to conflate:

- **Arrival rate** — how fast requests are *offered* (`--rate` / `aggregate_rate`).
  The input knob.
- **Response rate** (`responses_per_sec`) — how fast the server *completes*
  requests. A measured output.

The cliff is on the **arrival** axis: it is the arrival rate at which the server
can no longer match what is offered. You *locate* it by watching the **response
rate** stop rising:

- **Below the cliff:** offering more arrivals yields more completions — response
  rate climbs, and latency/backlog stay **bounded** (queue fully drains).
- **At/above the cliff:** the response rate is **pinned at the server's ceiling**
  no matter how many more arrivals you offer; the surplus becomes unbounded
  backlog and runaway latency.

The cliff μ\* equals that ceiling, because the server tips over exactly when
offered load reaches the rate it can serve.

**Important subtlety:** below the cliff the response rate can still lag the
arrival rate *slightly* (e.g. offer 20, complete 18) — that is a finite-run
artifact (batch fill at the start, in-flight requests draining at the end), NOT
saturation. The reliable signal is not "response == arrival"; it is **"response
rate stops rising as you push arrivals higher, while latency stays bounded."**

---

## 2. The method: sweep arrivals until the response rate flatlines

Run the model at a ladder of increasing arrival rates with a **fixed workload**
(constant input/output token counts) and watch three columns:

1. `responses_per_sec` — the completion rate (the thing that plateaus).
2. `tokens_per_sec` — the token-throughput ceiling `r_dec` (plateaus at the same
   point; a second view of the same limit).
3. `e2e_mean_ms` — average latency (starts running away past the cliff).

Also confirm S1 (non-shedding) holds by running with `--timeout -1` and checking
`timed_out_requests == 0` / `dropped_unservable == 0`.

### Measured sweep (vLLM defaults, E[I]=512, E[O]=256, B=256)

Reproduced on this machine:

| arrival req/s | tokens/s | **response req/s** | e2e_mean_ms | queued | running |
|---:|---:|---:|---:|---:|---:|
| 8  | 1981 |  7.74 |  3692 | 0 | 0 |
| 20 | 4652 | 18.17 |  4910 | 0 | 0 |
| 35 | 7005 | 27.36 |  7421 | 0 | 0 |
| 40 | 7176 | 28.03 |  8392 | 0 | 0 |
| 45 | 7282 | **28.45** |  9232 | 0 | 0 |
| 55 | 7315 | **28.57** | 10514 | 0 | 0 |
| 90 | 7371 | **28.79** | 12781 | 0 | 0 |

**Reading it:**
- Response rate rises with arrivals up to ~rate 35–40, then **flattens at
  ≈ 28.6–28.8 req/s** — pushing arrivals from 45 → 90 (2×) barely moves
  completions (28.45 → 28.79). That plateau is the server's serving ceiling.
- `tokens_per_sec` plateaus in lockstep at **r_dec ≈ 7350 tok/s**.
- `e2e_mean_ms` climbs monotonically past the knee (7.4 s → 12.8 s) — the latency
  runaway that confirms the queue is no longer draining fast enough.

So the **capacity = the response rate that stops growing ≈ 28.7 req/s**.

### Cross-check via the token ceiling

The paper's decode-dominant formula gives the same number a second way
(`metamorphic_tests.md` §3.3, `μ* = r_dec / E[O]`):

```
μ* = r_dec / E[O] = 7350 tok/s ÷ 256 tok/req ≈ 28.7 req/s
```

The response-rate plateau (≈28.7) and the token-derived value (≈28.7) agree to
within rounding. `config.MU_STAR = 28.7` is set to this value.

The token formula is not strictly necessary to *find* μ\* (the response-rate
plateau shows it directly), but it is what lets you **predict** how μ\* shifts if
the request size changes without re-sweeping: e.g. doubling E[O] to 512 halves
the cliff to ≈14 req/s. The size ladders (T2-OUT/T2-IN) rely on that
size↔cliff relationship.

---

## 3. Reproduce it

From the repo root, with a built `blis`:

```bash
go build -o blis main.go

for rate in 8 20 35 40 45 55 90; do
  ./blis run --model qwen/qwen3-14b --rate $rate --num-requests 800 \
    --output-tokens 256 --output-tokens-stdev 0 \
    --prompt-tokens 512 --prompt-tokens-stdev 0 \
    --max-num-seqs 256 --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.9 --timeout -1 2>/dev/null \
  | tail -n +2 \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(f\"arrival=$rate resp/s={d['responses_per_sec']:.2f} tok/s={d['tokens_per_sec']:.0f} e2e_ms={d['e2e_mean_ms']:.0f} q={d['still_queued']} run={d['still_running']}\")"
done
```

Or use the committed helper (same command matrix):

```bash
cd experiments/metamorphic-saturation/harness
python3 probe_cliff.py            # prints the sweep table + the derived μ*
```

**Flags that matter and why:**
- `--max-num-seqs 256 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.9`
  — vLLM defaults; these set the capacity, so they must match the suite's frozen
  apparatus (`config.py`).
- `--output-tokens-stdev 0 --prompt-tokens-stdev 0` — constant sizes so the cliff
  is a single clean number (matches the suite's constant token distributions).
- `--timeout -1` — S1 non-shedding; without it, overloaded requests hit the 300 s
  deadline and get dropped, which flatters throughput and hides the cliff.
- `2>/dev/null` + `tail -n +2` — drop stderr warnings and the
  `=== Simulation Metrics ===` header line so the body is valid JSON.

**How to read the result:** find the arrival rate past which `resp/s` stops
increasing (here ~40–45) — that plateau value (~28.7) is μ\*. Widen the rate grid
if the plateau is not yet visible; the goal is only to bracket the knee, not to
pin it precisely.

---

## 4. Re-measuring after any apparatus change

Re-run the probe (and update `config.MU_STAR`) whenever you change anything that
moves the serving ceiling or the per-request work:

- **model** (`--model`) or **hardware/TP** — different `r_dec`.
- **capacity** (`--max-num-seqs`, `--max-num-batched-tokens`,
  `--gpu-memory-utilization`, KV blocks) — different ceiling. (Historical note: an
  earlier exploratory run used `--max-num-seqs 8`, which put μ\* at ≈2.4 req/s;
  moving to vLLM defaults raised it to ≈28.7.)
- **workload mix** (`E[I]`, `E[O]`) — the token formula rescales μ\* by 1/E[O]
  (decode-dominant); re-probe for mixed/prefill-heavy workloads since the
  decode-only formula overestimates there (`metamorphic_tests.md` §3.3).

After updating `config.MU_STAR`, regenerate specs and re-run the suite
(`python3 run_suite.py all`) — the frozen FPR configs and ladders are all
relative to μ\*, so they should be rebuilt together.
