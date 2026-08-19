"""Build results/SUMMARY.md — one consolidated summary across the arrival arms.

Reads each arm's per-arm result set (results/<arm>/fpr.yaml,
ladder-verdicts.csv, t4-curve.csv) and answers the causal questions the Poisson
control was added for:
  - Is composite's FPR blowup burst-driven, or intrinsic? (compare FPR across arms)
  - Is backlog-drift's T2 non-monotonicity burst-driven, or real size-blindness?
    (compare per-ladder PASS/monotonicity across arms)
"""

from __future__ import annotations

import csv
import os

import yaml

import config

LADDERS = ["L-RATE", "L-SIZE-OUT", "L-SIZE-IN", "L-CAP"]
DETECTORS = ["composite", "threshold", "backlog-drift"]


def _arm_dir(arm: str) -> str:
    return os.path.join(config.EXP_DIR, "results", arm)


def _load_fpr(arm: str) -> dict:
    p = os.path.join(_arm_dir(arm), "fpr.yaml")
    return yaml.safe_load(open(p)) if os.path.exists(p) else {}


def _load_verdicts(arm: str) -> dict:
    """Return {(detector, ladder): (fired_string, first_mult, monotone)}."""
    p = os.path.join(_arm_dir(arm), "ladder-verdicts.csv")
    out: dict = {}
    if not os.path.exists(p):
        return out
    rows = list(csv.DictReader(open(p)))
    from collections import defaultdict
    bykey = defaultdict(list)
    for r in rows:
        bykey[(r["detector"], r["ladder"])].append((float(r["load_mult"]), r["fired"] == "True"))
    for key, seq in bykey.items():
        seq.sort()
        fired_str = "".join("F" if f else "." for _, f in seq)
        first = next((m for m, f in seq if f), None)
        # monotone-from-first-firing
        idx = next((i for i, (_, f) in enumerate(seq) if f), None)
        mono = idx is not None and all(f for _, f in seq[idx:])
        out[key] = (fired_str, first, mono)
    return out


def _load_t4(arm: str) -> dict:
    """Return {detector: (raw_flips_at_w0, lead_ms_at_w0)} from t4-curve.csv."""
    p = os.path.join(_arm_dir(arm), "t4-curve.csv")
    out: dict = {}
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p)):
        if int(r["w_us"]) == 0:
            lead = r["lead_us"]
            lead_ms = round(int(lead) / 1000) if lead not in ("", None) else None
            out[r["detector"]] = (int(r["flips"]), lead_ms)
    return out


def _arm_response_pass(verd, arm, detector, ladder):
    """PASS iff fired-at-all and monotone-from-first-firing."""
    fs, first, mono = verd[arm].get((detector, ladder), ("", None, None))
    return (first is not None) and bool(mono)


def _overall_verdict(fpr, verd, t4, arm, detector, n):
    """The suite verdict for one detector on one arm, as a short string.
    Encodes the §9 rule + the FPR gate."""
    fired = fpr[arm].get(detector, {}).get("fired", 0)
    over_fpr = isinstance(fired, int) and fired > max(1, int(0.05 * n))
    t1 = _arm_response_pass(verd, arm, detector, "L-RATE")
    t2o = _arm_response_pass(verd, arm, detector, "L-SIZE-OUT")
    flips = t4[arm].get(detector, (None, None))[0]
    t4ok = (flips == 0)
    if over_fpr:
        return f"DISQUALIFIED (FPR {fired}/{n})"
    if t1 and t2o and t4ok:
        return "PASS"
    fails = []
    if not t1:
        fails.append("T1")
    if not t2o:
        fails.append("T2-OUT")
    if not t4ok:
        fails.append(f"T4({flips} flips)")
    return "FAIL: " + ",".join(fails)


def write_comparison(arms: list[str]) -> str:
    """Write ONE consolidated results/SUMMARY.md spanning every arm."""
    fpr = {a: _load_fpr(a) for a in arms}
    verd = {a: _load_verdicts(a) for a in arms}
    t4 = {a: _load_t4(a) for a in arms}
    n = fpr.get(arms[0], {}).get("n_runs", 20)

    L = []
    L.append("# Metamorphic Suite — Overall Summary\n")
    L.append(f"BLIS saturation detectors ({', '.join(DETECTORS)}) scored via the "
             "Saturation Bank (`blis run --detectors all`) over the metamorphic "
             "suite (T1–T4). Apparatus: `qwen/qwen3-14b`, trained-physics, vLLM "
             "defaults, single non-shedding instance, μ\\* ≈ 28.7 req/s. "
             "See `docs/CLIFF-PROBE.md` for μ\\*, `README.md` for method.\n")
    L.append(f"**Two arrival arms** (identical workload except the arrival "
             f"process): **gamma** (CV=3, bursty) and **poisson** (CV=1, smooth "
             f"— the burstiness *control*). Per-arm detail lives in "
             f"`results/<arm>/`; this file is the consolidated view.\n")

    # --- Overall verdict table ---
    L.append("## Verdict (per detector × arm)\n")
    L.append("| detector | " + " | ".join(arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for d in DETECTORS:
        cells = [_overall_verdict(fpr, verd, t4, a, d, n) for a in arms]
        L.append(f"| `{d}` | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Verdict rule (§9): PASS iff T1 & T2-OUT & T4(0 flips) pass **and** "
             "FPR ≤ 5%. A detector over the FPR budget is DISQUALIFIED regardless "
             "of response/T4 (its passes are read at an uncontrolled FPR).\n")

    # --- FPR ---
    L.append("## False-positive rate (§3.4)\n")
    L.append("| detector | " + " | ".join(f"{a} (fired/{n})" for a in arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for d in DETECTORS:
        cells = [str(fpr[a].get(d, {}).get("fired", "?")) for a in arms]
        L.append(f"| `{d}` | " + " | ".join(cells) + " |")
    L.append("")

    # --- T4 raw flips + lead ---
    L.append("## T4 temporal (raw, no dwell)\n")
    L.append("| detector | " + " | ".join(f"{a}: flips / lead" for a in arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for d in DETECTORS:
        cells = []
        for a in arms:
            flips, lead = t4[a].get(d, ("?", None))
            lead_s = f"{lead/1000:+.1f}s" if isinstance(lead, (int, float)) else "n/a"
            cells.append(f"{flips} / {lead_s}")
        L.append(f"| `{d}` | " + " | ".join(cells) + " |")
    L.append("")

    # --- Response tests: first-firing (× nominal) + monotonicity, per arm ---
    L.append("## Response tests — first-firing (× nominal) / monotone / pattern\n")
    for d in DETECTORS:
        L.append(f"### `{d}`\n")
        L.append("| ladder | " + " | ".join(arms) + " |")
        L.append("|---|" + "---|" * len(arms))
        for lad in LADDERS:
            cells = []
            for a in arms:
                fs, first, mono = verd[a].get((d, lad), ("", None, None))
                fm = f"{first:.2f}×" if first is not None else "none"
                mo = "y" if mono else ("n" if mono is not None else "?")
                cells.append(f"{fm} / {mo} / `{fs or '?'}`")
            L.append(f"| {lad} | " + " | ".join(cells) + " |")
        L.append("")

    # --- Causal readings (auto-derived from the two arms) ---
    L.append("## Causal readings (what the Poisson control isolates)\n")
    comp = {a: fpr[a].get("composite", {}).get("fired", 0) for a in arms}
    comp_intrinsic = all(isinstance(v, int) and v > n * 0.25 for v in comp.values())
    L.append(f"- **composite false alarms** — FPR {comp}. "
             + ("High in **every** arm ⇒ **intrinsic**, not burst-driven: its "
                "1/√arrivals noise floor shrinks as arrival count grows, so it "
                "trips on healthy traffic regardless of burstiness. Parameterless "
                "⇒ uncorrectable."
                if comp_intrinsic else
                "Differs across arms ⇒ **burst-driven**."))
    bd_t4 = {a: t4[a].get("backlog-drift", (None, None))[0] for a in arms}
    L.append(f"- **backlog-drift T4 flips** — {bd_t4}. "
             + ("Flips under bursty gamma but 0 under smooth Poisson ⇒ the "
                "flip-flop is **burst-driven** (its windowed slope dips negative "
                "during burst lulls; no lulls ⇒ nothing to flip on)."
                if bd_t4.get("gamma", 0) and not bd_t4.get("poisson", 1) else
                "See per-arm detail."))
    # backlog-drift response coverage across arms
    bd_fires_any = {a: any(verd[a].get(("backlog-drift", lad), ("", None, None))[1] is not None
                           for lad in LADDERS) for a in arms}
    if not bd_fires_any.get("poisson", True):
        L.append("- **backlog-drift on smooth traffic** — under Poisson it "
                 "**never fires on any ladder** (fails even T1 at 1.6× overload): "
                 "a gradual in-flight rise keeps its OLS slope below the K=3×noise "
                 "band. It needs *jumpy* traffic to trip — backwards for a "
                 "saturation detector. Exposed only by the two-arm contrast.")
    L.append("")

    # --- Bottom line ---
    L.append("## Bottom line\n")
    L.append("- **`threshold`** is the only detector that passes cleanly, and it "
             "does so in **both** arms — but it is the latest to warn (negative T4 "
             "lead) and depends on hand-calibrating `threshold_ms`.")
    L.append("- **`composite`** has the best instinct (earliest firing, best T4 "
             "lead) but is **disqualified in both arms** by an uncorrectable false-"
             "alarm rate. A tunable threshold could rescue it.")
    L.append("- **`backlog-drift`** fails in both arms for *different* reasons "
             "(bursty: flip-flops; smooth: never fires) — it is miscalibrated for "
             "its own slope signal.")
    L.append("")

    os.makedirs(os.path.join(config.EXP_DIR, "results"), exist_ok=True)
    out = os.path.join(config.EXP_DIR, "results", "SUMMARY.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {out}")
    return out


# Backwards-compatible alias.
write_summary = write_comparison


if __name__ == "__main__":
    import sys
    write_comparison(sys.argv[1:] or list(config._ARMS.keys()))
