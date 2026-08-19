"""Rung layout for the four ladders (§3.2).

A ladder is a sweep along exactly one axis from obviously-idle to
obviously-overloaded, every other axis held fixed. Rungs are expressed as an
*offered-load multiple* of the nominal cliff mu* (§3.3), geometrically spaced
0.3x -> 1.6x, denser near 1.0x. The same multiples are reused across all four
ladders so their sharpness numbers (§4.4) are on one scale; each ladder then
maps a multiple onto its own axis knob.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import config


def rung_multiples() -> list[float]:
    """12 offered-load multiples of mu*, geometric with densification near 1.0x.

    We geometrically space in log-space between LADDER_LOW and LADDER_HIGH, then
    warp toward DENSE_CENTER so more rungs land near the cliff (§3.2: "denser
    near 1.0x", so 0.95x vs 0.4x firing is distinguishable).
    """
    n = config.RUNG_COUNT
    lo, hi, c = config.LADDER_LOW, config.LADDER_HIGH, config.DENSE_CENTER
    # Base geometric grid in [0,1].
    base = [i / (n - 1) for i in range(n)]
    # Warp: pull points toward the fraction of the range where mu* sits, using a
    # symmetric power warp around the center's normalized position.
    c_frac = (math.log(c) - math.log(lo)) / (math.log(hi) - math.log(lo))
    warped = []
    for t in base:
        # Ease toward c_frac: denser where |t - c_frac| is small. Exponent >1 on
        # |d| pulls interior points toward the center.
        d = t - c_frac
        w = c_frac + math.copysign(abs(d) ** 1.5, d)
        warped.append(w)
    # The warp shrinks the span; rescale so the endpoints land exactly on
    # LADDER_LOW and LADDER_HIGH (§3.2 wants the full 0.3x-1.6x range with
    # densification only in the middle, not endpoints pulled inward).
    wmin, wmax = min(warped), max(warped)
    warped = [(w - wmin) / (wmax - wmin) for w in warped]
    warped = sorted(set(round(w, 6) for w in warped))
    # Guarantee n distinct rungs; if densification collapsed any, fall back to
    # the plain geometric grid.
    if len(warped) < n:
        warped = base
    mults = [math.exp(math.log(lo) + w * (math.log(hi) - math.log(lo))) for w in warped]
    return [round(m, 4) for m in mults[:n]]


@dataclass(frozen=True)
class Rung:
    ladder: str          # L-RATE | L-SIZE-OUT | L-SIZE-IN | L-CAP
    index: int           # 0..11
    load_mult: float     # offered-load multiple of mu* (the sharpness x-axis)
    # Per-ladder knob values (only the named axis varies):
    rate_rps: float
    input_tokens: int
    output_tokens: int
    max_num_seqs: int
    label: str           # human tag, e.g. "L-RATE r03 x0.30"


# For the size ladders we hold the arrival rate at a comfortably feasible base
# (§T3 build rule: fix workload at ~0.7x the healthy cliff, then vary the axis).
SIZE_LADDER_BASE_MULT = 0.7   # rate held here for L-SIZE-*; the *size* carries load
CAP_LADDER_BASE_MULT = 0.7    # workload fixed here for L-CAP; capacity shrinks
# L-CAP shrinks B geometrically from HEALTHY_B down to this floor. Probed at
# vLLM defaults: the fixed 0.7x workload is healthy at B>=96 and firmly
# overloaded by B~64, so a floor of 40 guarantees the top rungs are super-cliff.
CAP_B_FLOOR = 40


def build_ladder(ladder: str) -> list[Rung]:
    mults = rung_multiples()
    rungs: list[Rung] = []
    for i, m in enumerate(mults):
        if ladder == "L-RATE":
            # Vary arrivals/s directly; size + capacity fixed.
            rungs.append(Rung(
                ladder, i, m,
                rate_rps=round(m * config.MU_STAR, 4),
                input_tokens=config.E_IN,
                output_tokens=config.E_OUT,
                max_num_seqs=config.HEALTHY_B,
                label=f"L-RATE r{i:02d} x{m:.2f}",
            ))
        elif ladder == "L-SIZE-OUT":
            # Rate fixed at base; scale output tokens so offered decode-work
            # spans the same load multiples. At base rate the load multiple is
            # carried by E[O]: out = E_OUT * (m / base).
            base = SIZE_LADDER_BASE_MULT
            out = max(1, round(config.E_OUT * (m / base)))
            rungs.append(Rung(
                ladder, i, m,
                rate_rps=round(base * config.MU_STAR, 4),
                input_tokens=config.E_IN,
                output_tokens=out,
                max_num_seqs=config.HEALTHY_B,
                label=f"L-SIZE-OUT r{i:02d} x{m:.2f} out={out}",
            ))
        elif ladder == "L-SIZE-IN":
            # Rate + output fixed; scale input (prompt) tokens.
            base = SIZE_LADDER_BASE_MULT
            inp = max(1, round(config.E_IN * (m / base)))
            rungs.append(Rung(
                ladder, i, m,
                rate_rps=round(base * config.MU_STAR, 4),
                input_tokens=inp,
                output_tokens=config.E_OUT,
                max_num_seqs=config.HEALTHY_B,
                label=f"L-SIZE-IN r{i:02d} x{m:.2f} in={inp}",
            ))
        elif ladder == "L-CAP":
            # Workload byte-identical across rungs (fixed at CAP base = 0.7x the
            # HEALTHY cliff); capacity shrinks by lowering B. At vLLM defaults the
            # cliff falls SUBLINEARLY as B drops and the fixed workload stays
            # healthy until B ~= 96, saturating by B ~= 64 (probed). So we sweep B
            # GEOMETRICALLY from the anchor (HEALTHY_B) down past the knee to
            # CAP_B_FLOOR, guaranteeing the ladder spans healthy -> overloaded.
            base = CAP_LADDER_BASE_MULT
            frac = i / (config.RUNG_COUNT - 1)
            b = round(config.HEALTHY_B * (CAP_B_FLOOR / config.HEALTHY_B) ** frac)
            b = max(CAP_B_FLOOR, min(config.HEALTHY_B, b))
            # Nominal capacity-reduction scaffold (monotone as B shrinks); the
            # true L-CAP sharpness is the workload as a multiple of the rung's OWN
            # measured cliff (mu*_rung = r_dec,rung/E[O]), computed at scoring
            # time from per-rung throughput (§4.4).
            nominal_mult = round(base * config.HEALTHY_B / b, 3)
            rungs.append(Rung(
                ladder, i, nominal_mult,
                rate_rps=round(base * config.MU_STAR, 4),
                input_tokens=config.E_IN,
                output_tokens=config.E_OUT,
                max_num_seqs=b,
                label=f"L-CAP r{i:02d} B={b} (~x{nominal_mult:.2f})",
            ))
        else:
            raise ValueError(f"unknown ladder {ladder}")
    return rungs


LADDERS = ["L-RATE", "L-SIZE-OUT", "L-SIZE-IN", "L-CAP"]


if __name__ == "__main__":
    for lad in LADDERS:
        print(f"== {lad} ==")
        for r in build_ladder(lad):
            print(f"  {r.label:32s} rate={r.rate_rps:6.3f} in={r.input_tokens:5d} "
                  f"out={r.output_tokens:5d} B={r.max_num_seqs}")
