"""Load blis metrics JSON and saturation reports; split traces per detector."""

from __future__ import annotations

import json
from dataclasses import dataclass


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@dataclass
class TraceRecord:
    timestamp: int          # µs
    detector: str
    level: str              # STABLE | BACKLOGGED | OVERLOADED
    score: float
    confidence: float
    signals: dict


@dataclass
class Report:
    final: dict                        # detector -> level
    per_detector: dict                 # detector -> [TraceRecord] (time-sorted)


def load_report(path: str) -> Report:
    with open(path) as f:
        raw = json.load(f)
    final = raw.get("final", {})
    per: dict[str, list[TraceRecord]] = {}
    for rec in raw.get("trace", []):
        det = rec["detector"]
        res = rec["result"]
        per.setdefault(det, []).append(TraceRecord(
            timestamp=rec["timestamp"],
            detector=det,
            level=res["level"],
            score=res.get("score", 0.0),
            confidence=res.get("confidence", 0.0),
            signals=res.get("signals", {}),
        ))
    for det in per:
        per[det].sort(key=lambda r: r.timestamp)
    return Report(final=final, per_detector=per)
