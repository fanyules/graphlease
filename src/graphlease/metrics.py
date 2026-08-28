from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def output_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def summarize_request_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("request metrics cannot be empty")
    ttft = [row["ttft_ms"] for row in rows]
    tpot = [row["tpot_ms"] for row in rows]
    e2e = [row["e2e_ms"] for row in rows]
    return {
        "ttft_median_ms": statistics.median(ttft),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "tpot_median_ms": statistics.median(tpot),
        "tpot_p99_ms": percentile(tpot, 0.99),
        "e2e_median_ms": statistics.median(e2e),
        "e2e_p99_ms": percentile(e2e, 0.99),
    }
