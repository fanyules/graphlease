from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FORMAL_PORTFOLIOS = ("eager", "default", "small_dense", "large_dense")
PHASES = ("low_short", "high_long")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_capture_sizes(name: str, sizes: list[int]) -> None:
    if not sizes or sizes != sorted(set(sizes)):
        raise ValueError(f"{name} capture sizes must be nonempty, sorted, and unique")
    if sizes[0] <= 0:
        raise ValueError(f"{name} capture sizes must be positive")


def validate_config(config: dict[str, Any]) -> None:
    portfolios = config["portfolios"]
    if tuple(name for name, value in portfolios.items() if value.get("formal")) != FORMAL_PORTFOLIOS:
        raise ValueError("formal portfolio order differs from the frozen contract")
    for name, value in portfolios.items():
        sizes = value["capture_sizes"]
        if sizes is not None and sizes:
            validate_capture_sizes(name, sizes)
    small = portfolios["small_dense"]["capture_sizes"]
    large = portfolios["large_dense"]["capture_sizes"]
    union = portfolios["coverage_union"]["capture_sizes"]
    if union != sorted(set(small) | set(large)):
        raise ValueError("coverage_union must be the exact specialized union")
    if abs(len(small) - len(large)) > 1:
        raise ValueError("specialized portfolios must have matched graph counts")

    phases = config["workload"]["phases"]
    if tuple(phases) != PHASES:
        raise ValueError("phase order differs from the frozen contract")
    max_num_seqs = config["engine"]["max_num_seqs"]
    for phase_name, phase in phases.items():
        waves = phase["wave_concurrency"]
        if not waves or any(not 1 <= value <= max_num_seqs for value in waves):
            raise ValueError(f"invalid wave concurrency for {phase_name}")
        if phase["instrumented_wave_concurrency"] != waves:
            raise ValueError("instrumented replay must use the formal wave trace")
        if phase["prompt_tokens"] + phase["output_tokens"] > config["engine"]["max_model_len"]:
            raise ValueError(f"phase exceeds max_model_len: {phase_name}")
    orders = config["workload"]["phase_order_by_restart"]
    expected = config["execution"]["fresh_processes_per_formal_portfolio"]
    if len(orders) != expected or any(sorted(order) != sorted(PHASES) for order in orders):
        raise ValueError("each restart must contain both phases exactly once")


def compilation_config(config: dict[str, Any], portfolio: str) -> dict[str, Any] | None:
    value = config["portfolios"][portfolio]
    if value["enforce_eager"]:
        return {"mode": "none", "cudagraph_mode": "NONE"}
    sizes = value["capture_sizes"]
    if sizes is None:
        return None
    return {
        "mode": 3,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": sizes,
        "max_cudagraph_capture_size": sizes[-1],
    }


def phase_order(config: dict[str, Any], restart_index: int) -> list[str]:
    return list(config["workload"]["phase_order_by_restart"][restart_index])
