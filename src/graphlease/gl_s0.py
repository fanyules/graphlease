from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PLATFORMS = ("a100", "910b")
PORTFOLIOS = ("default", "coverage_union")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "graphlease.gl_s0.v1":
        raise ValueError("unexpected GL-S0 config schema")
    if tuple(config["platforms"]) != PLATFORMS:
        raise ValueError("GL-S0 requires the frozen A100/910B platform order")
    if tuple(config["portfolios"]) != PORTFOLIOS:
        raise ValueError("GL-S0 requires only default and coverage_union")
    if any(config["platforms"][name]["tensor_parallel_size"] != 4 for name in PLATFORMS):
        raise ValueError("GL-S0 is TP4 only")
    union = config["portfolios"]["coverage_union"]["capture_sizes"]
    if len(union) != 60 or union != sorted(set(union)):
        raise ValueError("coverage_union must be the original 60 sorted unique sizes")
    if config["portfolios"]["default"]["capture_sizes"] is not None:
        raise ValueError("default must remain the runtime default")
    if config["execution"]["fresh_processes_per_platform_portfolio"] != 3:
        raise ValueError("GL-S0 requires exactly three fresh processes")
    if config["execution"]["workload_requests"] != 0:
        raise ValueError("GL-S0 is resource-only and must issue zero requests")
    if any(
        config["execution"][name]
        for name in ("controller_enabled", "router_enabled", "network_hot_path_enabled")
    ):
        raise ValueError("GL-S0 forbids controller, router, and network hot path")
    return config


def compilation_config(config: dict[str, Any], portfolio: str) -> dict[str, Any] | None:
    sizes = config["portfolios"][portfolio]["capture_sizes"]
    if sizes is None:
        return None
    return {
        "mode": 3,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": sizes,
        "max_cudagraph_capture_size": sizes[-1],
    }


def run_filename(platform: str, portfolio: str, restart_index: int) -> str:
    return f"{platform}_{portfolio}_r{restart_index}.json"
