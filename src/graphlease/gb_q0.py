from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

PHASES = ("default_pre", "u60", "default_recovery")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "graphlease.gb_q0.v1":
        raise ValueError("unexpected GB-Q0 config schema")
    if config["platform"]["tensor_parallel_size"] != 1:
        raise ValueError("GB-Q0 is TP1 only")
    if config["platform"]["physical_devices_by_restart"] != [0, 1, 2]:
        raise ValueError("GB-Q0 physical-device assignment changed")
    if config["execution"]["fresh_processes"] != 3:
        raise ValueError("GB-Q0 requires exactly three fresh trials")
    if config["execution"]["workload_requests"] != 0:
        raise ValueError("GB-Q0 must issue zero requests")
    sizes = config["portfolios"]["u60"]["capture_sizes"]
    if len(sizes) != 60 or sizes != sorted(set(sizes)):
        raise ValueError("U60 differs from the frozen 60-size union")
    if config["portfolios"]["default"]["capture_sizes"] is not None:
        raise ValueError("default must remain the runtime default")
    markers = config["composite_resource_signature"]["required_substrings"]
    if len(markers) != 5 or config["composite_resource_signature"]["match"] != "all":
        raise ValueError("GB-Q0 requires the five-part all-of classifier")
    return config


def compilation_config(config: dict[str, Any], phase: str) -> dict[str, Any] | None:
    if phase != "u60":
        return None
    sizes = config["portfolios"]["u60"]["capture_sizes"]
    return {
        "mode": 3,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": sizes,
        "max_cudagraph_capture_size": sizes[-1],
    }


def run_filename(phase: str, restart_index: int) -> str:
    return f"910b_tp1_{phase}_r{restart_index}.json"


def composite_failure(run: dict[str, Any], config: dict[str, Any]) -> bool:
    text = "\n".join(str(run.get(name, "")) for name in ("error_type", "error", "traceback"))
    return run.get("status") == "failed" and all(
        marker.lower() in text.lower()
        for marker in config["composite_resource_signature"]["required_substrings"]
    )


def failure_boundary(run: dict[str, Any], config: dict[str, Any]) -> int | None:
    text = "\n".join(str(run.get(name, "")) for name in ("error", "traceback"))
    match = re.search(config["composite_resource_signature"]["boundary_regex"], text)
    return int(match.group(1)) if match else None
