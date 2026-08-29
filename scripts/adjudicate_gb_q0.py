#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graphlease.gb_q0 import (
    composite_failure,
    failure_boundary,
    load_config,
    run_filename,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate the frozen GB-Q0 matrix")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "gb_q0.json"
    )
    return parser.parse_args()


def load_run(results_dir: Path, phase: str, restart: int) -> dict[str, Any]:
    path = results_dir / run_filename(phase, restart)
    if not path.is_file():
        return {"status": "missing", "error": f"missing result: {path.name}"}
    return json.loads(path.read_text(encoding="utf-8"))


def successful(run: dict[str, Any]) -> bool:
    return (
        run.get("status") == "success"
        and run.get("workload_requests_issued") == 0
        and len(run.get("resource_after_init", [])) == 1
        and not run.get("stack_mismatches")
    )


def effective_sizes(run: dict[str, Any]) -> list[int]:
    return list(
        run.get("resolved_runtime", {})
        .get("compilation", {})
        .get("cudagraph_capture_sizes", [])
    )


def default_fingerprint(run: dict[str, Any]) -> dict[str, Any] | None:
    if not successful(run):
        return None
    resource = run["resource_after_init"][0]
    return {
        "effective_capture_sizes": effective_sizes(run),
        "acl_wrapper_count": resource["acl_wrapper_count"],
        "acl_graph_entry_count": resource["acl_graph_entry_count"],
        "kv_token_capacity": run["resolved_runtime"]["cache"]["kv_token_capacity"],
    }


def stack_valid(run: dict[str, Any], config: dict[str, Any]) -> bool:
    audit = run.get("stack_audit")
    if not audit or run.get("stack_mismatches"):
        return False
    expected = config["official_stack"]
    values = {
        "driver_version": expected["driver_version"],
        "driver_package_version": expected["driver_package_version"],
        "firmware_version": expected["firmware_version"],
        "firmware_package_version": expected["firmware_package_version"],
        "cann_version": expected["cann_version"],
        "torch_version": expected["torch_version"],
        "torch_npu_version": expected["torch_npu_version"],
        "vllm_version": expected["vllm_version"],
        "vllm_ascend_version": expected["vllm_ascend_version"],
    }
    return all(str(audit.get(name, "")).split("+", 1)[0] == value for name, value in values.items())


def adjudicate(results_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    restarts = config["execution"]["fresh_processes"]
    defaults = [load_run(results_dir, "default_pre", index) for index in range(restarts)]
    unions = [load_run(results_dir, "u60", index) for index in range(restarts)]
    recoveries = [load_run(results_dir, "default_recovery", index) for index in range(restarts)]
    all_present_runs = [
        run
        for run in defaults + unions + recoveries
        if run.get("status") != "missing"
    ]
    stack_ok = bool(all_present_runs) and all(stack_valid(run, config) for run in all_present_runs)
    default_success_count = sum(successful(run) for run in defaults)
    u60_success_count = sum(successful(run) for run in unions)
    requested_u60 = config["portfolios"]["u60"]["capture_sizes"]
    u60_exact_count = sum(
        successful(run) and effective_sizes(run) == requested_u60 for run in unions
    )
    composite_count = sum(composite_failure(run, config) for run in unions)
    boundaries = [failure_boundary(run, config) for run in unions]
    recovery_required = [run.get("status") != "success" for run in unions]
    recovery_success = [
        not required or successful(recovery)
        for required, recovery in zip(recovery_required, recoveries, strict=True)
    ]
    recovery_equivalent = []
    for required, default, recovery in zip(
        recovery_required, defaults, recoveries, strict=True
    ):
        if not required:
            recovery_equivalent.append(True)
            continue
        before = default_fingerprint(default)
        after = default_fingerprint(recovery)
        if before is None or after is None:
            recovery_equivalent.append(False)
            continue
        graph_equal = all(
            before[name] == after[name]
            for name in (
                "effective_capture_sizes",
                "acl_wrapper_count",
                "acl_graph_entry_count",
            )
        )
        kv_floor = before["kv_token_capacity"] * (
            1.0 - config["recovery"]["kv_capacity_loss_fraction_max"]
        )
        recovery_equivalent.append(graph_equal and after["kv_token_capacity"] >= kv_floor)

    if not stack_ok:
        verdict = "technical_invalid_stack_mismatch_or_missing_audit"
        continue_research = False
    elif default_success_count != restarts:
        verdict = "technical_invalid_default_start_failure"
        continue_research = False
    elif u60_success_count == restarts:
        if u60_exact_count == restarts:
            verdict = "stop_graphbudget_old_stack_limitation"
        else:
            verdict = "technical_invalid_u60_filtered_or_not_exact"
        continue_research = False
    elif 0 < u60_success_count < restarts:
        verdict = "stop_runtime_instability_mixed_u60_outcomes"
        continue_research = False
    elif composite_count != restarts:
        verdict = "technical_invalid_unregistered_u60_failure"
        continue_research = False
    elif len(set(boundaries)) != 1 or boundaries[0] is None:
        verdict = "stop_runtime_instability_failure_boundary_changed"
        continue_research = False
    elif not all(recovery_success) or not all(recovery_equivalent):
        verdict = "stop_runtime_instability_default_did_not_recover_equivalently"
        continue_research = False
    else:
        verdict = "confirm_cross_runtime_resource_cliff"
        continue_research = True

    return {
        "schema": "graphlease.gb_q0.decision.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate": "GB-Q0",
        "gl_s0_frozen_decision_unchanged": True,
        "verdict": verdict,
        "continue_to_gb_g0o_design": continue_research,
        "graphbudget_repository_unblocked": False,
        "counts": {
            "stack_valid_runs": sum(stack_valid(run, config) for run in all_present_runs),
            "present_runs": len(all_present_runs),
            "default_pre_success": default_success_count,
            "u60_success": u60_success_count,
            "u60_exact_capture": u60_exact_count,
            "u60_composite_resource_failure": composite_count,
            "default_recovery_success": sum(
                required and success
                for required, success in zip(recovery_required, recovery_success, strict=True)
            ),
            "default_recovery_equivalent": sum(
                required and equivalent
                for required, equivalent in zip(recovery_required, recovery_equivalent, strict=True)
            ),
        },
        "u60_failure_boundaries": boundaries,
        "default_pre_fingerprints": [default_fingerprint(run) for run in defaults],
        "default_recovery_fingerprints": [default_fingerprint(run) for run in recoveries],
        "statuses": {
            "default_pre": [run.get("status") for run in defaults],
            "u60": [run.get("status") for run in unions],
            "default_recovery": [run.get("status") for run in recoveries],
        },
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = load_config(args.config)
    decision = adjudicate(args.results_dir, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": decision["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
