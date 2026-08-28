#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graphlease.gl_s0 import PLATFORMS, PORTFOLIOS, load_config, run_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate the frozen GL-S0 matrix")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "gl_s0.json"
    )
    return parser.parse_args()


def load_runs(results_dir: Path, config: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    runs = {}
    restarts = config["execution"]["fresh_processes_per_platform_portfolio"]
    for platform in PLATFORMS:
        for portfolio in PORTFOLIOS:
            for restart in range(restarts):
                path = results_dir / run_filename(platform, portfolio, restart)
                if path.is_file():
                    runs[(platform, portfolio, restart)] = json.loads(path.read_text(encoding="utf-8"))
                else:
                    runs[(platform, portfolio, restart)] = {
                        "status": "missing",
                        "error": f"missing result: {path.name}",
                    }
    return runs


def successful(run: dict[str, Any]) -> bool:
    return run.get("status") == "success" and len(run.get("resource_after_init", [])) == 4


def effective_sizes(run: dict[str, Any]) -> list[int]:
    return list(
        run.get("resolved_runtime", {})
        .get("compilation", {})
        .get("cudagraph_capture_sizes", [])
    )


def kv_capacity(run: dict[str, Any]) -> int:
    return int(run["resolved_runtime"]["cache"]["kv_token_capacity"])


def rank_values(run: dict[str, Any], field: str) -> dict[int, int]:
    return {int(row["rank"]): int(row[field]) for row in run["resource_after_init"]}


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def platform_summary(
    platform: str,
    runs: dict[tuple[str, str, int], dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    restarts = config["execution"]["fresh_processes_per_platform_portfolio"]
    requested_union = config["portfolios"]["coverage_union"]["capture_sizes"]
    defaults = [runs[(platform, "default", index)] for index in range(restarts)]
    unions = [runs[(platform, "coverage_union", index)] for index in range(restarts)]
    union_successes = [successful(run) for run in unions]
    filtering = [
        successful(run) and effective_sizes(run) != requested_union for run in unions
    ]
    markers = [item.lower() for item in config["thresholds"]["confirmed_ascend_resource_markers"]]
    resource_failures = []
    for run in unions:
        text = "\n".join(str(run.get(name, "")) for name in ("error_type", "error", "traceback")).lower()
        resource_failures.append(not successful(run) and any(marker in text for marker in markers))

    def snapshot_values(items: list[dict[str, Any]], field: str) -> list[int]:
        return [
            int(row[field])
            for item in items
            if successful(item)
            for row in item["resource_after_init"]
            if row.get(field) is not None
        ]

    default_wrappers = snapshot_values(defaults, "acl_wrapper_count")
    union_wrappers = snapshot_values(unions, "acl_wrapper_count")
    default_entries = snapshot_values(defaults, "acl_graph_entry_count")
    union_entries = snapshot_values(unions, "acl_graph_entry_count")
    unregistered_failures = [
        not successful(run) and not resource_failure
        for run, resource_failure in zip(unions, resource_failures, strict=True)
    ]

    summary: dict[str, Any] = {
        "default_success_count": sum(successful(run) for run in defaults),
        "union_success_count": sum(union_successes),
        "union_exact_capture_count": sum(
            success and effective_sizes(run) == requested_union
            for success, run in zip(union_successes, unions, strict=True)
        ),
        "union_filtering_count": sum(filtering),
        "union_confirmed_resource_failure_count": sum(resource_failures),
        "union_unregistered_failure_count": sum(unregistered_failures),
        "union_resource_constraint_count": sum(
            filtered or failed for filtered, failed in zip(filtering, resource_failures, strict=True)
        ),
        "run_statuses": {
            "default": [run.get("status") for run in defaults],
            "coverage_union": [run.get("status") for run in unions],
        },
        "default_effective_capture_size_counts": [
            len(effective_sizes(run)) if successful(run) else None for run in defaults
        ],
        "union_effective_capture_size_counts": [
            len(effective_sizes(run)) if successful(run) else None for run in unions
        ],
        "default_acl_wrapper_counts": default_wrappers,
        "union_acl_wrapper_counts": union_wrappers,
        "default_acl_graph_entry_counts": default_entries,
        "union_acl_graph_entry_counts": union_entries,
    }
    if not all(successful(run) for run in defaults + unions):
        return summary

    kv_losses = [
        (kv_capacity(default) - kv_capacity(union)) / kv_capacity(default)
        for default, union in zip(defaults, unions, strict=True)
    ]
    reserved_deltas: list[float] = []
    allocated_deltas: list[float] = []
    for default, union in zip(defaults, unions, strict=True):
        default_reserved = rank_values(default, "reserved_bytes")
        union_reserved = rank_values(union, "reserved_bytes")
        default_allocated = rank_values(default, "allocated_bytes")
        union_allocated = rank_values(union, "allocated_bytes")
        if set(default_reserved) != set(union_reserved) or len(default_reserved) != 4:
            raise ValueError(f"rank mismatch in {platform} resource snapshots")
        reserved_deltas.extend(
            union_reserved[rank] - default_reserved[rank] for rank in sorted(default_reserved)
        )
        allocated_deltas.extend(
            union_allocated[rank] - default_allocated[rank] for rank in sorted(default_allocated)
        )
    ready_deltas = [
        (float(union["process_to_ready_ms"]) - float(default["process_to_ready_ms"]))
        / float(default["process_to_ready_ms"])
        for default, union in zip(defaults, unions, strict=True)
    ]
    summary.update(
        {
            "paired_kv_capacity_loss_fractions": kv_losses,
            "median_kv_capacity_loss_fraction": median(kv_losses),
            "median_per_rank_reserved_memory_increase_bytes": median(reserved_deltas),
            "median_per_rank_allocated_memory_increase_bytes_diagnostic": median(allocated_deltas),
            "paired_process_to_ready_increase_fractions": ready_deltas,
            "median_process_to_ready_increase_fraction": median(ready_deltas),
        }
    )
    threshold = config["thresholds"]
    summary["scale_stressed"] = (
        summary["median_kv_capacity_loss_fraction"]
        >= threshold["kv_capacity_loss_fraction_min"]
        or summary["median_per_rank_reserved_memory_increase_bytes"]
        >= threshold["reserved_memory_increase_bytes_min"]
    )
    return summary


def adjudicate(
    runs: dict[tuple[str, str, int], dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    summaries = {platform: platform_summary(platform, runs, config) for platform in PLATFORMS}
    restarts = config["execution"]["fresh_processes_per_platform_portfolio"]
    if any(summary["default_success_count"] != restarts for summary in summaries.values()):
        verdict = "technical_invalid_default_start_failure"
        continue_research = False
    else:
        minimum = config["thresholds"]["resource_constraint_restarts_min"]
        cross_runtime = (
            summaries["a100"]["union_success_count"] == restarts
            and summaries["a100"]["union_exact_capture_count"] == restarts
            and summaries["910b"]["union_resource_constraint_count"] >= minimum
            and summaries["910b"]["union_unregistered_failure_count"] == 0
        )
        all_union_success = all(
            summary["union_success_count"] == restarts for summary in summaries.values()
        )
        all_union_exact = all(
            summary["union_exact_capture_count"] == restarts for summary in summaries.values()
        )
        if cross_runtime:
            verdict = "continue_graphbudget_cross_runtime_resource_constraint"
            continue_research = True
        elif all_union_success and all_union_exact and all(
            summary.get("scale_stressed", False) for summary in summaries.values()
        ):
            verdict = "continue_scale_conditioned_phase_gate"
            continue_research = True
        elif all_union_success and all_union_exact:
            one_sided = any(summary.get("scale_stressed", False) for summary in summaries.values())
            startup_only = any(
                summary.get("median_process_to_ready_increase_fraction", 0.0)
                >= config["thresholds"]["startup_only_relative_increase_min"]
                for summary in summaries.values()
            )
            if one_sided:
                verdict = "stop_one_sided_capacity_cost_not_sufficient"
            elif startup_only:
                verdict = "stop_startup_only_no_capacity_or_coverage_constraint"
            else:
                verdict = "stop_complete_capture_low_resource_cost"
            continue_research = False
        else:
            verdict = "technical_invalid_or_unregistered_union_failure"
            continue_research = False
    return {
        "schema": "graphlease.gl_s0.decision.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate": "GL-S0",
        "old_g0m_status_unchanged": True,
        "verdict": verdict,
        "continue_graph_resource_research": continue_research,
        "graphlease_controller_restored": False,
        "platforms": summaries,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = load_config(args.config)
    decision = adjudicate(load_runs(args.results_dir, config), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": decision["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
