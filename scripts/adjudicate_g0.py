#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graphlease.contracts import FORMAL_PORTFOLIOS, PHASES, load_config, sha256
from graphlease.metrics import percentile

GRAPH_PORTFOLIOS = ("default", "small_dense", "large_dense")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate GraphLease Gate 0")
    parser.add_argument("--stage", choices=("preflight", "formal"), required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "g0.json"
    )
    return parser.parse_args()


def repository_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_runs(paths: list[Path]) -> dict[tuple[str, str, int], dict[str, Any]]:
    runs = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (
            payload.get("platform"),
            payload.get("portfolio"),
            int(payload.get("restart_index", -1)),
        )
        if key in runs:
            raise ValueError(f"duplicate run key: {key}")
        payload["_source_path"] = str(path)
        payload["_source_sha256"] = sha256(path)
        runs[key] = payload
    return runs


def bootstrap_lower(
    values: list[float], *, resamples: int, seed: int, statistic: Callable[[list[float]], float] = statistics.mean
) -> float:
    if not values:
        raise ValueError("bootstrap requires values")
    generator = random.Random(seed)
    samples = []
    for _ in range(resamples):
        draw = [values[generator.randrange(len(values))] for _ in values]
        samples.append(statistic(draw))
    return percentile(samples, 0.025)


def phase_result(run: dict[str, Any], phase: str) -> dict[str, Any]:
    return next(item for item in run["phase_results"] if item["phase"] == phase)


def resource_row(run: dict[str, Any]) -> dict[str, Any]:
    rows = run.get("resource_after_run") or run.get("resource_after_init")
    if not rows:
        raise ValueError("run omitted resource snapshot")
    return rows[0]


def adjudicate_preflight(
    runs: dict[tuple[str, str, int], dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    expected = {
        (platform, portfolio, 0)
        for platform in config["platforms"]
        for portfolio in config["preflight"]["portfolios"]
    }
    if set(runs) != expected:
        raise ValueError("preflight input matrix is incomplete or oversized")
    platforms = {}
    all_pass = True
    for platform in config["platforms"]:
        eager = runs[(platform, "eager", 0)]
        eager_digests = [item["output_digest"] for item in eager.get("preflight_cases", [])]
        rows = []
        for portfolio in config["preflight"]["portfolios"]:
            run = runs[(platform, portfolio, 0)]
            cases = run.get("preflight_cases", [])
            digests = [item.get("output_digest") for item in cases]
            dispatch_observed = bool(cases) and all(
                item.get("dispatch", {}).get("dispatch_rows") for item in cases
            )
            kv_capacity = run.get("resolved_runtime", {}).get("cache", {}).get(
                "kv_token_capacity", 0
            )
            checks = {
                "status_success": run.get("status") == "success",
                "case_count": len(cases) == len(config["preflight"]["wave_cases"]),
                "exact_plan_invariance": digests == eager_digests,
                "effective_plan_recorded": bool(
                    run.get("resolved_runtime", {}).get("compilation")
                ),
                "dispatch_stats_recorded": dispatch_observed,
                "kv_capacity_recorded": isinstance(kv_capacity, int)
                and kv_capacity > 0,
            }
            passed = all(checks.values())
            all_pass &= passed
            rows.append(
                {
                    "portfolio": portfolio,
                    "checks": checks,
                    "passed": passed,
                    "output_digests": digests,
                    "effective_compilation": run.get("resolved_runtime", {}).get(
                        "compilation"
                    ),
                    "kv_token_capacity": kv_capacity,
                }
            )
        platforms[platform] = rows
    return {
        "stage": "G0-P",
        "verdict": "pass_unblock_formal" if all_pass else "stop_plan_invariance_or_observability",
        "passed": all_pass,
        "platforms": platforms,
    }


def formal_matrix_checks(
    runs: dict[tuple[str, str, int], dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    restarts = config["execution"]["fresh_processes_per_formal_portfolio"]
    expected_formal = {
        (platform, portfolio, restart)
        for platform in config["platforms"]
        for portfolio in FORMAL_PORTFOLIOS
        for restart in range(restarts)
    }
    expected_resource = {(platform, "coverage_union", 0) for platform in config["platforms"]}
    if set(runs) != expected_formal | expected_resource:
        raise ValueError("formal input matrix is incomplete or oversized")
    checks = {}
    passed = True
    for platform in config["platforms"]:
        platform_rows = []
        for restart in range(restarts):
            eager = runs[(platform, "eager", restart)]
            eager_digest = {
                phase: phase_result(eager, phase)["formal"]["output_digest"]
                for phase in PHASES
            }
            for portfolio in FORMAL_PORTFOLIOS:
                run = runs[(platform, portfolio, restart)]
                phase_rows = []
                for phase in PHASES:
                    item = phase_result(run, phase) if run.get("status") == "success" else None
                    phase_checks = {
                        "exact_eager_output": bool(item)
                        and item["formal"]["output_digest"] == eager_digest[phase],
                        "instrumented_matches_formal": bool(item)
                        and item.get("instrumented_matches_formal") is True,
                        "dispatch_recorded": bool(item)
                        and bool(item["instrumented"].get("dispatch_rows")),
                    }
                    phase_rows.append({"phase": phase, "checks": phase_checks})
                row_checks = {
                    "status_success": run.get("status") == "success",
                    "engine_idle": run.get("engine_idle_after_run") is True,
                    "phases": all(all(value["checks"].values()) for value in phase_rows),
                }
                row_pass = all(row_checks.values())
                passed &= row_pass
                platform_rows.append(
                    {
                        "portfolio": portfolio,
                        "restart_index": restart,
                        "checks": row_checks,
                        "phase_checks": phase_rows,
                        "passed": row_pass,
                    }
                )
        checks[platform] = platform_rows
    return checks, passed


def question_one(
    runs: dict[tuple[str, str, int], dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    restarts = config["execution"]["fresh_processes_per_formal_portfolio"]
    rule = config["statistics"]["phase_winner"]
    result = {}
    all_platforms = True
    for platform_index, platform in enumerate(config["platforms"]):
        phase_rows = {}
        winners = []
        for phase_index, phase in enumerate(PHASES):
            median_goodput = {
                portfolio: statistics.median(
                    phase_result(runs[(platform, portfolio, restart)], phase)["formal"]
                    ["correct_token_goodput_per_s"]
                    for restart in range(restarts)
                )
                for portfolio in GRAPH_PORTFOLIOS
            }
            ordered = sorted(
                GRAPH_PORTFOLIOS,
                key=lambda portfolio: (median_goodput[portfolio], portfolio),
                reverse=True,
            )
            winner, runner_up = ordered[:2]
            winners.append(winner)
            goodput_gain = []
            p99_reduction = []
            paired_wins = 0
            for restart in range(restarts):
                winner_row = phase_result(runs[(platform, winner, restart)], phase)["formal"]
                other_row = phase_result(runs[(platform, runner_up, restart)], phase)["formal"]
                gain = winner_row["correct_token_goodput_per_s"] / other_row[
                    "correct_token_goodput_per_s"
                ] - 1.0
                reduction = 1.0 - winner_row["request_metrics"]["e2e_p99_ms"] / other_row[
                    "request_metrics"
                ]["e2e_p99_ms"]
                goodput_gain.append(gain)
                p99_reduction.append(reduction)
                paired_wins += gain > 0
            seed = config["statistics"]["bootstrap_seed"] + platform_index * 100 + phase_index
            goodput_l95 = bootstrap_lower(
                goodput_gain,
                resamples=config["statistics"]["bootstrap_resamples"],
                seed=seed,
            )
            p99_l95 = bootstrap_lower(
                p99_reduction,
                resamples=config["statistics"]["bootstrap_resamples"],
                seed=seed + 1,
            )
            stable = paired_wins >= rule["paired_restart_wins_min"] and (
                goodput_l95 >= rule["relative_goodput_lower95_min"]
                or p99_l95 >= rule["or_p99_latency_reduction_lower95_min"]
            )
            phase_rows[phase] = {
                "winner": winner,
                "runner_up": runner_up,
                "median_goodput": median_goodput,
                "paired_wins": paired_wins,
                "goodput_gain_by_restart": goodput_gain,
                "goodput_gain_lower95": goodput_l95,
                "p99_reduction_by_restart": p99_reduction,
                "p99_reduction_lower95": p99_l95,
                "stable": stable,
            }
        platform_pass = len(set(winners)) == 2 and all(
            value["stable"] for value in phase_rows.values()
        )
        all_platforms &= platform_pass
        result[platform] = {
            "phase_results": phase_rows,
            "different_stable_winners": platform_pass,
        }
    return {"passed": all_platforms, "platforms": result}


def question_two(
    runs: dict[tuple[str, str, int], dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    rule = config["statistics"]["resource_scarcity"]
    restarts = config["execution"]["fresh_processes_per_formal_portfolio"]
    result = {}
    all_platforms = True
    for platform in config["platforms"]:
        default_runs = [runs[(platform, "default", restart)] for restart in range(restarts)]
        default_kv = statistics.median(
            run["resolved_runtime"]["cache"]["kv_token_capacity"] for run in default_runs
        )
        default_reserved = statistics.median(
            resource_row(run)["reserved_bytes"] for run in default_runs
        )
        union = runs[(platform, "coverage_union", 0)]
        error_text = f"{union.get('error', '')}\n{union.get('traceback', '')}".lower()
        confirmed_stream_error = (
            platform == "910b"
            and (
                "stream-resource exhaustion" in error_text
                or "stream resources are insufficient" in error_text
                or "insufficient_stream_resources" in error_text
            )
        )
        if union.get("status") == "success":
            union_kv = union["resolved_runtime"]["cache"]["kv_token_capacity"]
            union_reserved = resource_row(union)["reserved_bytes"]
            effective = set(
                union["resolved_runtime"]["compilation"]["cudagraph_capture_sizes"]
            )
            requested = set(config["portfolios"]["coverage_union"]["capture_sizes"])
            filtered_count = len(requested - effective)
            kv_loss = (default_kv - union_kv) / default_kv
            memory_increase = union_reserved - default_reserved
        else:
            union_kv = None
            union_reserved = None
            filtered_count = 0
            kv_loss = None
            memory_increase = None
        passed = confirmed_stream_error or (
            union.get("status") == "success"
            and (
                kv_loss >= rule["kv_capacity_loss_fraction_min"]
                or memory_increase >= rule["resident_memory_increase_bytes_min"]
                or (platform == "910b" and filtered_count > 0)
            )
        )
        all_platforms &= passed
        result[platform] = {
            "passed": passed,
            "union_status": union.get("status"),
            "default_kv_token_capacity": default_kv,
            "union_kv_token_capacity": union_kv,
            "kv_capacity_loss_fraction": kv_loss,
            "default_reserved_bytes": default_reserved,
            "union_reserved_bytes": union_reserved,
            "resident_memory_increase_bytes": memory_increase,
            "filtered_requested_sizes": filtered_count,
            "confirmed_stream_resource_error": confirmed_stream_error,
        }
    return {"passed": all_platforms, "platforms": result}


def normalized_static_score(
    phase_goodput: dict[str, dict[str, float]], portfolio: str
) -> float:
    ratios = []
    for phase in PHASES:
        oracle = max(phase_goodput[phase].values())
        ratios.append(phase_goodput[phase][portfolio] / oracle)
    return len(ratios) / sum(1.0 / value for value in ratios)


def question_three(
    runs: dict[tuple[str, str, int], dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    restarts = config["execution"]["fresh_processes_per_formal_portfolio"]
    rule = config["statistics"]["static_oracle_gap"]
    result = {}
    all_platforms = True
    black_box_near = True
    heldout = config["statistics"]["black_box_stop"]["heldout_restart_index"]
    for platform_index, platform in enumerate(config["platforms"]):
        static_scores_by_restart: list[dict[str, float]] = []
        for restart in range(restarts):
            phase_goodput = {
                phase: {
                    portfolio: phase_result(
                        runs[(platform, portfolio, restart)], phase
                    )["formal"]["correct_token_goodput_per_s"]
                    for portfolio in FORMAL_PORTFOLIOS
                }
                for phase in PHASES
            }
            static_scores_by_restart.append(
                {
                    portfolio: normalized_static_score(phase_goodput, portfolio)
                    for portfolio in FORMAL_PORTFOLIOS
                }
            )
        best_static = max(
            FORMAL_PORTFOLIOS,
            key=lambda portfolio: statistics.median(
                row[portfolio] for row in static_scores_by_restart
            ),
        )
        goodput_gaps = [1.0 - row[best_static] for row in static_scores_by_restart]
        goodput_l95 = bootstrap_lower(
            goodput_gaps,
            resamples=config["statistics"]["bootstrap_resamples"],
            seed=config["statistics"]["bootstrap_seed"] + 1000 + platform_index,
        )

        p99_gaps = []
        for restart in range(restarts):
            ratios = []
            for phase in PHASES:
                values = {
                    portfolio: phase_result(
                        runs[(platform, portfolio, restart)], phase
                    )["formal"]["request_metrics"]["e2e_p99_ms"]
                    for portfolio in FORMAL_PORTFOLIOS
                }
                ratios.append(values[best_static] / min(values.values()) - 1.0)
            p99_gaps.append(statistics.mean(ratios))
        p99_l95 = bootstrap_lower(
            p99_gaps,
            resamples=config["statistics"]["bootstrap_resamples"],
            seed=config["statistics"]["bootstrap_seed"] + 1100 + platform_index,
        )
        passed = (
            goodput_l95 >= rule["correct_token_goodput_gap_lower95_min"]
            or p99_l95 >= rule["or_p99_latency_gap_lower95_min"]
        )
        all_platforms &= passed

        training_restarts = [index for index in range(restarts) if index != heldout]
        trained_mapping = {}
        for phase in PHASES:
            trained_mapping[phase] = max(
                FORMAL_PORTFOLIOS,
                key=lambda portfolio: statistics.mean(
                    phase_result(runs[(platform, portfolio, restart)], phase)["formal"]
                    ["correct_token_goodput_per_s"]
                    for restart in training_restarts
                ),
            )
        heldout_ratios = []
        for phase in PHASES:
            values = {
                portfolio: phase_result(runs[(platform, portfolio, heldout)], phase)[
                    "formal"
                ]["correct_token_goodput_per_s"]
                for portfolio in FORMAL_PORTFOLIOS
            }
            heldout_ratios.append(values[trained_mapping[phase]] / max(values.values()))
        black_box_score = len(heldout_ratios) / sum(1.0 / value for value in heldout_ratios)
        black_box_regret = 1.0 - black_box_score
        near = black_box_regret <= config["statistics"]["black_box_stop"][
            "maximum_oracle_goodput_regret"
        ]
        black_box_near &= near
        result[platform] = {
            "passed": passed,
            "best_static": best_static,
            "static_score_by_restart": static_scores_by_restart,
            "goodput_gap_by_restart": goodput_gaps,
            "goodput_gap_lower95": goodput_l95,
            "p99_gap_by_restart": p99_gaps,
            "p99_gap_lower95": p99_l95,
            "black_box_mapping": trained_mapping,
            "black_box_heldout_regret": black_box_regret,
            "black_box_within_five_percent": near,
        }
    return {
        "passed": all_platforms,
        "platforms": result,
        "black_box_within_five_percent_on_both": black_box_near,
    }


def adjudicate_formal(
    runs: dict[tuple[str, str, int], dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    matrix_checks, matrix_passed = formal_matrix_checks(runs, config)
    if not matrix_passed:
        return {
            "stage": "G0-M",
            "verdict": "stop_correctness_or_execution_failure",
            "passed": False,
            "matrix_checks": matrix_checks,
        }
    q1 = question_one(runs, config)
    q2 = question_two(runs, config)
    q3 = question_three(runs, config)
    if not q1["passed"]:
        verdict = "stop_no_stable_phase_dependent_optimum"
    elif not q2["passed"]:
        verdict = "stop_no_cross_platform_portfolio_scarcity"
    elif not q3["passed"]:
        verdict = "stop_best_static_closes_oracle_gap"
    elif q3["black_box_within_five_percent_on_both"]:
        verdict = "stop_black_box_latency_mapping_near_oracle"
    else:
        verdict = "pass_first_three_unblock_lifecycle_gate"
    return {
        "stage": "G0-M",
        "verdict": verdict,
        "passed": verdict == "pass_first_three_unblock_lifecycle_gate",
        "matrix_checks": matrix_checks,
        "questions": {"q1": q1, "q2": q2, "q3": q3, "q4": "not_run_conditional"},
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = load_config(args.config)
    runs = load_runs(args.inputs)
    decision = (
        adjudicate_preflight(runs, config)
        if args.stage == "preflight"
        else adjudicate_formal(runs, config)
    )
    decision.update(
        {
            "schema": "graphlease.g0.decision.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "config_sha256": sha256(args.config),
            "source_files": [
                {"path": str(path), "sha256": sha256(path)} for path in args.inputs
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": decision["stage"], "verdict": decision["verdict"]}))
    return 0 if decision.get("passed") else 4


if __name__ == "__main__":
    raise SystemExit(main())
