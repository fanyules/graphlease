from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from adjudicate_g0 import adjudicate_formal, adjudicate_preflight
from graphlease.contracts import FORMAL_PORTFOLIOS, PHASES, load_config


def preflight_run(platform: str, portfolio: str) -> dict:
    return {
        "status": "success",
        "preflight_cases": [
            {
                "output_digest": f"case-{index}",
                "dispatch": {"dispatch_rows": [{"runtime_mode": "NONE"}]},
            }
            for index in range(2)
        ],
        "resolved_runtime": {
            "compilation": {"cudagraph_mode": portfolio},
            "cache": {"kv_token_capacity": 10000},
        },
    }


def formal_run(platform: str, portfolio: str, restart: int, config: dict) -> dict:
    values = {
        "low_short": {
            "eager": 70.0,
            "default": 90.0,
            "small_dense": 140.0,
            "large_dense": 90.0,
        },
        "high_long": {
            "eager": 70.0,
            "default": 90.0,
            "small_dense": 90.0,
            "large_dense": 140.0,
        },
    }
    phase_results = []
    for phase in PHASES:
        goodput = values[phase][portfolio]
        phase_results.append(
            {
                "phase": phase,
                "formal": {
                    "output_digest": phase,
                    "correct_token_goodput_per_s": goodput,
                    "request_metrics": {"e2e_p99_ms": 1000.0 / goodput},
                },
                "instrumented": {"dispatch_rows": [{"runtime_mode": "FULL"}]},
                "instrumented_matches_formal": True,
            }
        )
    return {
        "status": "success",
        "engine_idle_after_run": True,
        "phase_results": phase_results,
        "resolved_runtime": {
            "cache": {"kv_token_capacity": 10000},
            "compilation": {"cudagraph_capture_sizes": []},
        },
        "resource_after_run": [{"reserved_bytes": 1_000_000_000}],
    }


class AdjudicationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(REPOSITORY_ROOT / "configs" / "g0.json")

    def test_preflight_requires_all_plans_to_match_eager(self):
        runs = {
            (platform, portfolio, 0): preflight_run(platform, portfolio)
            for platform in self.config["platforms"]
            for portfolio in self.config["preflight"]["portfolios"]
        }
        self.assertTrue(adjudicate_preflight(runs, self.config)["passed"])
        runs[("910b", "large_dense", 0)]["preflight_cases"][0]["output_digest"] = "mismatch"
        self.assertFalse(adjudicate_preflight(runs, self.config)["passed"])

    def test_formal_positive_mechanism_still_stops_for_simple_black_box(self):
        runs = {}
        for platform in self.config["platforms"]:
            for portfolio in FORMAL_PORTFOLIOS:
                for restart in range(3):
                    runs[(platform, portfolio, restart)] = formal_run(
                        platform, portfolio, restart, self.config
                    )
            runs[(platform, "coverage_union", 0)] = {
                "status": "success",
                "resolved_runtime": {
                    "cache": {"kv_token_capacity": 8000},
                    "compilation": {
                        "cudagraph_capture_sizes": self.config["portfolios"]
                        ["coverage_union"]["capture_sizes"]
                    },
                },
                "resource_after_run": [{"reserved_bytes": 2_000_000_000}],
            }
        decision = adjudicate_formal(runs, self.config)
        self.assertEqual(decision["verdict"], "stop_black_box_latency_mapping_near_oracle")


if __name__ == "__main__":
    unittest.main()
