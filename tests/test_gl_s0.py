import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from adjudicate_gl_s0 import adjudicate
from graphlease.gl_s0 import PLATFORMS, PORTFOLIOS, load_config


def run(platform: str, portfolio: str, restart: int, config: dict) -> dict:
    sizes = (
        config["portfolios"][portfolio]["capture_sizes"]
        if portfolio == "coverage_union"
        else list(range(1, 20))
    )
    kv = 100_000 if portfolio == "default" else 99_000
    reserved = 10_000_000_000 if portfolio == "default" else 10_100_000_000
    return {
        "status": "success",
        "process_to_ready_ms": 1000 if portfolio == "default" else 1050,
        "resolved_runtime": {
            "compilation": {"cudagraph_capture_sizes": sizes},
            "cache": {"kv_token_capacity": kv},
        },
        "resource_after_init": [
            {
                "rank": rank,
                "reserved_bytes": reserved,
                "allocated_bytes": reserved - 1000,
            }
            for rank in range(4)
        ],
    }


def matrix(config: dict) -> dict:
    return {
        (platform, portfolio, restart): run(platform, portfolio, restart, config)
        for platform in PLATFORMS
        for portfolio in PORTFOLIOS
        for restart in range(3)
    }


class GLS0Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(REPOSITORY_ROOT / "configs" / "gl_s0.json")

    def test_config_preserves_original_union_and_negative_gate(self):
        self.assertEqual(len(self.config["portfolios"]["coverage_union"]["capture_sizes"]), 60)
        self.assertEqual(self.config["held_out_prediction_basis"]["union_acl_graph_entry_count"], 1796)
        self.assertEqual(self.config["frozen_negative"]["commit"][:7], "9343564")

    def test_cross_runtime_constraint_continues_only_as_graphbudget(self):
        runs = matrix(self.config)
        for restart in (0, 1):
            runs[("910b", "coverage_union", restart)] = {
                "status": "failed",
                "error": "ACL error 207008: stream resources are insufficient",
            }
        decision = adjudicate(runs, self.config)
        self.assertEqual(
            decision["verdict"], "continue_graphbudget_cross_runtime_resource_constraint"
        )
        self.assertFalse(decision["graphlease_controller_restored"])

    def test_both_platform_capacity_cost_continues_scale_gate(self):
        runs = matrix(self.config)
        for platform in PLATFORMS:
            for restart in range(3):
                item = runs[(platform, "coverage_union", restart)]
                item["resolved_runtime"]["cache"]["kv_token_capacity"] = 96_000
        decision = adjudicate(runs, self.config)
        self.assertEqual(decision["verdict"], "continue_scale_conditioned_phase_gate")

    def test_cheap_complete_capture_closes_family(self):
        decision = adjudicate(matrix(self.config), self.config)
        self.assertEqual(decision["verdict"], "stop_complete_capture_low_resource_cost")

    def test_default_failure_is_technical_invalid(self):
        runs = matrix(self.config)
        runs[("a100", "default", 0)] = {"status": "failed", "error": "TP error"}
        decision = adjudicate(runs, self.config)
        self.assertEqual(decision["verdict"], "technical_invalid_default_start_failure")


if __name__ == "__main__":
    unittest.main()
