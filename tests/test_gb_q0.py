import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from adjudicate_gb_q0 import adjudicate
from graphlease.gb_q0 import composite_failure, load_config, run_filename


def stack_audit(config: dict) -> dict:
    stack = config["official_stack"]
    return {
        "driver_version": stack["driver_version"],
        "driver_package_version": stack["driver_package_version"],
        "firmware_version": stack["firmware_version"],
        "firmware_package_version": stack["firmware_package_version"],
        "cann_version": stack["cann_version"],
        "torch_version": stack["torch_version"],
        "torch_npu_version": stack["torch_npu_version"],
        "vllm_version": stack["vllm_version"],
        "vllm_ascend_version": stack["vllm_ascend_version"],
    }


def success(phase: str, restart: int, config: dict) -> dict:
    sizes = (
        config["portfolios"]["u60"]["capture_sizes"]
        if phase == "u60"
        else [1, 2, 4, 8]
    )
    return {
        "status": "success",
        "workload_requests_issued": 0,
        "stack_audit": stack_audit(config),
        "stack_mismatches": [],
        "resolved_runtime": {
            "compilation": {"cudagraph_capture_sizes": sizes},
            "cache": {"kv_token_capacity": 100_000},
        },
        "resource_after_init": [
            {"rank": 0, "acl_wrapper_count": 42, "acl_graph_entry_count": len(sizes) * 42}
        ],
    }


def resource_failure(restart: int, config: dict, stream_id: int = 1984) -> dict:
    text = (
        "AclmdlRICaptureBegin error code 207005; Alloc sq cq fail, "
        f"stream_id={stream_id}, retCode=0x7020023; resource alloc fail"
    )
    return {
        "status": "failed",
        "error": text,
        "traceback": text,
        "stack_audit": stack_audit(config),
        "stack_mismatches": [],
    }


def write_run(directory: Path, phase: str, restart: int, payload: dict) -> None:
    (directory / run_filename(phase, restart)).write_text(
        json.dumps(payload), encoding="utf-8"
    )


class GBQ0Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(REPOSITORY_ROOT / "configs" / "gb_q0.json")

    def test_contract_is_tp1_u60_and_preserves_gl_s0(self):
        self.assertEqual(self.config["platform"]["tensor_parallel_size"], 1)
        self.assertEqual(len(self.config["portfolios"]["u60"]["capture_sizes"]), 60)
        self.assertTrue(self.config["prior_evidence"]["must_remain_unmodified"])

    def test_classifier_requires_all_five_markers(self):
        run = resource_failure(0, self.config)
        self.assertTrue(composite_failure(run, self.config))
        run["error"] = run["error"].replace("0x7020023", "")
        run["traceback"] = ""
        self.assertFalse(composite_failure(run, self.config))

    def test_new_stack_u60_success_stops_graphbudget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for restart in range(3):
                write_run(root, "default_pre", restart, success("default_pre", restart, self.config))
                write_run(root, "u60", restart, success("u60", restart, self.config))
            decision = adjudicate(root, self.config)
        self.assertEqual(decision["verdict"], "stop_graphbudget_old_stack_limitation")
        self.assertFalse(decision["continue_to_gb_g0o_design"])

    def test_stable_failure_and_equivalent_recovery_confirms_cliff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for restart in range(3):
                write_run(root, "default_pre", restart, success("default_pre", restart, self.config))
                write_run(root, "u60", restart, resource_failure(restart, self.config))
                write_run(
                    root,
                    "default_recovery",
                    restart,
                    success("default_recovery", restart, self.config),
                )
            decision = adjudicate(root, self.config)
        self.assertEqual(decision["verdict"], "confirm_cross_runtime_resource_cliff")

    def test_boundary_change_stops_for_instability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for restart in range(3):
                write_run(root, "default_pre", restart, success("default_pre", restart, self.config))
                write_run(root, "u60", restart, resource_failure(restart, self.config, 1984 + restart))
                write_run(
                    root,
                    "default_recovery",
                    restart,
                    success("default_recovery", restart, self.config),
                )
            decision = adjudicate(root, self.config)
        self.assertEqual(
            decision["verdict"], "stop_runtime_instability_failure_boundary_changed"
        )


if __name__ == "__main__":
    unittest.main()
