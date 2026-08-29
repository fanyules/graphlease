#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROCESS_STARTED_NS = time.perf_counter_ns()
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graphlease.gb_q0 import PHASES, compilation_config, load_config, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one GB-Q0 TP1 engine process")
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--restart-index", type=int, required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "gb_q0.json"
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "gb_q0" / "GB_Q0_FREEZE.json",
    )
    return parser.parse_args()


def repository_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit_is_ancestor(commit: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
    ).returncode == 0


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def version_info(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def find_cann_install_info() -> Path:
    root = Path("/usr/local/Ascend")
    matches = sorted(root.glob("**/ascend_toolkit_install.info"))
    if not matches:
        raise FileNotFoundError("ascend_toolkit_install.info is absent")
    return matches[0]


def stack_audit() -> dict[str, Any]:
    driver_path = Path("/usr/local/Ascend/driver/version.info")
    firmware_path = Path("/usr/local/Ascend/firmware/version.info")
    cann_path = find_cann_install_info()
    driver = version_info(driver_path)
    firmware = version_info(firmware_path)
    cann = version_info(cann_path)
    return {
        "driver_version_path": str(driver_path),
        "driver_version": driver.get("Version"),
        "driver_package_version": driver.get("package_version"),
        "driver_inner_version": driver.get("Innerversion"),
        "firmware_version_path": str(firmware_path),
        "firmware_version": firmware.get("Version"),
        "firmware_package_version": firmware.get("package_version"),
        "cann_install_info_path": str(cann_path),
        "cann_version": cann.get("version"),
        "torch_version": package_version("torch"),
        "torch_npu_version": package_version("torch-npu"),
        "vllm_version": package_version("vllm"),
        "vllm_ascend_version": package_version("vllm-ascend"),
    }


def normalized(value: str | None) -> str | None:
    return value.split("+", 1)[0] if value else None


def stack_mismatches(audit: dict[str, Any], config: dict[str, Any]) -> list[str]:
    expected = config["official_stack"]
    comparisons = {
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
    return [
        f"{name}: expected {wanted}, observed {audit.get(name)}"
        for name, wanted in comparisons.items()
        if normalized(audit.get(name)) != wanted
    ]


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name") and hasattr(value, "value"):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def worker_resource_snapshot(worker: Any) -> dict[str, Any]:
    import torch
    from vllm.compilation.counter import compilation_counter
    from vllm_ascend.compilation.acl_graph import _acl_graph_wrappers

    device = worker.device
    if device.type != "npu":
        raise RuntimeError(f"GB-Q0 expected NPU worker, observed {device}")
    torch.npu.synchronize()
    try:
        free_bytes, total_bytes = torch.npu.mem_get_info()
    except AttributeError:
        total_bytes = torch.npu.get_device_properties(0).total_memory
        free_bytes = total_bytes - torch.npu.memory_reserved()
    wrappers = list(_acl_graph_wrappers)
    return {
        "rank": int(getattr(worker, "rank", -1)),
        "device_type": device.type,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.npu.memory_allocated()),
        "reserved_bytes": int(torch.npu.memory_reserved()),
        "acl_wrapper_count": len(wrappers),
        "acl_graph_entry_count": sum(
            len(wrapper.concrete_aclgraph_entries) for wrapper in wrappers
        ),
        "compilation_counter": {
            name: int(getattr(compilation_counter, name))
            for name in dir(compilation_counter)
            if name.startswith("num_") and isinstance(getattr(compilation_counter, name), int)
        },
    }


def resolved_runtime(llm: Any) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config
    compilation = config.compilation_config
    cache = config.cache_config
    num_blocks = int(cache.num_gpu_blocks or 0)
    block_size = int(cache.block_size)
    return {
        "compilation": {
            "mode": str(compilation.mode),
            "backend": compilation.backend,
            "cudagraph_mode": str(compilation.cudagraph_mode),
            "cudagraph_capture_sizes": list(compilation.cudagraph_capture_sizes or ()),
            "max_cudagraph_capture_size": compilation.max_cudagraph_capture_size,
        },
        "parallel": {
            "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
            "disable_custom_all_reduce": config.parallel_config.disable_custom_all_reduce,
        },
        "scheduler": {
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        },
        "cache": {
            "num_gpu_blocks": num_blocks,
            "block_size": block_size,
            "kv_token_capacity": num_blocks * block_size,
            "gpu_memory_utilization": cache.gpu_memory_utilization,
        },
        "attention_backend": str(config.attention_config.backend),
    }


def configure_environment() -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def validate_run(args: argparse.Namespace, config: dict[str, Any], freeze: dict[str, Any]) -> None:
    if freeze.get("status") != "frozen" or freeze.get("config_sha256") != sha256(args.config):
        raise ValueError("active config differs from the public GB-Q0 freeze")
    if not commit_is_ancestor(freeze["pre_output_repository_commit"]):
        raise ValueError("GB-Q0 pre-output commit is not an ancestor")
    devices = config["platform"]["physical_devices_by_restart"]
    if not 0 <= args.restart_index < len(devices):
        raise ValueError("restart index is outside the frozen matrix")
    if args.physical_device != devices[args.restart_index]:
        raise ValueError("physical-device assignment differs from the frozen matrix")
    model_config = args.model / "config.json"
    if sha256(model_config) != config["model"]["config_sha256"]:
        raise ValueError("model config hash differs from the frozen asset")
    prior = REPOSITORY_ROOT / config["prior_evidence"]["gl_s0_frozen_decision_path"]
    if sha256(prior) != config["prior_evidence"]["gl_s0_frozen_decision_sha256"]:
        raise ValueError("the frozen GL-S0 decision was modified")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    configure_environment()
    config = load_config(args.config)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    validate_run(args, config, freeze)
    audit = stack_audit()
    requested = compilation_config(config, args.phase)
    payload: dict[str, Any] = {
        "schema": "graphlease.gb_q0.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "repository_commit": repository_commit(),
        "config_sha256": sha256(args.config),
        "freeze_sha256": sha256(args.freeze),
        "model_config_sha256": sha256(args.model / "config.json"),
        "phase": args.phase,
        "restart_index": args.restart_index,
        "physical_device": args.physical_device,
        "requested_compilation_config": requested,
        "stack_audit": audit,
        "stack_mismatches": stack_mismatches(audit, config),
    }
    llm = None
    try:
        if payload["stack_mismatches"]:
            raise RuntimeError("official stack mismatch: " + "; ".join(payload["stack_mismatches"]))

        import torch
        import torch_npu  # noqa: F401
        import vllm
        from vllm import LLM

        if torch.npu.device_count() != 1:
            raise RuntimeError(f"GB-Q0 requires one visible NPU, found {torch.npu.device_count()}")
        payload["runtime"] = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "visible_device_count": torch.npu.device_count(),
        }
        engine = config["engine"]
        kwargs: dict[str, Any] = {
            "model": str(args.model),
            "tensor_parallel_size": 1,
            "dtype": config["model"]["dtype"],
            "seed": engine["seed"] + args.restart_index,
            "max_model_len": engine["max_model_len"],
            "max_num_seqs": engine["max_num_seqs"],
            "max_num_batched_tokens": engine["max_num_batched_tokens"],
            "gpu_memory_utilization": engine["gpu_memory_utilization"],
            "enable_prefix_caching": engine["enable_prefix_caching"],
            "async_scheduling": engine["async_scheduling"],
            "disable_log_stats": True,
            "cudagraph_metrics": False,
            "enforce_eager": False,
            "disable_custom_all_reduce": False,
        }
        if requested is not None:
            kwargs["compilation_config"] = requested
        init_started = time.perf_counter_ns()
        llm = LLM(**kwargs)
        payload["model_init_ms"] = (time.perf_counter_ns() - init_started) / 1_000_000
        payload["process_to_ready_ms"] = (time.perf_counter_ns() - PROCESS_STARTED_NS) / 1_000_000
        payload["resolved_runtime"] = resolved_runtime(llm)
        payload["resource_after_init"] = llm.collective_rpc(worker_resource_snapshot, timeout=120)
        if len(payload["resource_after_init"]) != 1:
            raise RuntimeError("resource snapshot did not return the TP1 rank")
        payload["workload_requests_issued"] = 0
        payload["status"] = "success"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
        payload["traceback"] = traceback.format_exc()
    finally:
        if llm is not None and hasattr(llm.llm_engine, "shutdown"):
            started = time.perf_counter_ns()
            try:
                llm.llm_engine.shutdown()
            except BaseException as error:
                payload["shutdown_error_type"] = type(error).__name__
                payload["shutdown_error"] = str(error)
                payload["status"] = "failed"
            payload["shutdown_ms"] = (time.perf_counter_ns() - started) / 1_000_000
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        payload["total_process_ms"] = (time.perf_counter_ns() - PROCESS_STARTED_NS) / 1_000_000
    return jsonable(payload)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        payload = execute(args)
    except BaseException as error:
        payload = {
            "schema": "graphlease.gb_q0.run.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "repository_commit": repository_commit(),
            "phase": args.phase,
            "restart_index": args.restart_index,
            "physical_device": args.physical_device,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
