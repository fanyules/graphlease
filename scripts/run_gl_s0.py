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

from graphlease.gl_s0 import PLATFORMS, PORTFOLIOS, compilation_config, load_config, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one zero-request GL-S0 resource probe")
    parser.add_argument("--platform", choices=PLATFORMS, required=True)
    parser.add_argument("--portfolio", choices=PORTFOLIOS, required=True)
    parser.add_argument("--restart-index", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--physical-devices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "gl_s0.json"
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "gl_s0" / "GL_S0_FREEZE.json",
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

    device = worker.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
    elif device.type == "npu":
        torch.npu.synchronize()
        try:
            free_bytes, total_bytes = torch.npu.mem_get_info()
        except AttributeError:
            total_bytes = torch.npu.get_device_properties(0).total_memory
            free_bytes = total_bytes - torch.npu.memory_reserved()
        allocated = torch.npu.memory_allocated()
        reserved = torch.npu.memory_reserved()
    else:
        raise RuntimeError(f"unsupported worker device: {device}")

    acl_wrapper_count = None
    acl_entry_count = None
    if device.type == "npu":
        from vllm_ascend.compilation.acl_graph import _acl_graph_wrappers

        wrappers = list(_acl_graph_wrappers)
        acl_wrapper_count = len(wrappers)
        acl_entry_count = sum(len(wrapper.concrete_aclgraph_entries) for wrapper in wrappers)
    return {
        "rank": int(getattr(worker, "rank", -1)),
        "device_type": device.type,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(allocated),
        "reserved_bytes": int(reserved),
        "compilation_counter": {
            name: int(getattr(compilation_counter, name))
            for name in dir(compilation_counter)
            if name.startswith("num_") and isinstance(getattr(compilation_counter, name), int)
        },
        "acl_wrapper_count": acl_wrapper_count,
        "acl_graph_entry_count": acl_entry_count,
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
        raise ValueError("active config differs from the public GL-S0 freeze")
    if not commit_is_ancestor(freeze["pre_output_repository_commit"]):
        raise ValueError("GL-S0 pre-output freeze commit is not an ancestor")
    restarts = config["execution"]["fresh_processes_per_platform_portfolio"]
    if not 0 <= args.restart_index < restarts:
        raise ValueError("restart index is outside the frozen matrix")
    expected_devices = config["platforms"][args.platform]["physical_devices"]
    actual_devices = [int(item) for item in args.physical_devices.split(",")]
    if actual_devices != expected_devices:
        raise ValueError("physical device order differs from the frozen matrix")
    model_config = args.model / "config.json"
    if sha256(model_config) != config["model"]["config_sha256"]:
        raise ValueError("model config hash differs from the frozen asset")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    configure_environment()
    config = load_config(args.config)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    validate_run(args, config, freeze)

    import torch
    import vllm
    from vllm import LLM

    runtime_extra: dict[str, Any]
    if args.platform == "910b":
        import torch_npu  # noqa: F401

        device_count = torch.npu.device_count()
        runtime_extra = {
            "torch_npu": package_version("torch-npu"),
            "vllm_ascend": package_version("vllm-ascend"),
        }
    else:
        device_count = torch.cuda.device_count()
        runtime_extra = {"cuda": torch.version.cuda}
    if device_count != 4:
        raise RuntimeError(f"GL-S0 requires exactly four visible devices, found {device_count}")

    requested = compilation_config(config, args.portfolio)
    engine = config["engine"]
    kwargs: dict[str, Any] = {
        "model": str(args.model),
        "tensor_parallel_size": 4,
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
    additional = config["platforms"][args.platform]["additional_config"]
    if additional:
        kwargs["additional_config"] = additional

    payload: dict[str, Any] = {
        "schema": "graphlease.gl_s0.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "repository_commit": repository_commit(),
        "config_sha256": sha256(args.config),
        "freeze_sha256": sha256(args.freeze),
        "model_config_sha256": sha256(args.model / "config.json"),
        "platform": args.platform,
        "portfolio": args.portfolio,
        "restart_index": args.restart_index,
        "physical_devices": args.physical_devices,
        "requested_compilation_config": requested,
        "runtime": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "visible_device_count": device_count,
            **runtime_extra,
        },
    }
    llm = None
    try:
        init_started = time.perf_counter_ns()
        llm = LLM(**kwargs)
        payload["model_init_ms"] = (time.perf_counter_ns() - init_started) / 1_000_000
        payload["process_to_ready_ms"] = (time.perf_counter_ns() - PROCESS_STARTED_NS) / 1_000_000
        payload["resolved_runtime"] = resolved_runtime(llm)
        payload["resource_after_init"] = llm.collective_rpc(worker_resource_snapshot, timeout=120)
        if len(payload["resource_after_init"]) != 4:
            raise RuntimeError("resource snapshot did not return all four TP ranks")
        payload["workload_requests_issued"] = 0
        payload["status"] = "success"
    except BaseException as error:  # retain external runtime failures as Gate evidence
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
        payload["traceback"] = traceback.format_exc()
    finally:
        if llm is not None and hasattr(llm.llm_engine, "shutdown"):
            shutdown_started = time.perf_counter_ns()
            try:
                llm.llm_engine.shutdown()
            except BaseException as error:
                payload["shutdown_error_type"] = type(error).__name__
                payload["shutdown_error"] = str(error)
                if payload["status"] == "success":
                    payload["status"] = "failed"
            payload["shutdown_ms"] = (time.perf_counter_ns() - shutdown_started) / 1_000_000
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
            "schema": "graphlease.gl_s0.run.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "repository_commit": repository_commit(),
            "platform": args.platform,
            "portfolio": args.portfolio,
            "restart_index": args.restart_index,
            "physical_devices": args.physical_devices,
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
