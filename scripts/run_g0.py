#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROCESS_STARTED_NS = time.perf_counter_ns()
REQUEST_COUNTER = 0
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graphlease.contracts import (
    FORMAL_PORTFOLIOS,
    compilation_config,
    load_config,
    phase_order,
    sha256,
)
from graphlease.metrics import output_digest, percentile, summarize_request_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one GraphLease Gate-0 process")
    parser.add_argument("--mode", choices=("preflight", "formal", "resource"), required=True)
    parser.add_argument("--platform", choices=("a100", "910b"), required=True)
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--restart-index", type=int, default=0)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--physical-device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "g0.json"
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "g0" / "G0_FREEZE.json",
    )
    return parser.parse_args()


def repository_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def freeze_commit_is_ancestor(commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            check=False,
        ).returncode
        == 0
    )


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


def configure_environment() -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def validate_run(args: argparse.Namespace, config: dict[str, Any], freeze: dict[str, Any]) -> None:
    if freeze.get("status") != "frozen" or freeze.get("config_sha256") != sha256(args.config):
        raise ValueError("active config differs from the public freeze")
    if not freeze_commit_is_ancestor(freeze["pre_output_repository_commit"]):
        raise ValueError("public freeze commit is not an ancestor of the running checkout")
    if args.portfolio not in config["portfolios"]:
        raise ValueError("unknown portfolio")
    if args.mode == "preflight":
        if args.portfolio not in config["preflight"]["portfolios"] or args.restart_index != 0:
            raise ValueError("invalid preflight portfolio or restart index")
    elif args.mode == "formal":
        restarts = config["execution"]["fresh_processes_per_formal_portfolio"]
        if args.portfolio not in FORMAL_PORTFOLIOS or not 0 <= args.restart_index < restarts:
            raise ValueError("invalid formal portfolio or restart index")
    elif args.portfolio != "coverage_union" or args.restart_index != 0:
        raise ValueError("resource mode is reserved for the frozen coverage union")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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
        acl_entry_count = sum(
            len(wrapper.concrete_aclgraph_entries) for wrapper in wrappers
        )
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
            if name.startswith("num_")
            and isinstance(getattr(compilation_counter, name), int)
        },
        "acl_wrapper_count": acl_wrapper_count,
        "acl_graph_entry_count": acl_entry_count,
    }


def worker_start_dispatch_trace(worker: Any) -> dict[str, Any]:
    model_runner = worker.model_runner
    marker = "_graphlease_dispatch_trace_state"
    if hasattr(worker, marker):
        raise RuntimeError("worker dispatch trace is already active")
    rows: list[dict[str, Any]] = []
    method_name = "_determine_batch_execution_and_padding"
    if hasattr(model_runner, method_name):
        original = getattr(model_runner, method_name)

        def traced_v1_dispatch(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            graph_stat = result[4]
            if graph_stat is not None:
                unpadded = int(graph_stat.num_unpadded_tokens)
                padded = int(graph_stat.num_padded_tokens)
                runtime_mode = str(graph_stat.runtime_mode)
            else:
                value = kwargs.get("num_tokens", args[0] if args else None)
                if value is None:
                    raise RuntimeError("dispatch trace cannot resolve tokens")
                unpadded = int(value)
                padded = int(result[1].num_tokens)
                runtime_mode = str(result[0])
            rows.append(
                {
                    "num_unpadded_tokens": unpadded,
                    "num_padded_tokens": padded,
                    "num_paddings": padded - unpadded,
                    "runtime_mode": runtime_mode,
                }
            )
            return result

        setattr(model_runner, method_name, traced_v1_dispatch)
        state = {
            "kind": "runner_method",
            "target": model_runner,
            "name": method_name,
            "original": original,
            "rows": rows,
        }
    else:
        module = sys.modules[type(model_runner).__module__]
        function_name = "dispatch_cg_and_sync_dp"
        original = getattr(module, function_name)

        def traced_v2_dispatch(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            value = kwargs.get("num_toks", args[2] if len(args) > 2 else None)
            if value is None:
                raise RuntimeError("dispatch trace cannot resolve V2 tokens")
            unpadded = int(value)
            batch_descriptor = result[0]
            padded = int(batch_descriptor.num_tokens)
            rows.append(
                {
                    "num_unpadded_tokens": unpadded,
                    "num_padded_tokens": padded,
                    "num_paddings": padded - unpadded,
                    "runtime_mode": str(batch_descriptor.cg_mode),
                }
            )
            return result

        setattr(module, function_name, traced_v2_dispatch)
        state = {
            "kind": "module_function",
            "target": module,
            "name": function_name,
            "original": original,
            "rows": rows,
        }
    setattr(worker, marker, state)
    return {"rank": int(getattr(worker, "rank", -1)), "source": state["kind"]}


def worker_finish_dispatch_trace(worker: Any) -> dict[str, Any]:
    marker = "_graphlease_dispatch_trace_state"
    state = getattr(worker, marker, None)
    rows = list(state["rows"]) if state is not None else []
    if state is not None:
        setattr(state["target"], state["name"], state["original"])
        delattr(worker, marker)
    return {
        "rank": int(getattr(worker, "rank", -1)),
        "source": state["kind"] if state is not None else None,
        "rows": rows,
    }


class DispatchTrace:
    def __init__(self, llm: Any):
        self.llm = llm
        self.scheduler_rows: list[Any] = []
        self._engine_core: Any | None = None
        self._original_get_output: Callable[[], Any] | None = None
        self._worker_active = False

    def start(self) -> None:
        if self._worker_active:
            raise RuntimeError("dispatch trace is already active")
        self.llm.collective_rpc(worker_start_dispatch_trace, timeout=60)
        self._worker_active = True
        engine_core = self.llm.llm_engine.engine_core
        original = engine_core.get_output

        def traced_get_output() -> Any:
            output = original()
            stats = getattr(output, "scheduler_stats", None)
            if stats is not None:
                self.scheduler_rows.append(jsonable(stats))
            return output

        engine_core.get_output = traced_get_output
        self._engine_core = engine_core
        self._original_get_output = original

    def finish(self) -> dict[str, Any]:
        if not self._worker_active:
            raise RuntimeError("dispatch trace is not active")
        if self._engine_core is not None and self._original_get_output is not None:
            self._engine_core.get_output = self._original_get_output
        worker_rows = self.llm.collective_rpc(worker_finish_dispatch_trace, timeout=60)
        self._worker_active = False
        self._engine_core = None
        self._original_get_output = None
        rank_rows = [item["rows"] for item in worker_rows]
        consistent = bool(rank_rows) and all(rows == rank_rows[0] for rows in rank_rows[1:])
        rows = rank_rows[0] if rank_rows else []
        modes = Counter(row["runtime_mode"] for row in rows)
        total = len(rows)
        return {
            "worker_dispatch": worker_rows,
            "dispatch_consistent": consistent,
            "dispatch_rows": rows,
            "dispatch_mode_counts": dict(sorted(modes.items())),
            "dispatch_mode_fraction": {
                mode: count / total for mode, count in sorted(modes.items())
            }
            if total
            else {},
            "total_unpadded_tokens": sum(row["num_unpadded_tokens"] for row in rows),
            "total_padded_tokens": sum(row["num_padded_tokens"] for row in rows),
            "total_padding_tokens": sum(row["num_paddings"] for row in rows),
            "scheduler_stats": self.scheduler_rows,
        }

    def stop(self) -> None:
        if self._engine_core is not None and self._original_get_output is not None:
            self._engine_core.get_output = self._original_get_output
        self._engine_core = None
        self._original_get_output = None
        if self._worker_active:
            self.llm.collective_rpc(worker_finish_dispatch_trace, timeout=60)
            self._worker_active = False


def token_pool(llm: Any, pattern: str) -> list[int]:
    values = list(llm.get_tokenizer().encode(pattern, add_special_tokens=False))
    if not values:
        raise RuntimeError("prompt pattern produced no tokens")
    return values


def prompts(concurrency: int, prompt_tokens: int, pool: list[int]) -> list[dict[str, list[int]]]:
    row = [pool[index % len(pool)] for index in range(prompt_tokens)]
    return [{"prompt_token_ids": list(row)} for _ in range(concurrency)]


def sampling_params(output_tokens: int, seed: int) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    value = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=output_tokens,
        detokenize=False,
        seed=seed,
    )
    value.output_kind = RequestOutputKind.CUMULATIVE
    return value


def run_wave(
    llm: Any,
    *,
    concurrency: int,
    prompt_tokens: int,
    output_tokens: int,
    pool: list[int],
    seed: int,
    wave_index: int,
) -> dict[str, Any]:
    global REQUEST_COUNTER

    if llm.llm_engine.has_unfinished_requests():
        raise RuntimeError("wave started while the engine was not idle")
    request_ids = []
    for position, prompt in enumerate(prompts(concurrency, prompt_tokens, pool)):
        request_id = f"graphlease-{REQUEST_COUNTER}"
        REQUEST_COUNTER += 1
        request_ids.append(
            llm.llm_engine.add_request(
                request_id,
                prompt,
                sampling_params(output_tokens, seed + position),
            )
        )
    positions = {str(request_id): index for index, request_id in enumerate(request_ids)}
    first_token_ns: dict[str, int] = {}
    finished_ns: dict[str, int] = {}
    tokens_by_id: dict[str, list[int]] = {}
    step_ms: list[float] = []
    started_ns = time.perf_counter_ns()
    while llm.llm_engine.has_unfinished_requests():
        step_started = time.perf_counter_ns()
        outputs = llm.llm_engine.step()
        now_ns = time.perf_counter_ns()
        step_ms.append((now_ns - step_started) / 1_000_000)
        for output in outputs:
            request_id = str(output.request_id)
            if request_id not in positions:
                raise RuntimeError(
                    f"engine returned request {request_id!r}; expected one of "
                    f"{sorted(positions)!r}"
                )
            if not output.outputs:
                continue
            sequence = output.outputs[0]
            token_ids = [int(value) for value in sequence.token_ids]
            if token_ids and request_id not in first_token_ns:
                first_token_ns[request_id] = now_ns
            if output.finished:
                finished_ns[request_id] = now_ns
                tokens_by_id[request_id] = token_ids
    ended_ns = time.perf_counter_ns()
    if set(first_token_ns) != set(positions) or set(finished_ns) != set(positions):
        raise RuntimeError("wave did not observe first and final output for every request")
    request_metrics = []
    output_rows = []
    for request_id, position in sorted(positions.items(), key=lambda item: item[1]):
        tokens = tokens_by_id[request_id]
        if len(tokens) != output_tokens:
            raise RuntimeError("ignore_eos wave produced the wrong output length")
        first = first_token_ns[request_id]
        finished = finished_ns[request_id]
        request_metrics.append(
            {
                "wave_index": wave_index,
                "position": position,
                "ttft_ms": (first - started_ns) / 1_000_000,
                "tpot_ms": (finished - first) / 1_000_000 / max(1, output_tokens - 1),
                "e2e_ms": (finished - started_ns) / 1_000_000,
            }
        )
        output_rows.append(
            {"wave_index": wave_index, "position": position, "token_ids": tokens}
        )
    return {
        "concurrency": concurrency,
        "wall_ms": (ended_ns - started_ns) / 1_000_000,
        "step_ms": step_ms,
        "request_metrics": request_metrics,
        "output_rows": output_rows,
    }


def run_trace(
    llm: Any,
    *,
    phase_name: str,
    phase: dict[str, Any],
    waves: list[int],
    pool: list[int],
    seed_base: int,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    results = [
        run_wave(
            llm,
            concurrency=concurrency,
            prompt_tokens=phase["prompt_tokens"],
            output_tokens=phase["output_tokens"],
            pool=pool,
            seed=seed_base + wave_index,
            wave_index=wave_index,
        )
        for wave_index, concurrency in enumerate(waves)
    ]
    wall_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    requests = [row for result in results for row in result["request_metrics"]]
    outputs = [row for result in results for row in result["output_rows"]]
    steps = [value for result in results for value in result["step_ms"]]
    useful_tokens = sum(waves) * phase["output_tokens"]
    return {
        "phase": phase_name,
        "wave_concurrency": waves,
        "wall_ms": wall_ms,
        "useful_output_tokens": useful_tokens,
        "correct_token_goodput_per_s": useful_tokens / (wall_ms / 1000.0),
        "request_metrics": summarize_request_metrics(requests),
        "step_median_ms": statistics.median(steps),
        "step_p99_ms": percentile(steps, 0.99),
        "step_count": len(steps),
        "output_digest": output_digest(outputs),
        "waves": results,
    }


def warmup_phase(llm: Any, phase: dict[str, Any], pool: list[int], seed: int) -> None:
    run_wave(
        llm,
        concurrency=phase["wave_concurrency"][0],
        prompt_tokens=phase["prompt_tokens"],
        output_tokens=min(8, phase["output_tokens"]),
        pool=pool,
        seed=seed,
        wave_index=-1,
    )


def run_instrumented_trace(
    llm: Any,
    *,
    phase_name: str,
    phase: dict[str, Any],
    pool: list[int],
    seed_base: int,
) -> dict[str, Any]:
    trace = DispatchTrace(llm)
    trace.start()
    try:
        replay = run_trace(
            llm,
            phase_name=phase_name,
            phase=phase,
            waves=phase["instrumented_wave_concurrency"],
            pool=pool,
            seed_base=seed_base,
        )
        dispatch = trace.finish()
    finally:
        trace.stop()
    return {
        "output_digest": replay["output_digest"],
        "timing_admissible": False,
        **dispatch,
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


def execute(args: argparse.Namespace) -> dict[str, Any]:
    configure_environment()
    config = load_config(args.config)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    validate_run(args, config, freeze)
    model_config_path = args.model / "config.json"
    if not model_config_path.is_file():
        raise FileNotFoundError(f"model config is absent: {model_config_path}")

    import torch
    import vllm
    from vllm import LLM

    if args.platform == "910b":
        import torch_npu  # noqa: F401

        if torch.npu.device_count() != 1:
            raise RuntimeError("GraphLease requires exactly one visible NPU")
        torch.npu.set_device(0)
        runtime_extra = {
            "torch_npu": package_version("torch-npu"),
            "vllm_ascend": package_version("vllm-ascend"),
        }
    else:
        if shutil.which("ninja") is None:
            raise RuntimeError(
                "A100 runtime requires ninja on PATH for the existing FlashInfer sampler"
            )
        if torch.cuda.device_count() != 1:
            raise RuntimeError("GraphLease requires exactly one visible CUDA GPU")
        torch.cuda.set_device(0)
        runtime_extra = {"cuda": torch.version.cuda, "ninja": shutil.which("ninja")}

    portfolio = config["portfolios"][args.portfolio]
    requested_compilation = compilation_config(config, args.portfolio)
    kwargs: dict[str, Any] = {
        "model": str(args.model),
        "tensor_parallel_size": 1,
        "dtype": config["model"]["dtype"],
        "seed": config["engine"]["seed"] + args.restart_index,
        "max_model_len": config["engine"]["max_model_len"],
        "max_num_seqs": config["engine"]["max_num_seqs"],
        "max_num_batched_tokens": config["engine"]["max_num_batched_tokens"],
        "gpu_memory_utilization": config["engine"]["gpu_memory_utilization"],
        "enable_prefix_caching": config["engine"]["enable_prefix_caching"],
        "async_scheduling": config["engine"]["async_scheduling"],
        "disable_log_stats": True,
        "cudagraph_metrics": False,
        "enforce_eager": portfolio["enforce_eager"],
        "compilation_config": requested_compilation,
        "disable_custom_all_reduce": args.platform == "910b",
    }
    if args.platform == "910b":
        kwargs["additional_config"] = config["platforms"]["910b"]["additional_config"]

    payload: dict[str, Any] = {
        "schema": "graphlease.g0.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "repository_commit": repository_commit(),
        "config_sha256": sha256(args.config),
        "freeze_sha256": sha256(args.freeze),
        "model_config_sha256": sha256(model_config_path),
        "mode": args.mode,
        "platform": args.platform,
        "portfolio": args.portfolio,
        "restart_index": args.restart_index,
        "physical_device": args.physical_device,
        "requested_compilation_config": requested_compilation,
        "runtime": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "vllm": vllm.__version__,
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
        payload["resource_after_init"] = llm.collective_rpc(worker_resource_snapshot, timeout=60)
        pool = token_pool(llm, config["workload"]["prompt_pattern"])
        payload["prompt_token_pool"] = pool

        if args.mode == "preflight":
            cases = []
            for case_index, case in enumerate(config["preflight"]["wave_cases"]):
                phase = config["workload"]["phases"][case["phase"]]
                trace = DispatchTrace(llm)
                trace.start()
                try:
                    result = run_wave(
                        llm,
                        concurrency=case["concurrency"],
                        prompt_tokens=phase["prompt_tokens"],
                        output_tokens=case["output_tokens"],
                        pool=pool,
                        seed=config["engine"]["seed"] + 10_000 + case_index,
                        wave_index=case_index,
                    )
                    dispatch = trace.finish()
                finally:
                    trace.stop()
                cases.append(
                    {
                        "case": case,
                        "output_digest": output_digest(result["output_rows"]),
                        "dispatch": dispatch,
                    }
                )
            payload["preflight_cases"] = cases
        elif args.mode == "formal":
            phase_results = []
            for phase_index, phase_name in enumerate(
                phase_order(config, args.restart_index)
            ):
                phase = config["workload"]["phases"][phase_name]
                seed_base = config["engine"]["seed"] + phase_index * 100_000
                warmup_phase(llm, phase, pool, seed_base - 1)
                formal = run_trace(
                    llm,
                    phase_name=phase_name,
                    phase=phase,
                    waves=phase["wave_concurrency"],
                    pool=pool,
                    seed_base=seed_base,
                )
                instrumented = run_instrumented_trace(
                    llm,
                    phase_name=phase_name,
                    phase=phase,
                    pool=pool,
                    seed_base=seed_base,
                )
                phase_results.append(
                    {
                        "phase": phase_name,
                        "formal": formal,
                        "instrumented": instrumented,
                        "instrumented_matches_formal": instrumented["output_digest"]
                        == formal["output_digest"],
                    }
                )
            payload["phase_results"] = phase_results
        else:
            for case_index, case in enumerate(config["preflight"]["wave_cases"]):
                phase = config["workload"]["phases"][case["phase"]]
                run_wave(
                    llm,
                    concurrency=case["concurrency"],
                    prompt_tokens=phase["prompt_tokens"],
                    output_tokens=case["output_tokens"],
                    pool=pool,
                    seed=config["engine"]["seed"] + 20_000 + case_index,
                    wave_index=case_index,
                )
            payload["resource_after_materialization"] = llm.collective_rpc(
                worker_resource_snapshot, timeout=60
            )
        payload["resource_after_run"] = llm.collective_rpc(
            worker_resource_snapshot, timeout=60
        )
        payload["engine_idle_after_run"] = not llm.llm_engine.has_unfinished_requests()
        payload["status"] = "success" if payload["engine_idle_after_run"] else "failed"
    except Exception as error:  # noqa: BLE001 - preserve external runtime failures
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
        payload["traceback"] = traceback.format_exc()
    finally:
        if llm is not None and hasattr(llm.llm_engine, "shutdown"):
            shutdown_started = time.perf_counter_ns()
            llm.llm_engine.shutdown()
            payload["shutdown_ms"] = (time.perf_counter_ns() - shutdown_started) / 1_000_000
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        payload["total_process_ms"] = (time.perf_counter_ns() - PROCESS_STARTED_NS) / 1_000_000
    return payload


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        payload = execute(args)
    except Exception as error:  # noqa: BLE001 - always emit a failed-run artifact
        payload = {
            "schema": "graphlease.g0.run.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "repository_commit": repository_commit(),
            "mode": args.mode,
            "platform": args.platform,
            "portfolio": args.portfolio,
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
    print(json.dumps({"status": payload["status"], "mode": args.mode, "portfolio": args.portfolio}))
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
