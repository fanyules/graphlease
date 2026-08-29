#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graphlease.gb_q0 import load_config, run_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ordered GB-Q0 trial")
    parser.add_argument("--restart-index", type=int, required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "gb_q0.json"
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "gb_q0" / "GB_Q0_FREEZE.json",
    )
    return parser.parse_args()


def run_phase(args: argparse.Namespace, phase: str) -> dict:
    output = args.output_dir / run_filename(phase, args.restart_index)
    log = args.log_dir / output.with_suffix(".log").name
    if output.exists() or log.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {log}")
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_gb_q0.py"),
        "--phase",
        phase,
        "--restart-index",
        str(args.restart_index),
        "--physical-device",
        str(args.physical_device),
        "--model",
        str(args.model),
        "--output",
        str(output),
        "--config",
        str(args.config),
        "--freeze",
        str(args.freeze),
    ]
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_device)
    started = datetime.now(timezone.utc).isoformat()
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    return {
        "phase": phase,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": result.returncode,
        "result_status": payload.get("status"),
        "output": output.name,
        "log": log.name,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    devices = config["platform"]["physical_devices_by_restart"]
    if not 0 <= args.restart_index < len(devices):
        raise ValueError("restart index is outside the frozen trial set")
    if args.physical_device != devices[args.restart_index]:
        raise ValueError("physical device differs from the frozen trial assignment")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"trial_r{args.restart_index}.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")

    summary = {
        "schema": "graphlease.gb_q0.trial.v1",
        "restart_index": args.restart_index,
        "physical_device": args.physical_device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }
    default_pre = run_phase(args, "default_pre")
    summary["runs"].append(default_pre)
    if default_pre["result_status"] == "success":
        u60 = run_phase(args, "u60")
        summary["runs"].append(u60)
        if u60["result_status"] != "success":
            summary["runs"].append(run_phase(args, "default_recovery"))
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"trial": args.restart_index, "runs": len(summary["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
