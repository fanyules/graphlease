#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graphlease.gb_q0 import load_config, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze GB-Q0 before output")
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "gb_q0.json"
    )
    parser.add_argument(
        "--protocol", type=Path, default=REPOSITORY_ROOT / "docs" / "GB_Q0_PROTOCOL.md"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def repository_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = load_config(args.config)
    prior = REPOSITORY_ROOT / config["prior_evidence"]["gl_s0_frozen_decision_path"]
    if sha256(prior) != config["prior_evidence"]["gl_s0_frozen_decision_sha256"]:
        raise ValueError("the frozen GL-S0 decision differs from the GB-Q0 reference")
    sources = [
        REPOSITORY_ROOT / "src" / "graphlease" / "gb_q0.py",
        REPOSITORY_ROOT / "scripts" / "run_gb_q0.py",
        REPOSITORY_ROOT / "scripts" / "run_gb_q0_trial.py",
        REPOSITORY_ROOT / "scripts" / "adjudicate_gb_q0.py",
    ]
    payload = {
        "schema": "graphlease.gb_q0.freeze.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "pre_output_repository_commit": repository_commit(),
        "config_path": args.config.relative_to(REPOSITORY_ROOT).as_posix(),
        "config_sha256": sha256(args.config),
        "protocol_path": args.protocol.relative_to(REPOSITORY_ROOT).as_posix(),
        "protocol_sha256": sha256(args.protocol),
        "source_sha256": {
            path.relative_to(REPOSITORY_ROOT).as_posix(): sha256(path) for path in sources
        },
        "prior_gl_s0_decision_sha256": sha256(prior),
        "official_stack": config["official_stack"],
        "matrix": {
            "tensor_parallel_size": 1,
            "physical_devices_by_restart": config["platform"]["physical_devices_by_restart"],
            "trial_order": config["execution"]["trial_order"],
            "fresh_u60_processes": 3,
            "default_recovery_after_failure": True,
        },
        "composite_resource_signature": config["composite_resource_signature"],
        "performance_unblocked": False,
        "router_unblocked": False,
        "graphbudget_repository_unblocked": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "frozen", "commit": payload["pre_output_repository_commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
