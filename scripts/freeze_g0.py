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

from graphlease.contracts import load_config, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze GraphLease Gate 0")
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "g0.json"
    )
    parser.add_argument(
        "--protocol", type=Path, default=REPOSITORY_ROOT / "docs" / "G0_PROTOCOL.md"
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


def execute(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    source_files = [
        REPOSITORY_ROOT / "src" / "graphlease" / "contracts.py",
        REPOSITORY_ROOT / "src" / "graphlease" / "metrics.py",
        REPOSITORY_ROOT / "scripts" / "run_g0.py",
        REPOSITORY_ROOT / "scripts" / "adjudicate_g0.py",
    ]
    return {
        "schema": "graphlease.g0.freeze.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "pre_output_repository_commit": repository_commit(),
        "config_path": args.config.relative_to(REPOSITORY_ROOT).as_posix(),
        "config_sha256": sha256(args.config),
        "protocol_path": args.protocol.relative_to(REPOSITORY_ROOT).as_posix(),
        "protocol_sha256": sha256(args.protocol),
        "source_sha256": {
            path.relative_to(REPOSITORY_ROOT).as_posix(): sha256(path)
            for path in source_files
        },
        "model": config["model"],
        "formal_portfolios": [
            name for name, value in config["portfolios"].items() if value.get("formal")
        ],
        "resource_probe_portfolio": "coverage_union",
        "phases": config["workload"]["phases"],
        "a100_unblocked": True,
        "910b_unblocked": True,
        "controller_unblocked": False,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = execute(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "commit": payload["pre_output_repository_commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
