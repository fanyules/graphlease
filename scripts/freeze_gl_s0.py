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

from graphlease.gl_s0 import load_config, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the prospective GL-S0 probe")
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "gl_s0.json"
    )
    parser.add_argument(
        "--protocol", type=Path, default=REPOSITORY_ROOT / "docs" / "GL_S0_PROTOCOL.md"
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
    sources = [
        REPOSITORY_ROOT / "src" / "graphlease" / "gl_s0.py",
        REPOSITORY_ROOT / "scripts" / "run_gl_s0.py",
        REPOSITORY_ROOT / "scripts" / "adjudicate_gl_s0.py",
    ]
    payload = {
        "schema": "graphlease.gl_s0.freeze.v1",
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
        "frozen_negative": config["frozen_negative"],
        "held_out_prediction_basis": config["held_out_prediction_basis"],
        "matrix": {
            "platforms": list(config["platforms"]),
            "portfolios": list(config["portfolios"]),
            "fresh_processes_per_cell": 3,
            "total_processes": 12,
        },
        "workload_unblocked": False,
        "controller_unblocked": False,
        "router_unblocked": False,
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
