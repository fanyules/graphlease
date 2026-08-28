#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PLATFORM PYTHON PHYSICAL_DEVICE" >&2
  exit 2
fi

platform=$1
python_bin=$2
physical_device=$3
case "$platform" in
  a100|910b) ;;
  *) echo "unsupported platform: $platform" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"
mkdir -p results/g0/formal/logs results/g0/resource/logs

for restart in 0 1 2; do
  for portfolio in eager default small_dense large_dense; do
    stem="${platform}_${portfolio}_r${restart}"
    "$python_bin" scripts/run_g0.py \
      --mode formal \
      --platform "$platform" \
      --portfolio "$portfolio" \
      --restart-index "$restart" \
      --model /data/models/qwen/Qwen3-1.7B \
      --physical-device "$physical_device" \
      --output "results/g0/formal/${stem}.json" \
      >"results/g0/formal/logs/${stem}.txt" 2>&1
    echo "completed $stem"
  done
done

stem="${platform}_coverage_union_r0"
"$python_bin" scripts/run_g0.py \
  --mode resource \
  --platform "$platform" \
  --portfolio coverage_union \
  --restart-index 0 \
  --model /data/models/qwen/Qwen3-1.7B \
  --physical-device "$physical_device" \
  --output "results/g0/resource/${stem}.json" \
  >"results/g0/resource/logs/${stem}.txt" 2>&1
echo "completed $stem"
