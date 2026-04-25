#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd -P)"
cd "$ROOT_DIR"

export ENABLE_GEMINI_JUDGE="${ENABLE_GEMINI_JUDGE:-0}"

BENCH_FILE="tests/benchmarks_hard_multihop.yaml"
CONFIG_FILE="config/config.yaml"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_ROOT="tests/results/hard_multihop/${TIMESTAMP}"

mkdir -p "$RUN_ROOT"
cp "$BENCH_FILE" "$RUN_ROOT/benchmark_snapshot.yaml"
cp "$CONFIG_FILE" "$RUN_ROOT/config_snapshot.yaml"

write_metadata() {
  local run_dir="$1"
  local label="$2"
  local mode="$3"
  local lambda_value="${4:-}"
  local pool_value="${5:-}"

  python - "$run_dir" "$label" "$mode" "$lambda_value" "$pool_value" "$BENCH_FILE" "$CONFIG_FILE" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

run_dir, label, mode, lambda_value, pool_value, bench_file, config_file = sys.argv[1:]

def parse_optional_float(value):
    return None if value == "" else float(value)

def parse_optional_int(value):
    return None if value == "" else int(value)

payload = {
    "label": label,
    "mode": mode,
    "coverage_mmr_lambda": parse_optional_float(lambda_value),
    "coverage_mmr_candidate_pool": parse_optional_int(pool_value),
    "benchmark_file": bench_file,
    "config_file": config_file,
    "created_at": datetime.now().isoformat(),
}

Path(run_dir, "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_case() {
  local label="$1"
  local mode="$2"
  local lambda_value="${3:-}"
  local pool_value="${4:-}"
  local run_dir="$RUN_ROOT/$label"
  local terminal_log="$run_dir/terminal.log"

  mkdir -p "$run_dir"
  write_metadata "$run_dir" "$label" "$mode" "$lambda_value" "$pool_value"

  echo "============================================================"
  echo "Running label: $label"
  echo "Mode: $mode"
  echo "Benchmark file: $BENCH_FILE"
  if [[ -n "$lambda_value" ]]; then
    echo "coverage_mmr_lambda: $lambda_value"
  fi
  if [[ -n "$pool_value" ]]; then
    echo "coverage_mmr_candidate_pool: $pool_value"
  fi
  echo "Results dir: $run_dir"
  echo "============================================================"

  local pytest_args=(
    tests/test_benchmarks.py
    -s
    --output-mode=terminal
    --benchmarks-file="$BENCH_FILE"
    --config="$CONFIG_FILE"
    --system-prompt=tutor
    --rerank-mode="$mode"
    --results-dir="$run_dir"
    --metrics semantic
    --metrics keyword
    --metrics nli
    --metrics chunk_retrieval
  )

  if [[ -n "$lambda_value" ]]; then
    pytest_args+=(--coverage-mmr-lambda="$lambda_value")
  fi
  if [[ -n "$pool_value" ]]; then
    pytest_args+=(--coverage-mmr-candidate-pool="$pool_value")
  fi

  pytest "${pytest_args[@]}" | tee "$terminal_log"

  if [[ ! -f "$run_dir/benchmark_results.json" ]]; then
    echo "ERROR: Expected results file not found for run '$label'" >&2
    exit 1
  fi

  echo "Saved JSONL results: $run_dir/benchmark_results.json"
  echo "Saved terminal log: $terminal_log"
}

run_case "none" "none"
run_case "cross_encoder" "cross_encoder"

for lambda_value in 0.5 0.75 0.9; do
  for pool_value in 30 50; do
    label="mmr_l${lambda_value/./}_p${pool_value}"
    run_case "$label" "coverage_mmr" "$lambda_value" "$pool_value"
  done
done

python tests/utils/summarize_hard_multihop_sweep.py --run-root "$RUN_ROOT"

echo "Finished hard multihop sweep."
echo "Run root: $RUN_ROOT"
echo "Summary markdown: $RUN_ROOT/sweep_summary.md"
echo "Summary csv: $RUN_ROOT/sweep_summary.csv"
