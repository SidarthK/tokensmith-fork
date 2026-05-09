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
  local decomp_pool="${6:-}"
  local max_subquestions="${7:-}"
  local subquestion_weight="${8:-}"

  python3 - "$run_dir" "$label" "$mode" "$lambda_value" "$pool_value" "$decomp_pool" "$max_subquestions" "$subquestion_weight" "$BENCH_FILE" "$CONFIG_FILE" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

(
    run_dir,
    label,
    mode,
    lambda_value,
    pool_value,
    decomp_pool,
    max_subquestions,
    subquestion_weight,
    bench_file,
    config_file,
) = sys.argv[1:]

def parse_optional_float(value):
    return None if value == "" else float(value)

def parse_optional_int(value):
    return None if value == "" else int(value)

payload = {
    "label": label,
    "mode": mode,
    "coverage_mmr_lambda": parse_optional_float(lambda_value),
    "coverage_mmr_candidate_pool": parse_optional_int(pool_value),
    "decomposition_candidate_pool": parse_optional_int(decomp_pool),
    "decomposition_max_subquestions": parse_optional_int(max_subquestions),
    "coverage_subquestion_weight": parse_optional_float(subquestion_weight),
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
  local decomp_pool="${5:-}"
  local max_subquestions="${6:-}"
  local subquestion_weight="${7:-}"
  local run_dir="$RUN_ROOT/$label"
  local terminal_log="$run_dir/terminal.log"

  mkdir -p "$run_dir"
  write_metadata "$run_dir" "$label" "$mode" "$lambda_value" "$pool_value" "$decomp_pool" "$max_subquestions" "$subquestion_weight"

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
  if [[ -n "$decomp_pool" ]]; then
    echo "decomposition_candidate_pool: $decomp_pool"
  fi
  if [[ -n "$max_subquestions" ]]; then
    echo "decomposition_max_subquestions: $max_subquestions"
  fi
  if [[ -n "$subquestion_weight" ]]; then
    echo "coverage_subquestion_weight: $subquestion_weight"
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
  if [[ -n "$decomp_pool" ]]; then
    pytest_args+=(--decomposition-candidate-pool="$decomp_pool")
  fi
  if [[ -n "$max_subquestions" ]]; then
    pytest_args+=(--decomposition-max-subquestions="$max_subquestions")
  fi
  if [[ -n "$subquestion_weight" ]]; then
    pytest_args+=(--coverage-subquestion-weight="$subquestion_weight")
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

# Decomposition-guided reranking: start with the best MMR region, then widen slightly.
run_case "decomp_l075_p30" "decompose_then_coverage_mmr" "0.75" "30" "12" "4" "0.35"
run_case "decomp_l05_p30" "decompose_then_coverage_mmr" "0.5" "30" "12" "4" "0.35"
run_case "decomp_l075_p50" "decompose_then_coverage_mmr" "0.75" "50" "12" "4" "0.35"
run_case "decomp_l09_p30" "decompose_then_coverage_mmr" "0.9" "30" "12" "4" "0.35"

python3 tests/utils/summarize_hard_multihop_sweep.py --run-root "$RUN_ROOT"

echo "Finished hard multihop sweep."
echo "Run root: $RUN_ROOT"
echo "Summary markdown: $RUN_ROOT/sweep_summary.md"
echo "Summary csv: $RUN_ROOT/sweep_summary.csv"
echo "Family breakdown: $RUN_ROOT/family_breakdown.csv"
echo "Topic breakdown: $RUN_ROOT/topic_breakdown.csv"
echo "Question scores: $RUN_ROOT/question_level_scores.csv"
