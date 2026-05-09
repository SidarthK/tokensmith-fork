#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd -P)"
cd "$ROOT_DIR"

RESULT_DIR="tests/results"
mkdir -p "$RESULT_DIR"

BASELINE_OUT="$RESULT_DIR/baseline_rerank_multihop.txt"
COVERAGE_OUT="$RESULT_DIR/coverage_mmr_multihop.txt"

echo "Running baseline rerank (cross_encoder) on multi-hop benchmark slice..."
pytest tests/test_benchmarks.py -s \
  --output-mode=terminal \
  --benchmarks-file=tests/benchmarks_multihop.yaml \
  --benchmark-ids="aries_atomicity,sql_isolation,oltp_vs_analytics,lossy_decomposition" \
  --rerank-mode=cross_encoder \
  --config=config/config.yaml \
  | tee "$BASELINE_OUT"

echo "Running coverage-aware rerank (coverage_mmr) on same benchmark slice..."
pytest tests/test_benchmarks.py -s \
  --output-mode=terminal \
  --benchmarks-file=tests/benchmarks_multihop.yaml \
  --benchmark-ids="aries_atomicity,sql_isolation,oltp_vs_analytics,lossy_decomposition" \
  --rerank-mode=coverage_mmr \
  --config=config/config.yaml \
  | tee "$COVERAGE_OUT"

echo "Saved outputs:"
echo "  - $BASELINE_OUT"
echo "  - $COVERAGE_OUT"
