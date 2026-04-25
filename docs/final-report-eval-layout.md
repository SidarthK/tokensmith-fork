# Final Report Evaluation Layout (Hard Multihop Sweep)

## Experiment Goal
Compare answer quality, retrieval quality, and latency across:
1. `rerank_mode=none` (baseline no reranker)
2. `rerank_mode=cross_encoder` (current)
3. `rerank_mode=coverage_mmr` with a small parameter sweep:
   - `coverage_mmr_lambda ∈ {0.5, 0.75, 0.9}`
   - `coverage_mmr_candidate_pool ∈ {30, 50}`

## Dataset
- File: `tests/benchmarks_hard_multihop.yaml`
- Total questions: 16
  - 3 `singlehop_*`
  - 13 `multihop_*`
  - 0 `distractor_*`

## Controlled Settings
- Same config/model/chunking/retrieval settings for all runs
- Same question order and benchmark file for all runs
- Only reranker mode / MMR sweep settings change per run
- All three runs use `system_prompt=tutor` so answers stay grounded in retrieved textbook excerpts

## Run Procedure
```bash
bash scripts/run_hard_multihop_sweep.sh
```

## Output Artifacts
- Timestamped run root:
  - `tests/results/hard_multihop/<timestamp>/`
- Per-run subdirectories for:
  - `none`
  - `cross_encoder`
  - `mmr_l05_p30`, `mmr_l05_p50`
  - `mmr_l075_p30`, `mmr_l075_p50`
  - `mmr_l09_p30`, `mmr_l09_p50`
- Aggregated report-ready summaries inside the run root:
  - `sweep_summary.md`
  - `sweep_summary.csv`

## Suggested Final Report Questions
- Correctness:
  - Did all 16 questions return non-empty answers in each run?
  - Do diagnostics reflect the intended rerank mode?
- Experimental Results:
  - Which run has best mean `final_score` overall and by category?
  - Does `coverage_mmr` improve the harder multihop questions more than `none` and `cross_encoder`?
  - How sensitive is performance to `coverage_mmr_lambda` and candidate-pool size?
  - What is the latency overhead relative to `cross_encoder` and `none`?
  - How does the chunk/page hit ratio change across modes?
- Future Work:
  - Query decomposition or section-aware retrieval for very hard multihop questions
  - Threshold and lambda tuning beyond the initial sweep
  - Runtime optimization for reranking latency
