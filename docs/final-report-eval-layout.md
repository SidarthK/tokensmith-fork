# Final Report Evaluation Layout (Hard Multihop Sweep)

## Experiment Goal
Compare answer quality, retrieval quality, and latency across:
1. `rerank_mode=none` (baseline no reranker)
2. `rerank_mode=cross_encoder` (current)
3. `rerank_mode=coverage_mmr` with a small parameter sweep:
   - `coverage_mmr_lambda ∈ {0.5, 0.75, 0.9}`
   - `coverage_mmr_candidate_pool ∈ {30, 50}`
4. `rerank_mode=decompose_then_coverage_mmr` with a small follow-up sweep:
   - `coverage_mmr_lambda ∈ {0.5, 0.75, 0.9}`
   - `coverage_mmr_candidate_pool ∈ {30, 50}`
   - `decomposition_candidate_pool = 12`
   - `decomposition_max_subquestions = 4`

## Dataset
- File: `tests/benchmarks_hard_multihop.yaml`
- Total questions: 22
  - 3 `singlehop_*`
  - 19 multi-hop questions spanning:
    - `procedure_*`
    - `tradeoff_*`
    - `bridge_*`
    - `compare3_*`
    - `application_*`
    - `cross_section_*`

## Controlled Settings
- Same config/model/chunking/retrieval settings for all runs
- Same question order and benchmark file for all runs
- Only reranker mode / MMR sweep settings change per run
- All runs use `system_prompt=tutor` so answers stay grounded in retrieved textbook excerpts

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
  - `decomp_l05_p30`
  - `decomp_l075_p30`, `decomp_l075_p50`
  - `decomp_l09_p30`
- Aggregated report-ready summaries inside the run root:
  - `sweep_summary.md`
  - `sweep_summary.csv`
  - `family_breakdown.csv`
  - `topic_breakdown.csv`
  - `question_level_scores.csv`
  - `overall_final_score.png`
  - `pass_rate_by_family.png`
  - `latency_vs_final_score.png`
  - `chunk_hit_vs_final_score.png`
  - `question_win_heatmap.png`

## Suggested Final Report Questions
- Correctness:
  - Did all 22 questions return non-empty answers in each run?
  - Do diagnostics reflect the intended rerank mode?
- Experimental Results:
  - Which run has best mean `final_score` overall and by family/topic?
  - Does `coverage_mmr` improve the harder multi-hop families more than `none` and `cross_encoder`?
  - Does `decompose_then_coverage_mmr` outperform plain `coverage_mmr` on bridge, application, and cross-section questions?
  - How sensitive is performance to `coverage_mmr_lambda` and candidate-pool size?
  - What is the latency overhead relative to `cross_encoder` and `none`?
  - How does the chunk/page hit ratio change across modes?
- Future Work:
  - Adaptive decomposition triggers instead of always decomposing in the new mode
  - Threshold, lambda, and subquestion-weight tuning beyond the initial sweep
  - Runtime optimization for reranking latency
