# TokenSmith Architecture Map and Checkpoint Execution Plan

This document captures the current TokenSmith pipeline and the implementation sequence for the CS 4423 checkpoint milestone on branch `sk/project-dev`.

## Current End-to-End Pipeline

1. PDF extraction
- Entry: `make run-extract` -> `src.preprocessing.extraction.main()`
- Converts `data/chapters/*.pdf` to markdown in `data/*.md`.
- Injects page markers for chunk-to-page traceability.

2. Section parsing and chunking
- Entry: `src.index_builder.build_index()`
- Section extraction: `extract_sections_from_markdown()`
- Chunking strategy: `SectionRecursiveStrategy` via `RAGConfig`.

3. Index artifact creation
- Embeddings from local GGUF embedding model (`src.embedder.SentenceTransformer`).
- FAISS index + BM25 index + chunk/source/meta artifacts written under `index/sections/`.

4. Retrieval and candidate fusion
- Chat path (`src.main.get_answer()` and API paths) loads artifacts.
- Retriever signals: FAISS, BM25, optional index-keyword retriever.
- Fusion: `EnsembleRanker` with RRF/linear weighted combination.

5. Re-ranking
- Current path supports cross-encoder reranking in `src.ranking.reranker`.
- Checkpoint upgrade adds coverage-aware MMR reranking to reduce redundancy and improve multi-hop coverage.

6. Generation
- Local LLM generation via `llama_cpp` (`src.generator`).
- Prompt formatted with retrieved chunks and optional system prompt mode.
- Streaming and logging enabled for CLI and API.

## Checkpoint Implementation Sequence

1. Add coverage-aware reranker controls to config (`coverage_mmr_*` knobs).
2. Implement MMR reranker using:
- Relevance: cross-encoder query-chunk scores.
- Redundancy: chunk-chunk cosine similarity from embedding vectors.
3. Integrate reranking in both CLI and API answer paths.
4. Add reranking diagnostics into logs for reproducibility.
5. Add tests for deterministic selection and edge cases.
6. Add benchmark scenarios focused on multi-hop and redundancy.
7. Replace report template with checkpoint report + baseline learning episode appendix.

## Git Workflow for This Milestone

- Development branch: `sk/project-dev`.
- Keep `main` synchronized with `upstream/main` only.
- Use incremental, topic-focused commits (no single mega-commit).
- Avoid committing generated artifacts (`*.fls`, `*.fdb_latexmk`) unless explicitly required.
