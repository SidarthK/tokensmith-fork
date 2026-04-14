import numpy as np
from unittest.mock import patch

from src.ranking.reranker import rerank_with_coverage_mmr


def test_mmr_prefers_diverse_chunks():
    chunks = ["chunk_a", "chunk_b", "chunk_c"]
    relevance = np.array([0.95, 0.90, 0.85], dtype=np.float32)
    similarity = np.array(
        [
            [1.0, 0.99, 0.10],
            [0.99, 1.0, 0.20],
            [0.10, 0.20, 1.0],
        ],
        dtype=np.float32,
    )

    with patch("src.ranking.reranker.get_cross_encoder_scores", return_value=relevance), \
         patch("src.ranking.reranker.get_similarity_matrix", return_value=similarity):
        reranked, diagnostics = rerank_with_coverage_mmr(
            query="multi-hop question",
            chunks=chunks,
            top_n=2,
            lambda_param=0.6,
        )

    assert reranked == ["chunk_a", "chunk_c"]
    assert diagnostics["selected_indices"] == [0, 2]


def test_mmr_selection_is_deterministic():
    chunks = ["a", "b", "c"]
    relevance = np.array([0.7, 0.6, 0.5], dtype=np.float32)
    similarity = np.array(
        [
            [1.0, 0.3, 0.3],
            [0.3, 1.0, 0.3],
            [0.3, 0.3, 1.0],
        ],
        dtype=np.float32,
    )

    with patch("src.ranking.reranker.get_cross_encoder_scores", return_value=relevance), \
         patch("src.ranking.reranker.get_similarity_matrix", return_value=similarity):
        first, first_diag = rerank_with_coverage_mmr("q", chunks, top_n=3, lambda_param=0.7)
        second, second_diag = rerank_with_coverage_mmr("q", chunks, top_n=3, lambda_param=0.7)

    assert first == second
    assert first_diag["selected_indices"] == second_diag["selected_indices"]


def test_mmr_edge_cases():
    empty, empty_diag = rerank_with_coverage_mmr("q", [], top_n=3)
    assert empty == []
    assert empty_diag["selected_indices"] == []

    with patch(
        "src.ranking.reranker.get_cross_encoder_scores",
        return_value=np.array([0.9], dtype=np.float32),
    ):
        single, single_diag = rerank_with_coverage_mmr("q", ["only"], top_n=5)
    assert single == ["only"]
    assert single_diag["selected_indices"] == [0]
