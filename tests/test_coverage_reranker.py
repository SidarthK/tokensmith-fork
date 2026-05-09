import numpy as np
from unittest.mock import patch

from src.ranking.reranker import (
    _clean_subquestions,
    rerank_with_coverage_mmr,
    rerank_with_subquestion_coverage,
)


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


def test_clean_subquestions_dedupes_and_limits():
    cleaned = _clean_subquestions(
        "How do joins and indexes interact?",
        [
            "How do joins and indexes interact?",
            " What are merge joins? ",
            "What are merge joins?",
            "",
            "How do indexes help?",
            "What are hash joins?",
        ],
        max_subquestions=2,
    )

    assert cleaned == ["What are merge joins?", "How do indexes help?"]


def test_subquestion_coverage_rewards_complementary_chunks():
    chunks = ["covers subquestion one", "covers subquestion two", "redundant overview"]

    def fake_cross_encoder_scores(query, query_chunks):
        mapping = {
            "complex question": np.array([0.82, 0.81, 0.79], dtype=np.float32),
            "subquestion one": np.array([0.98, 0.15, 0.45], dtype=np.float32),
            "subquestion two": np.array([0.15, 0.97, 0.44], dtype=np.float32),
        }
        return mapping[query]

    similarity = np.array(
        [
            [1.0, 0.10, 0.85],
            [0.10, 1.0, 0.86],
            [0.85, 0.86, 1.0],
        ],
        dtype=np.float32,
    )

    with patch("src.ranking.reranker.get_cross_encoder_scores", side_effect=fake_cross_encoder_scores), \
         patch("src.ranking.reranker.get_similarity_matrix", return_value=similarity):
        reranked, diagnostics = rerank_with_subquestion_coverage(
            query="complex question",
            chunks=chunks,
            subquestions=["subquestion one", "subquestion two"],
            top_n=2,
            lambda_param=0.65,
            subquestion_weight=0.50,
        )

    assert reranked == ["covers subquestion one", "covers subquestion two"]
    assert diagnostics["subquestions"] == ["subquestion one", "subquestion two"]
    assert len(diagnostics["selection_scores"]) == 2


def test_subquestion_coverage_falls_back_without_subquestions():
    with patch(
        "src.ranking.reranker.get_cross_encoder_scores",
        return_value=np.array([0.9, 0.6], dtype=np.float32),
    ), patch(
        "src.ranking.reranker.get_similarity_matrix",
        return_value=np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32),
    ):
        reranked, diagnostics = rerank_with_subquestion_coverage(
            query="simple question",
            chunks=["chunk a", "chunk b"],
            subquestions=[],
            top_n=1,
        )

    assert reranked == ["chunk a"]
    assert diagnostics["coverage_mode"] == "fallback_to_coverage_mmr"
