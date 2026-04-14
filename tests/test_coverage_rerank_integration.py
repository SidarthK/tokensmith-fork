import argparse
from unittest.mock import MagicMock, patch

from src.config import RAGConfig
from src.instrumentation.logging import RunLogger
from src.ranking.ranker import EnsembleRanker


class MockRetriever:
    def __init__(self, name, scores):
        self.name = name
        self.scores = scores

    def get_scores(self, query, pool_size, chunks):
        return self.scores


def test_get_answer_applies_rerank_selected_indices():
    from src.main import get_answer

    cfg = RAGConfig(
        top_k=2,
        num_candidates=5,
        ensemble_method="linear",
        ranker_weights={"faiss": 0.5, "bm25": 0.5},
        chunk_mode="recursive_sections",
        rerank_mode="coverage_mmr",
        rerank_top_k=2,
    )

    args = argparse.Namespace(
        system_prompt_mode="baseline",
        index_prefix="test_index",
        double_prompt=False,
    )

    chunks = [
        "Chunk 0: overview",
        "Chunk 1: complementary detail",
        "Chunk 2: unrelated",
    ]
    sources = ["doc1", "doc1", "doc1"]
    faiss_scores = {0: 0.9, 1: 0.8, 2: 0.1}
    bm25_scores = {0: 0.9, 1: 0.7, 2: 0.2}

    retrievers = [MockRetriever("faiss", faiss_scores), MockRetriever("bm25", bm25_scores)]
    ranker = EnsembleRanker(ensemble_method="linear", weights={"faiss": 0.5, "bm25": 0.5})

    artifacts = {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "meta": [{"page_numbers": [1]} for _ in chunks],
    }

    def mock_stream():
        yield "stubbed"

    with patch("src.main.answer", side_effect=lambda *a, **k: mock_stream()), \
         patch(
             "src.main.rerank",
             return_value=([chunks[1], chunks[0]], {"selected_indices": [1, 0]}),
         ):
        answer_text, chunks_info, _hyde = get_answer(
            question="multi-hop question",
            cfg=cfg,
            args=args,
            logger=RunLogger(),
            console=MagicMock(),
            artifacts=artifacts,
            is_test_mode=True,
        )

    assert answer_text == "stubbed"
    assert [item["chunk_id"] for item in chunks_info] == [1, 0]
