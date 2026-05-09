from typing import List, Optional
from tests.metrics.base import MetricBase


class ChunkRetrievalMetric(MetricBase):
    """Chunk retrieval evaluation metric."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
    
    @property
    def name(self) -> str:
        return "chunk_retrieval"

    @property
    def weight(self) -> float:
        return 0.2
    
    def calculate(self, 
        ideal_retrieved_chunks: List[int],
        ideal_retrieved_pages: Optional[List[int]],
        retrieved_chunks) -> float:
        if not retrieved_chunks:
            return 0.0
        actual_chunk_ids = {chunk.get("chunk_id") for chunk in retrieved_chunks if chunk.get("chunk_id") is not None}
        actual_pages = {
            page
            for chunk in retrieved_chunks
            for page in chunk.get("page_numbers", [])
        }

        chunk_ratio = 0.0
        if ideal_retrieved_chunks:
            chunk_hits = len(actual_chunk_ids & set(ideal_retrieved_chunks))
            chunk_ratio = chunk_hits / max(len(set(ideal_retrieved_chunks)), 1)

        page_ratio = 0.0
        if ideal_retrieved_pages:
            page_hits = len(actual_pages & set(ideal_retrieved_pages))
            page_ratio = page_hits / max(len(set(ideal_retrieved_pages)), 1)

        return max(chunk_ratio, page_ratio)
