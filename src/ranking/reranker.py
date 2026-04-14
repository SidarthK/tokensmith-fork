"""
reranker.py

This module supports re-ranking strategies applied before the generative LLM call.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np
from sentence_transformers import CrossEncoder
from src.embedder import CachedEmbedder

# -------------------------- Cross-Encoder Cache --------------------------
_CROSS_ENCODER_CACHE: Dict[str, CrossEncoder] = {}
_SIMILARITY_EMBEDDER_CACHE: Dict[str, CachedEmbedder] = {}

def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
    """
    Fetch the cached cross-encoder model to prevent reloading on every query.
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]

def get_similarity_embedder(model_path: str) -> CachedEmbedder:
    if model_path not in _SIMILARITY_EMBEDDER_CACHE:
        _SIMILARITY_EMBEDDER_CACHE[model_path] = CachedEmbedder(model_path)
    return _SIMILARITY_EMBEDDER_CACHE[model_path]

def get_cross_encoder_scores(query: str, chunks: List[str]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    model = get_cross_encoder()
    pairs = [(query, chunk) for chunk in chunks]
    return np.array(model.predict(pairs, show_progress_bar=False), dtype=np.float32)

def _token_jaccard_similarity_matrix(chunks: List[str]) -> np.ndarray:
    n = len(chunks)
    matrix = np.zeros((n, n), dtype=np.float32)
    token_sets = [set(chunk.lower().split()) for chunk in chunks]
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            union = token_sets[i] | token_sets[j]
            score = 0.0 if not union else len(token_sets[i] & token_sets[j]) / len(union)
            matrix[i, j] = score
            matrix[j, i] = score
    return matrix

def get_similarity_matrix(chunks: List[str], embed_model_path: str | None) -> np.ndarray:
    if len(chunks) <= 1:
        return np.eye(len(chunks), dtype=np.float32)
    if not embed_model_path:
        return _token_jaccard_similarity_matrix(chunks)
    try:
        embedder = get_similarity_embedder(embed_model_path)
        vectors = embedder.encode(chunks).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1e-12
        vectors = vectors / norms
        sim = np.dot(vectors, vectors.T)
        np.fill_diagonal(sim, 1.0)
        return sim.astype(np.float32)
    except Exception:
        # Fall back to lexical similarity if embedding fails.
        return _token_jaccard_similarity_matrix(chunks)

def rerank_with_coverage_mmr(
    query: str,
    chunks: List[str],
    top_n: int,
    lambda_param: float = 0.7,
    embed_model_path: str | None = None,
) -> Tuple[List[str], Dict[str, Any]]:
    if not chunks:
        return [], {"selected_indices": [], "relevance_scores": [], "mmr_scores": []}

    top_n = min(top_n, len(chunks))
    relevance_scores = get_cross_encoder_scores(query, chunks)
    similarity_matrix = get_similarity_matrix(chunks, embed_model_path)

    selected: List[int] = []
    remaining = set(range(len(chunks)))
    mmr_scores: Dict[int, float] = {}

    while len(selected) < top_n and remaining:
        best_idx = None
        best_score = -np.inf
        for idx in remaining:
            if not selected:
                redundancy_penalty = 0.0
            else:
                redundancy_penalty = float(max(similarity_matrix[idx, j] for j in selected))
            score = (lambda_param * float(relevance_scores[idx])) - ((1.0 - lambda_param) * redundancy_penalty)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        mmr_scores[best_idx] = float(best_score)

    diagnostics = {
        "selected_indices": selected,
        "relevance_scores": [float(s) for s in relevance_scores.tolist()],
        "mmr_scores": [mmr_scores[idx] for idx in selected],
        "lambda_param": lambda_param,
    }
    return [chunks[i] for i in selected], diagnostics

# -------------------------- Reranking Strategies -------------------------
def rerank_with_cross_encoder(query: str, chunks: List[str], top_n: int) -> List[str]:
    """
    Reranks a list of documents using the cross-encoder model.
    """
    if not chunks:
        print("[INSIDE RERANKER] Warning: No chunks to rerank. Returning empty list.")
        return []

    scores = get_cross_encoder_scores(query, chunks)

    # Combine chunks with their scores and sort
    chunk_with_scores = list(zip(chunks, scores))
    chunk_with_scores.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _score in chunk_with_scores[:top_n]]


# -------------------------- Reranking Router -----------------------------
def rerank(
    query: str,
    chunks: List[str],
    mode: str,
    top_n: int,
    *,
    coverage_mmr_lambda: float = 0.7,
    embed_model_path: str | None = None,
    return_diagnostics: bool = False,
) -> List[str] | Tuple[List[str], Dict[str, Any]]:
    """
    Routes to the appropriate reranker based on the mode in the config.
    """
    if mode == "cross_encoder":
        reranked = rerank_with_cross_encoder(query, chunks, top_n)
        if return_diagnostics:
            return reranked, {"selected_indices": list(range(len(reranked)))}
        return reranked
    if mode == "coverage_mmr":
        reranked, diagnostics = rerank_with_coverage_mmr(
            query,
            chunks,
            top_n,
            lambda_param=coverage_mmr_lambda,
            embed_model_path=embed_model_path,
        )
        if return_diagnostics:
            return reranked, diagnostics
        return reranked

    # We can add other re-ranking strategies to switch between them.
    if return_diagnostics:
        return chunks, {"selected_indices": list(range(min(top_n, len(chunks))))}
    return chunks
