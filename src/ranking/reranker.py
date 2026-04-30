"""
reranker.py

This module supports re-ranking strategies applied before the generative LLM call.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
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

def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if np.isclose(max_score, min_score):
        return np.ones_like(scores, dtype=np.float32)
    return ((scores - min_score) / (max_score - min_score)).astype(np.float32)

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
    raw_relevance_scores = get_cross_encoder_scores(query, chunks)
    relevance_scores = _normalize_scores(raw_relevance_scores)
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
        "raw_relevance_scores": [float(s) for s in raw_relevance_scores.tolist()],
        "relevance_scores": [float(s) for s in relevance_scores.tolist()],
        "mmr_scores": [mmr_scores[idx] for idx in selected],
        "lambda_param": lambda_param,
    }
    return [chunks[i] for i in selected], diagnostics


def _clean_subquestions(
    query: str,
    subquestions: Optional[List[str]],
    max_subquestions: int,
) -> List[str]:
    if not subquestions:
        return []
    seen = {query.strip().lower()}
    cleaned: List[str] = []
    for subquestion in subquestions:
        normalized = " ".join(subquestion.split()).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(normalized)
        if len(cleaned) >= max_subquestions:
            break
    return cleaned


def rerank_with_subquestion_coverage(
    query: str,
    chunks: List[str],
    subquestions: Optional[List[str]],
    top_n: int,
    lambda_param: float = 0.7,
    subquestion_weight: float = 0.35,
    embed_model_path: str | None = None,
) -> Tuple[List[str], Dict[str, Any]]:
    if not chunks:
        return [], {
            "selected_indices": [],
            "subquestions": [],
            "relevance_scores": [],
            "subquestion_scores": {},
            "selection_scores": [],
        }

    cleaned_subquestions = _clean_subquestions(query, subquestions, max_subquestions=8)
    if not cleaned_subquestions:
        reranked, diagnostics = rerank_with_coverage_mmr(
            query,
            chunks,
            top_n,
            lambda_param=lambda_param,
            embed_model_path=embed_model_path,
        )
        diagnostics["subquestions"] = []
        diagnostics["coverage_mode"] = "fallback_to_coverage_mmr"
        return reranked, diagnostics

    top_n = min(top_n, len(chunks))
    raw_relevance_scores = get_cross_encoder_scores(query, chunks)
    relevance_scores = _normalize_scores(raw_relevance_scores)
    similarity_matrix = get_similarity_matrix(chunks, embed_model_path)

    subquestion_raw: Dict[str, np.ndarray] = {}
    subquestion_norm: Dict[str, np.ndarray] = {}
    for subquestion in cleaned_subquestions:
        sq_raw = get_cross_encoder_scores(subquestion, chunks)
        subquestion_raw[subquestion] = sq_raw
        subquestion_norm[subquestion] = _normalize_scores(sq_raw)

    selected: List[int] = []
    remaining = set(range(len(chunks)))
    covered_aspects = {subquestion: 0.0 for subquestion in cleaned_subquestions}

    selection_scores: Dict[int, float] = {}
    coverage_bonus_by_idx: Dict[int, float] = {}
    redundancy_penalty_by_idx: Dict[int, float] = {}

    while len(selected) < top_n and remaining:
        best_idx = None
        best_score = -np.inf
        best_coverage_bonus = 0.0
        best_redundancy_penalty = 0.0

        for idx in remaining:
            if not selected:
                redundancy_penalty = 0.0
            else:
                redundancy_penalty = float(max(similarity_matrix[idx, j] for j in selected))

            aspect_terms = []
            for subquestion in cleaned_subquestions:
                sq_score = float(subquestion_norm[subquestion][idx])
                aspect_terms.append(sq_score * (1.0 - covered_aspects[subquestion]))
            coverage_bonus = float(np.mean(aspect_terms)) if aspect_terms else 0.0

            score = (
                lambda_param * float(relevance_scores[idx])
                + subquestion_weight * coverage_bonus
                - (1.0 - lambda_param) * redundancy_penalty
            )
            if score > best_score:
                best_score = score
                best_idx = idx
                best_coverage_bonus = coverage_bonus
                best_redundancy_penalty = redundancy_penalty

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        selection_scores[best_idx] = float(best_score)
        coverage_bonus_by_idx[best_idx] = float(best_coverage_bonus)
        redundancy_penalty_by_idx[best_idx] = float(best_redundancy_penalty)

        for subquestion in cleaned_subquestions:
            covered_aspects[subquestion] = max(
                covered_aspects[subquestion],
                float(subquestion_norm[subquestion][best_idx]),
            )

    diagnostics = {
        "selected_indices": selected,
        "subquestions": cleaned_subquestions,
        "coverage_mode": "subquestion_xquad",
        "lambda_param": lambda_param,
        "coverage_subquestion_weight": subquestion_weight,
        "raw_relevance_scores": [float(s) for s in raw_relevance_scores.tolist()],
        "relevance_scores": [float(s) for s in relevance_scores.tolist()],
        "subquestion_scores": {
            subquestion: [float(score) for score in subquestion_norm[subquestion].tolist()]
            for subquestion in cleaned_subquestions
        },
        "raw_subquestion_scores": {
            subquestion: [float(score) for score in subquestion_raw[subquestion].tolist()]
            for subquestion in cleaned_subquestions
        },
        "selection_scores": [selection_scores[idx] for idx in selected],
        "coverage_bonus": [coverage_bonus_by_idx[idx] for idx in selected],
        "redundancy_penalty": [redundancy_penalty_by_idx[idx] for idx in selected],
        "final_aspect_coverage": covered_aspects,
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
    ranked_indices = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)[:top_n]
    return [chunks[i] for i in ranked_indices]


# -------------------------- Reranking Router -----------------------------
def rerank(
    query: str,
    chunks: List[str],
    mode: str,
    top_n: int,
    *,
    coverage_mmr_lambda: float = 0.7,
    coverage_subquestion_weight: float = 0.35,
    subquestions: Optional[List[str]] = None,
    embed_model_path: str | None = None,
    return_diagnostics: bool = False,
) -> List[str] | Tuple[List[str], Dict[str, Any]]:
    """
    Routes to the appropriate reranker based on the mode in the config.
    """
    if mode == "cross_encoder":
        scores = get_cross_encoder_scores(query, chunks)
        ranked_indices = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)[:top_n]
        reranked = [chunks[i] for i in ranked_indices]
        if return_diagnostics:
            return reranked, {"selected_indices": ranked_indices}
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
    if mode == "decompose_then_coverage_mmr":
        reranked, diagnostics = rerank_with_subquestion_coverage(
            query,
            chunks,
            subquestions=subquestions,
            top_n=top_n,
            lambda_param=coverage_mmr_lambda,
            subquestion_weight=coverage_subquestion_weight,
            embed_model_path=embed_model_path,
        )
        if return_diagnostics:
            return reranked, diagnostics
        return reranked

    # We can add other re-ranking strategies to switch between them.
    if return_diagnostics:
        selected = list(range(min(top_n, len(chunks))))
        return [chunks[i] for i in selected], {"selected_indices": selected}
    return chunks[:top_n]
