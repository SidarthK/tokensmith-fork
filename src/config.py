from __future__ import annotations

import os
import inspect
from dataclasses import dataclass, field
from typing import Dict

import yaml
import pathlib

from src.preprocessing.chunking import ChunkStrategy, SectionRecursiveStrategy, SectionRecursiveConfig, ChunkConfig

@dataclass
class RAGConfig:
    # chunking
    chunk_config: ChunkConfig = field(init=False)
    chunk_mode: str = "recursive_sections"
    chunk_size_in_chars: int = 2000
    chunk_overlap: int = 300

    # retrieval + ranking
    top_k: int = 10
    num_candidates: int = 60
    embed_model: str = "models/embedders/Qwen3-Embedding-4B-Q5_K_M.gguf"
    embedding_model_context_window: int = 4096
    ensemble_method: str = "rrf"
    rrf_k: int = 60
    ranker_weights: Dict[str, float] = field(
        default_factory=lambda: {"faiss": 1.0, "bm25": 0.0, "index_keywords": 0.0}
    )
    rerank_mode: str = ""
    rerank_top_k: int = 5
    rerank_candidate_pool: int = 20
    coverage_mmr_candidate_pool: int = 40
    coverage_mmr_lambda: float = 0.7
    coverage_mmr_similarity_metric: str = "cosine"

    # generation
    max_gen_tokens: int = 400
    gen_model: str = "models/generators/qwen2.5-1.5b-instruct-q8_0.gguf"
    
    # testing
    system_prompt_mode: str = "baseline"
    disable_chunks: bool = False
    use_golden_chunks: bool = False
    output_mode: str = "terminal"
    metrics: list = field(default_factory=lambda: ["all"])

    # query enhancement
    use_hyde: bool = False
    hyde_max_tokens: int = 300
    use_double_prompt: bool = False

    # cache
    semantic_cache_enabled: bool = False
    semantic_cache_bi_encoder_threshold: float = 0.90
    semantic_cache_cross_encoder_threshold: float = 0.99

    # conversational memory
    enable_history: bool = True
    max_history_turns: int = 3

    # index parameters
    use_indexed_chunks: bool = False
    extracted_index_path: os.PathLike = "data/extracted_index.json"
    page_to_chunk_map_path: os.PathLike = "index/sections/textbook_index_page_to_chunk_map.json"

    # user feedback modeling
    enable_topic_extraction: bool = False

    # ---------- factory + validation ----------
    @classmethod
    def from_yaml(cls, path: os.PathLike) -> RAGConfig:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            data = {}

        # Accept older config files after merges/refactors.
        if "chunk_size" in data and "chunk_size_in_chars" not in data:
            data["chunk_size_in_chars"] = data.pop("chunk_size")
        if "model_path" in data and "gen_model" not in data:
            data["gen_model"] = data.pop("model_path")

        valid_fields = set(inspect.signature(cls).parameters.keys())
        data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**data)

    def __post_init__(self):
        """Validation logic runs automatically after initialization."""
        assert self.top_k > 0, "top_k must be > 0"
        assert self.num_candidates >= self.top_k, "num_candidates must be >= top_k"
        assert self.ensemble_method.lower() in {"linear","weighted","rrf"}
        assert self.rerank_top_k > 0, "rerank_top_k must be > 0"
        assert self.rerank_candidate_pool >= self.rerank_top_k, \
            "rerank_candidate_pool must be >= rerank_top_k"
        assert self.coverage_mmr_candidate_pool >= self.rerank_top_k, \
            "coverage_mmr_candidate_pool must be >= rerank_top_k"
        assert self.rerank_mode in {"", "none", "cross_encoder", "coverage_mmr"}, \
            "rerank_mode must be one of: '', none, cross_encoder, coverage_mmr"
        assert 0.0 <= self.coverage_mmr_lambda <= 1.0, "coverage_mmr_lambda must be in [0, 1]"
        assert self.coverage_mmr_similarity_metric in {"cosine"}, \
            "coverage_mmr_similarity_metric currently supports only 'cosine'"
        if self.ensemble_method.lower() in {"linear","weighted"}:
            s = sum(self.ranker_weights.values()) or 1.0
            self.ranker_weights = {k: v / s for k, v in self.ranker_weights.items()}
        self.chunk_config = self.get_chunk_config()
        self.chunk_config.validate()

    # ---------- chunking + artifact name helpers ----------

    def get_chunk_config(self) -> ChunkConfig:
        """Parse chunk configuration from YAML."""
        if self.chunk_mode == "recursive_sections":
            return SectionRecursiveConfig(
                recursive_chunk_size=self.chunk_size_in_chars,
                recursive_overlap=self.chunk_overlap,
            )
        else:
            raise ValueError(f"Unknown chunk_mode: {self.chunk_mode}. Supported: recursive_sections")

    def get_chunk_strategy(self) -> ChunkStrategy:
        if isinstance(self.chunk_config, SectionRecursiveConfig):
            return SectionRecursiveStrategy(self.chunk_config)
        raise ValueError(f"Unknown chunk config type: {self.chunk_config.__class__.__name__}")

    def get_artifacts_directory(self, partial: bool = False) -> os.PathLike:
        """
        Returns the path prefix for index artifacts.
        If partial=True, strictly returns the partial directory.
        If partial=False, returns the main directory if it exists, 
        otherwise falls back to the partial directory.
        """
        strategy = self.get_chunk_strategy()
        base_folder = strategy.artifact_folder_name()
        
        main_dir = pathlib.Path("index", base_folder)
        partial_dir = pathlib.Path("index", f"partial_{base_folder}")

        if partial:
            target_dir = partial_dir
            print("Using partial directory (change partial to false in config.yaml to use full directory)")
        else:
            # Fallback logic: use main if it exists, otherwise use partial if it exists
            if main_dir.exists():
                target_dir = main_dir
            elif partial_dir.exists():
                target_dir = partial_dir
                print("Using partial directory (unable to find full directory)")
            else:
                target_dir = main_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def get_page_to_chunk_map_path(self, artifacts_dir: os.PathLike, index_prefix: str) -> os.PathLike:
        """Returns the path to the page-to-chunk map file."""
        return pathlib.Path(artifacts_dir) / f"{index_prefix}_page_to_chunk_map.json"

    def get_rerank_candidate_pool_size(self, requested_top_k: int | None = None) -> int:
        effective_top_k = self.top_k if requested_top_k is None else requested_top_k
        base_pool = max(self.rerank_candidate_pool, self.rerank_top_k, effective_top_k)
        if self.rerank_mode == "coverage_mmr":
            return max(base_pool, self.coverage_mmr_candidate_pool, self.rerank_top_k * 6)
        return base_pool
    
    def get_config_state(self) -> None:
        """Returns dict of all config parameters except chunk_config """
        state = self.__dict__.copy()
        state.pop("chunk_config", None)
        for key in list(state.keys()):
            if not isinstance(state[key], (int, float, str, bool, list, dict, type(None))):
                state.pop(key)
        return state
