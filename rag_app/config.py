from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> None:
    env_path = path or ROOT_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _path_from_env(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _int_from_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_from_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    normalized_dir: Path
    index_dir: Path
    ollama_base_url: str
    ollama_model: str
    ollama_api_key: str | None
    ollama_timeout: int
    rag_top_k: int
    rag_temperature: float
    retrieval_backend: str
    embedding_model: str
    reranker_model: str
    embedding_device: str
    embedding_batch_size: int
    chunk_max_chars: int
    chunk_overlap: int
    hybrid_candidate_k: int
    hybrid_rerank_k: int
    hybrid_dense_weight: float
    hybrid_sparse_weight: float
    hybrid_use_reranker: bool
    api_host: str
    api_port: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_env()

    data_dir = _path_from_env("RAG_DATA_DIR", "data")
    normalized_dir = _path_from_env("RAG_NORMALIZED_DIR", str(data_dir / "normalized"))
    index_dir = _path_from_env("RAG_INDEX_DIR", str(data_dir / "index"))

    return Settings(
        root_dir=ROOT_DIR,
        data_dir=data_dir,
        normalized_dir=normalized_dir,
        index_dir=index_dir,
        ollama_base_url=(
            os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434"
        ).rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
        ollama_api_key=os.getenv("OLLAMA_API_KEY") or None,
        ollama_timeout=_int_from_env("RAG_OLLAMA_TIMEOUT", 120),
        rag_top_k=_int_from_env("RAG_TOP_K", 8),
        rag_temperature=_float_from_env("RAG_TEMPERATURE", 0.2),
        retrieval_backend=os.getenv("RAG_RETRIEVAL_BACKEND", "hybrid").lower(),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
        reranker_model=os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        embedding_device=os.getenv("RAG_EMBEDDING_DEVICE", "auto"),
        embedding_batch_size=_int_from_env("RAG_EMBEDDING_BATCH_SIZE", 12),
        chunk_max_chars=_int_from_env("RAG_CHUNK_MAX_CHARS", 2200),
        chunk_overlap=_int_from_env("RAG_CHUNK_OVERLAP", 250),
        hybrid_candidate_k=_int_from_env("RAG_HYBRID_CANDIDATE_K", 80),
        hybrid_rerank_k=_int_from_env("RAG_HYBRID_RERANK_K", 40),
        hybrid_dense_weight=_float_from_env("RAG_HYBRID_DENSE_WEIGHT", 0.55),
        hybrid_sparse_weight=_float_from_env("RAG_HYBRID_SPARSE_WEIGHT", 0.45),
        hybrid_use_reranker=os.getenv("RAG_HYBRID_USE_RERANKER", "true").lower()
        not in {"0", "false", "no", "off"},
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=_int_from_env("API_PORT", 8000),
    )
