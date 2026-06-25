from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .hybrid_index import BGEM3Embedder, BGEReranker, searchable_text
from .retriever import (
    _intent_boost,
    _matches_filters,
    _normalize_code,
    _query_codes,
    _query_curriculum_codes,
    _query_semesters,
    _wants_program_spread,
)


JsonDict = dict[str, Any]


def _load_documents(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high <= low:
        return {idx: 1.0 for idx in scores}
    return {idx: (score - low) / (high - low) for idx, score in scores.items()}


def _top_indices(scores: np.ndarray, limit: int) -> list[int]:
    if len(scores) == 0 or limit <= 0:
        return []
    limit = min(limit, len(scores))
    if limit == len(scores):
        ranked = np.argsort(scores)[::-1]
    else:
        candidates = np.argpartition(scores, -limit)[-limit:]
        ranked = candidates[np.argsort(scores[candidates])[::-1]]
    return [int(idx) for idx in ranked if float(scores[int(idx)]) > 0.0]


class HybridRetriever:
    def __init__(
        self,
        index_dir: Path,
        *,
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        embedding_device: str = "auto",
        dense_weight: float = 0.55,
        sparse_weight: float = 0.45,
        candidate_k: int = 80,
        rerank_k: int = 40,
        use_reranker: bool = True,
    ) -> None:
        self.index_dir = index_dir
        docs_path = index_dir / "documents.jsonl"
        dense_path = index_dir / "dense.npy"
        hybrid_path = index_dir / "hybrid.pkl"
        if not docs_path.exists() or not dense_path.exists() or not hybrid_path.exists():
            raise FileNotFoundError(
                f"Hybrid RAG index not found in {index_dir}. "
                "Run: python scripts/build_hybrid_index.py"
            )

        self.documents = _load_documents(docs_path)
        self.dense_matrix = np.load(dense_path).astype(np.float32)
        with hybrid_path.open("rb") as file:
            payload = pickle.load(file)
        self.sparse_index: dict[str, list[tuple[int, float]]] = payload["sparse_index"]
        self.doc_ids = payload["doc_ids"]
        self.embedding_model = payload.get("model_name") or embedding_model
        self.reranker_model = reranker_model
        self.embedding_device = embedding_device
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.use_reranker = use_reranker
        self._embedder: BGEM3Embedder | None = None
        self._reranker: BGEReranker | None = None

    @property
    def count(self) -> int:
        return len(self.documents)

    @property
    def embedder(self) -> BGEM3Embedder:
        if self._embedder is None:
            self._embedder = BGEM3Embedder(self.embedding_model, device=self.embedding_device)
        return self._embedder

    @property
    def reranker(self) -> BGEReranker:
        if self._reranker is None:
            self._reranker = BGEReranker(self.reranker_model)
        return self._reranker

    def _sparse_scores(self, query_sparse: dict[str, float]) -> dict[int, float]:
        scores: dict[int, float] = {}
        for term, query_weight in query_sparse.items():
            for idx, doc_weight in self.sparse_index.get(term, []):
                scores[idx] = scores.get(idx, 0.0) + float(query_weight) * float(doc_weight)
        return scores

    def _metadata_boost(self, idx: int, query: str) -> float:
        row = self.documents[idx]
        metadata = row.get("metadata") or {}
        doc_type = str(row.get("doc_type") or "")
        subject_code = _normalize_code(metadata.get("subject_code"))
        curriculum_code = _normalize_code(metadata.get("curriculum_code"))
        title = _normalize_code(row.get("title"))
        boost = _intent_boost(query, doc_type)

        semester = str(metadata.get("semester") or "").strip()
        semesters = _query_semesters(query)
        if semesters and doc_type in {"curriculum_semester", "curriculum_subject"}:
            boost += 0.55 if semester in semesters and doc_type == "curriculum_semester" else 0.0
            boost += 0.35 if semester in semesters and doc_type == "curriculum_subject" else 0.0
            boost -= 0.08 if semester not in semesters else 0.0

        for code in _query_codes(query):
            if code and code == subject_code:
                boost += 0.45 if doc_type.startswith("syllabus_") else 0.35
            elif code and code == curriculum_code:
                boost += 0.35
            elif code and code in title:
                boost += 0.12

        for code in _query_curriculum_codes(query):
            if code and code == curriculum_code:
                boost += 0.45
            elif code and code in title:
                boost += 0.16
        return boost

    def search(self, query: str, top_k: int = 8, filters: JsonDict | None = None) -> list[JsonDict]:
        query = str(query or "").strip()
        if not query:
            return []

        query_dense, query_sparse = self.embedder.encode_query(query)
        dense_scores_array = self.dense_matrix @ query_dense
        dense_top = _top_indices(dense_scores_array, self.candidate_k)
        sparse_raw = self._sparse_scores(query_sparse)
        sparse_top = [
            idx
            for idx, _ in sorted(sparse_raw.items(), key=lambda item: item[1], reverse=True)[: self.candidate_k]
            if sparse_raw[idx] > 0
        ]
        candidate_ids = set(dense_top) | set(sparse_top)
        if not candidate_ids:
            return []

        dense_scores = {idx: float(dense_scores_array[idx]) for idx in candidate_ids}
        sparse_scores = {idx: float(sparse_raw.get(idx, 0.0)) for idx in candidate_ids}
        dense_norm = _minmax(dense_scores)
        sparse_norm = _minmax(sparse_scores)

        merged: list[tuple[int, float, float, float]] = []
        for idx in candidate_ids:
            row = self.documents[idx]
            if not _matches_filters(row, filters):
                continue
            combined = (
                self.dense_weight * dense_norm.get(idx, 0.0)
                + self.sparse_weight * sparse_norm.get(idx, 0.0)
                + self._metadata_boost(idx, query)
            )
            if combined > 0:
                merged.append((idx, combined, dense_scores[idx], sparse_scores[idx]))
        if not merged:
            return []

        merged.sort(key=lambda item: item[1], reverse=True)
        rerank_pool = merged[: max(top_k, self.rerank_k)]
        rerank_scores: dict[int, float] = {}
        if self.use_reranker and rerank_pool:
            docs = [searchable_text(self.documents[idx]) for idx, *_ in rerank_pool]
            scores = self.reranker.score(query, docs)
            rerank_scores = {idx: score for (idx, *_), score in zip(rerank_pool, scores)}
            rerank_norm = _minmax(rerank_scores)
            merged = [
                (
                    idx,
                    0.35 * combined + 0.65 * rerank_norm.get(idx, 0.0),
                    dense_score,
                    sparse_score,
                )
                for idx, combined, dense_score, sparse_score in rerank_pool
            ] + merged[len(rerank_pool) :]
            merged.sort(key=lambda item: item[1], reverse=True)

        limit = max(int(top_k), 1)
        results: list[JsonDict] = []
        duplicate_counts: dict[tuple[str, str], int] = {}
        curriculum_codes = _query_curriculum_codes(query)
        wants_program_spread = _wants_program_spread(query)
        for idx, score, dense_score, sparse_score in merged:
            row = self.documents[idx]
            metadata = row.get("metadata") or {}
            doc_type = str(row.get("doc_type") or "")
            subject_code = _normalize_code(metadata.get("subject_code"))
            parent_id = str(metadata.get("parent_id") or row.get("id") or "")

            duplicate_key = (parent_id or row["id"], doc_type)
            if subject_code and doc_type == "curriculum_subject" and not curriculum_codes and not wants_program_spread:
                duplicate_key = (doc_type, subject_code)
            duplicate_counts[duplicate_key] = duplicate_counts.get(duplicate_key, 0) + 1
            if duplicate_counts[duplicate_key] > 1:
                continue

            results.append(
                {
                    "score": float(score),
                    "dense_score": float(dense_score),
                    "sparse_score": float(sparse_score),
                    "rerank_score": rerank_scores.get(idx),
                    "id": row["id"],
                    "source": row.get("source"),
                    "doc_type": row.get("doc_type"),
                    "title": row.get("title"),
                    "metadata": row.get("metadata") or {},
                    "text": row.get("text") or "",
                }
            )
            if len(results) >= limit:
                break
        return results
