from __future__ import annotations

import json
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np


JsonDict = dict[str, Any]


def _load_documents(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_code(value: Any) -> str:
    return str(value or "").upper().strip()


def _query_codes(query: str) -> set[str]:
    return {match.upper() for match in re.findall(r"\b[A-Za-z]{2,}\d+[A-Za-z0-9]*\b", query)}


def _query_curriculum_codes(query: str) -> set[str]:
    return {
        match.upper()
        for match in re.findall(r"\b[A-Za-z]{2,}(?:[_-][A-Za-z0-9]+)+\b", query)
    }


def _normalize_curriculum_code(code: str) -> str:
    code = code.upper().strip()
    return re.sub(r"([_-]K\d+)[A-Z]$", r"\1", code)


def _query_semesters(query: str) -> set[str]:
    folded = _fold(query)
    semesters = set(re.findall(r"(?:hoc ky|semester|ky)\s*(\d+)", folded))
    return semesters


def _fold(value: Any) -> str:
    text = str(value or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def _intent_boost(query: str, doc_type: str) -> float:
    folded = _fold(query)
    boost = 0.0

    if _is_curriculum_plan_query(query):
        if doc_type == "curriculum_semester":
            boost += 0.75
        elif doc_type == "curriculum_overview":
            boost += 0.55
        elif doc_type == "curriculum_subject":
            boost -= 0.12

    if any(term in folded for term in ["tin chi", "credit", "so tin", "tien quyet", "prerequisite"]):
        if doc_type == "syllabus_overview":
            boost += 0.22
        elif doc_type == "curriculum_subject":
            boost += 0.08

    if any(term in folded for term in ["danh gia", "trong so", "final", "midterm", "exam", "thi"]):
        if doc_type == "syllabus_assessments":
            boost += 0.28

    if any(term in folded for term in ["lich", "session", "buoi", "tuan", "schedule"]):
        if doc_type == "syllabus_schedule":
            boost += 0.24

    if any(term in folded for term in ["tai lieu", "sach", "material", "book"]):
        if doc_type == "syllabus_materials":
            boost += 0.24

    if any(term in folded for term in ["clo", "learning outcome", "chuan dau ra", "muc tieu"]):
        if doc_type == "syllabus_learning_outcomes":
            boost += 0.24
        elif doc_type == "curriculum_overview":
            boost += 0.08

    if any(term in folded for term in ["hoc ky", "semester", "ky may", "mon nao"]):
        if doc_type == "curriculum_semester":
            boost += 0.22
        elif doc_type == "curriculum_subject":
            boost += 0.12

    return boost


def _wants_program_spread(query: str) -> bool:
    folded = _fold(query)
    return any(term in folded for term in ["chuong trinh nao", "cac chuong trinh", "program", "curriculum", "lo trinh", "nganh"])


def _is_curriculum_plan_query(query: str) -> bool:
    folded = _fold(query)
    return any(
        term in folded
        for term in [
            "lo trinh",
            "ke hoach hoc",
            "chuong trinh hoc",
            "khung chuong trinh",
            "study plan",
            "roadmap",
            "learning path",
        ]
    )


def _matches_filters(row: JsonDict, filters: JsonDict | None) -> bool:
    if not filters:
        return True

    metadata = row.get("metadata") or {}
    for key, expected in filters.items():
        if expected in (None, "", [], {}):
            continue

        actual = row.get(key, metadata.get(key))
        if isinstance(expected, list):
            expected_set = {_normalize_code(item) for item in expected}
            if _normalize_code(actual) not in expected_set:
                return False
        elif _normalize_code(actual) != _normalize_code(expected):
            return False

    return True


class Retriever:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        docs_path = index_dir / "documents.jsonl"
        index_path = index_dir / "tfidf.pkl"

        if not docs_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                f"RAG index not found in {index_dir}. Run: python -m rag_app.prepare"
            )

        self.documents = _load_documents(docs_path)
        with index_path.open("rb") as file:
            try:
                payload = pickle.load(file)
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "RAG index cannot be loaded with the current Python packages. "
                    "Install the pinned requirements or rebuild the index with "
                    "python scripts/prepare_rag.py."
                ) from exc

        self.vectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]
        self.doc_ids = payload["doc_ids"]
        self.versions = payload.get("versions") or {}

    @property
    def count(self) -> int:
        return len(self.documents)

    def search(self, query: str, top_k: int = 8, filters: JsonDict | None = None) -> list[JsonDict]:
        query = str(query or "").strip()
        if not query:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        codes = _query_codes(query)
        curriculum_codes = _query_curriculum_codes(query)
        semesters = _query_semesters(query)
        wants_program_spread = _wants_program_spread(query)
        folded_query = _fold(query)
        query_cohorts = {match.upper() for match in re.findall(r"\b(k\d+)\b", folded_query)}

        for idx, row in enumerate(self.documents):
            if not _matches_filters(row, filters):
                scores[idx] = -1.0
                continue

            metadata = row.get("metadata") or {}
            subject_code = _normalize_code(metadata.get("subject_code"))
            curriculum_code = _normalize_code(metadata.get("curriculum_code"))
            title = _normalize_code(row.get("title"))
            doc_type = str(row.get("doc_type") or "")

            scores[idx] += _intent_boost(query, doc_type)

            doc_cohort = str(metadata.get("cohort") or "").upper().strip()
            if query_cohorts and doc_cohort:
                if doc_cohort in query_cohorts:
                    scores[idx] += 0.65
                else:
                    scores[idx] -= 0.50

            semester = str(metadata.get("semester") or "").strip()
            if semesters and doc_type in {"curriculum_semester", "curriculum_subject"}:
                if semester in semesters:
                    scores[idx] += 0.55 if doc_type == "curriculum_semester" else 0.35
                else:
                    scores[idx] -= 0.08

            for code in codes:
                if code and code == subject_code:
                    scores[idx] += 0.45 if doc_type.startswith("syllabus_") else 0.35
                elif code and code == curriculum_code:
                    scores[idx] += 0.35
                elif code and code in title:
                    scores[idx] += 0.12

            norm_curriculum_codes = {_normalize_curriculum_code(c) for c in curriculum_codes}
            norm_curr_code = _normalize_curriculum_code(curriculum_code)
            for code in norm_curriculum_codes:
                if code and code == norm_curr_code:
                    scores[idx] += 0.45
                elif code and code in title:
                    scores[idx] += 0.16

        valid = np.where(scores > 0)[0]
        if len(valid) == 0:
            return []

        limit = min(max(int(top_k), 1), len(valid))
        ranked = valid[np.argsort(scores[valid])[::-1]]

        results: list[JsonDict] = []
        duplicate_counts: dict[tuple[str, str], int] = {}
        for idx in ranked:
            row = self.documents[int(idx)]
            metadata = row.get("metadata") or {}
            doc_type = str(row.get("doc_type") or "")
            subject_code = _normalize_code(metadata.get("subject_code"))

            if (
                subject_code
                and doc_type == "curriculum_subject"
                and not curriculum_codes
                and not wants_program_spread
            ):
                duplicate_key = (doc_type, subject_code)
                duplicate_counts[duplicate_key] = duplicate_counts.get(duplicate_key, 0) + 1
                if duplicate_counts[duplicate_key] > 1:
                    continue

            results.append(
                {
                    "score": float(scores[int(idx)]),
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
