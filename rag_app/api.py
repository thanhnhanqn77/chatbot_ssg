from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any
import unicodedata

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .encoding import clean_text, repair_mojibake
from .hybrid_retriever import HybridRetriever
from .ollama_client import OllamaClient, OllamaError
from .retriever import Retriever


settings = get_settings()

NO_CONTEXT_MESSAGE = (
    "M\u00ecnh ch\u01b0a t\u00ecm th\u1ea5y ng\u1eef c\u1ea3nh ph\u00f9 h\u1ee3p "
    "trong d\u1eef li\u1ec7u hi\u1ec7n c\u00f3 \u0111\u1ec3 tr\u1ea3 l\u1eddi "
    "c\u00e2u h\u1ecfi n\u00e0y."
)


class Utf8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


app = FastAPI(
    title="SAGE Course RAG API",
    version="0.1.0",
    description="Single endpoint RAG API for SAGE curriculum and syllabus datasets.",
    default_response_class=Utf8JSONResponse,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_utf8_charset(request: Any, call_next: Any) -> Any:
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response


@app.get("/", response_class=HTMLResponse)
def read_root() -> HTMLResponse:
    static_file = Path(__file__).parent / "static" / "index.html"
    if not static_file.exists():
        raise HTTPException(status_code=404, detail="Frontend file static/index.html not found")
    return HTMLResponse(content=static_file.read_text(encoding="utf-8"))


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.rag_top_k, ge=1, le=20)


class ChatSource(BaseModel):
    source_id: str
    score: float
    id: str
    title: str | None
    doc_type: str | None
    source: str | None
    metadata: dict[str, Any]
    snippet: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChatSource]
    model: str


def _fold(value: Any) -> str:
    text = str(value or "").lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def is_study_plan_question(question: str) -> bool:
    folded = _fold(question)
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


def curriculum_codes_in_question(question: str) -> set[str]:
    return {
        match.upper()
        for match in re.findall(r"\b[A-Za-z]{2,}(?:[_-][A-Za-z0-9]+)+\b", question)
    }


def normalize_curriculum_code(code: str) -> str:
    code = code.upper().strip()
    return re.sub(r"([_-]K\d+)[A-Z]$", r"\1", code)


def cohort_codes_in_question(question: str) -> set[str]:
    folded = _fold(question)
    return {match.upper() for match in re.findall(r"\b(k\d+)\b", folded)}


def find_matching_curriculum_codes(q_codes: set[str], all_db_codes: set[str]) -> set[str]:
    matched = set()
    for q_code in q_codes:
        if q_code in all_db_codes:
            matched.add(q_code)
        else:
            norm_q = normalize_curriculum_code(q_code)
            for db_code in all_db_codes:
                if normalize_curriculum_code(db_code) == norm_q:
                    matched.add(db_code)
    return matched


def semester_numbers_in_question(question: str) -> set[str]:
    folded = _fold(question)
    return set(re.findall(r"(?:hoc ky|semester|ky|ki|k[iy])\s*(\d+)", folded))


def subject_codes_in_question(question: str) -> set[str]:
    return {
        match.upper()
        for match in re.findall(r"\b[A-Za-z]{2,}\d+[A-Za-z0-9]*\b", question)
    }


def subject_prefixes_in_question(question: str) -> set[str]:
    folded = _fold(question)
    prefixes = {
        match.upper()
        for match in re.findall(r"\b(?:mon|subject|course)\s+([a-z]{2,})(?!\d)\b", folded)
    }
    return {prefix for prefix in prefixes if prefix not in {"HOC", "NAO", "GI"}}


def is_subject_path_question(question: str) -> bool:
    folded = _fold(question)
    return any(
        term in folded
        for term in [
            "lo trinh",
            "lich hoc",
            "ke hoach hoc",
            "hoc mon",
            "mon",
            "session",
            "buoi",
        ]
    )


def is_subject_comparison_question(question: str) -> bool:
    folded = _fold(question)
    return any(
        term in folded
        for term in [
            "khac nhau",
            "so sanh",
            "compare",
            "difference",
            "different",
            "phan biet",
        ]
    )


def find_similar_subject_codes(q_code: str, db_codes: set[str]) -> list[str]:
    match = re.match(r"^([A-Z]+)(\d+.*)$", q_code.upper())
    if not match:
        return []
    prefix, suffix = match.groups()
    prefix_matches = [code for code in db_codes if code.startswith(prefix)]
    if prefix_matches:
        return sorted(prefix_matches)[:5]
    return []


def subject_codes_for_prefix(prefix: str, retriever: Retriever) -> list[str]:
    prefix = prefix.upper()
    codes = {
        str((row.get("metadata") or {}).get("subject_code") or "").upper()
        for row in retriever.documents
    }
    return sorted(code for code in codes if code.startswith(prefix))


def subject_option_rows(codes: list[str], retriever: Retriever) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code in codes:
        overview = next(
            (
                row
                for row in retriever.documents
                if row.get("doc_type") == "syllabus_overview"
                and str((row.get("metadata") or {}).get("subject_code") or "").upper() == code
            ),
            None,
        )
        if overview is None:
            overview = next(
                (
                    row
                    for row in retriever.documents
                    if str((row.get("metadata") or {}).get("subject_code") or "").upper() == code
                ),
                None,
            )
        if overview is not None:
            rows.append(_result_for_doc(overview, 1.0))
    return rows


def major_alias_groups() -> list[tuple[list[str], list[str]]]:
    return [
        (
            ["ai", "artificial intelligence", "tri tue nhan tao", "ttnt", "ad", "bcs_ad"],
            ["artificial intelligence", "tri tue nhan tao"],
        ),
        (
            ["data science", "khoa hoc du lieu", "khdl"],
            ["data science", "khoa hoc du lieu"],
        ),
        (
            ["cybersecurity", "an ninh mang", "an toan so"],
            ["cybersecurity", "an ninh mang", "an toan so"],
        ),
        (
            ["information technology", "cong nghe thong tin", "cntt"],
            ["information technology", "cong nghe thong tin", "cntt"],
        ),
        (
            ["digital marketing", "marketing so"],
            ["digital marketing", "marketing so"],
        ),
        (
            ["marketing"],
            ["marketing"],
        ),
        (
            ["international business", "kinh doanh quoc te"],
            ["international business", "kinh doanh quoc te"],
        ),
        (
            ["finance", "tai chinh"],
            ["finance", "tai chinh"],
        ),
        (
            ["hotel", "khach san"],
            ["hotel", "khach san"],
        ),
        (
            ["tourism", "du lich", "travel"],
            ["tourism", "du lich", "travel"],
        ),
        (
            ["logistics", "supply chain", "chuoi cung ung"],
            ["logistics", "supply chain", "chuoi cung ung"],
        ),
        (
            ["fintech", "cong nghe tai chinh"],
            ["fintech", "cong nghe tai chinh"],
        ),
    ]


def major_terms_for_query(question: str) -> list[str]:
    folded = _fold(question)
    for triggers, doc_terms in major_alias_groups():
        for trigger in triggers:
            if trigger == "ai":
                if re.search(r"(?:\b|_)ai(?:\b|_)", folded):
                    return doc_terms
            elif trigger == "ad":
                if re.search(r"(?:\b|_)ad(?:\b|_)", folded):
                    return doc_terms
            elif trigger in folded:
                return doc_terms
    return []


def row_matches_major(row: dict[str, Any], major_terms: list[str]) -> bool:
    if not major_terms:
        return False
    metadata = row.get("metadata") or {}
    code = str(metadata.get("curriculum_code") or "").upper()
    haystack = _fold(
        "\n".join(
            [
                str(row.get("title") or ""),
                str(metadata.get("curriculum_name") or ""),
            ]
        )
    )
    if "artificial intelligence" in major_terms:
        return (
            "artificial intelligence" in haystack
            or "tri tue nhan tao" in haystack
            or "_AI" in code
            or code.startswith("BCS_AD")
        )
    return any(term in haystack for term in major_terms)


def major_match_score(row: dict[str, Any], major_terms: list[str]) -> float:
    metadata = row.get("metadata") or {}
    code = str(metadata.get("curriculum_code") or "").upper()
    haystack = _fold(
        "\n".join(
            [
                str(row.get("title") or ""),
                str(metadata.get("curriculum_name") or ""),
            ]
        )
    )
    if "artificial intelligence" in major_terms:
        score = 0.0
        if code.startswith("BIT_AI") or "_AI_" in code:
            score += 4.0
        if code.startswith("BCS_AD"):
            score += 3.5
        if "artificial intelligence and data" in haystack or "tri tue nhan tao va khoa hoc du lieu" in haystack:
            score += 3.0
        if "artificial intelligence" in haystack or "tri tue nhan tao" in haystack:
            score += 2.5
        if "robotics" in haystack or "robot" in haystack:
            score -= 0.75
        return score
    return sum(1.0 for term in major_terms if term in haystack)


def matching_curriculum_overviews(question: str, retriever: Retriever) -> list[dict[str, Any]]:
    major_terms = major_terms_for_query(question)
    if not major_terms:
        return []
    folded_query = _fold(question)
    rows = [
        row
        for row in retriever.documents
        if row.get("doc_type") == "curriculum_overview" and row_matches_major(row, major_terms)
    ]
    if "artificial intelligence" in major_terms and not any(
        term in folded_query for term in ["robot", "robotics", "uav", "humanoid"]
    ):
        rows = [
            row
            for row in rows
            if "robot" not in _fold((row.get("title") or "") + "\n" + str((row.get("metadata") or {}).get("curriculum_name") or ""))
        ]
    rows = sorted(rows, key=lambda row: major_match_score(row, major_terms), reverse=True)
    
    q_codes = curriculum_codes_in_question(question)
    if q_codes:
        db_codes = {str((row.get("metadata") or {}).get("curriculum_code") or "") for row in rows}
        matched_codes = find_matching_curriculum_codes(q_codes, db_codes)
        rows = [
            row
            for row in rows
            if str((row.get("metadata") or {}).get("curriculum_code") or "") in matched_codes
        ]

    q_cohorts = cohort_codes_in_question(question)
    if q_cohorts:
        rows = [
            row
            for row in rows
            if str((row.get("metadata") or {}).get("cohort") or "").upper() in q_cohorts
        ]
        
    return dedupe_curriculum_families([_result_for_doc(row, 0.9) for row in rows], limit=6)


def is_business_admin_query(question: str) -> bool:
    folded = _fold(question)
    return any(
        term in folded
        for term in [
            "business administration",
            "business adminstration",
            "quan tri kinh doanh",
            "qtkd",
        ]
    )


def is_business_admin_doc(row: dict[str, Any]) -> bool:
    folded = _fold((row.get("title") or "") + "\n" + (row.get("text") or ""))
    return any(
        term in folded
        for term in [
            "business administration",
            "business adminstration",
            "quan tri kinh doanh",
            "qtkd",
        ]
    )


def has_specific_major(question: str) -> bool:
    return bool(major_terms_for_query(question))


def major_terms_for_question(question: str) -> list[str]:
    folded = _fold(question)
    groups = [
        (["digital marketing"], ["digital marketing", "marketing so"]),
        (["marketing"], ["marketing"]),
        (["international business", "kinh doanh quoc te"], ["international business", "kinh doanh quoc te"]),
        (["finance", "tai chinh"], ["finance", "tai chinh"]),
        (["hotel", "khach san"], ["hotel", "khach san"]),
        (["tourism", "du lich", "travel"], ["tourism", "du lich", "travel"]),
        (["logistics", "supply chain", "chuoi cung ung"], ["logistics", "supply chain", "chuoi cung ung"]),
        (["fintech", "cong nghe tai chinh"], ["fintech", "cong nghe tai chinh"]),
    ]
    for triggers, doc_terms in groups:
        if any(term in folded for term in triggers):
            return doc_terms
    return []


def doc_matches_major(row: dict[str, Any], question: str) -> bool:
    terms = major_terms_for_question(question)
    if not terms:
        return False
    metadata = row.get("metadata") or {}
    folded = _fold(
        "\n".join(
            [
                str(row.get("title") or ""),
                str(metadata.get("curriculum_name") or ""),
            ]
        )
    )
    return any(term in folded for term in terms)


def curriculum_family_key(code: str) -> str:
    match = re.match(r"(.+?)_K\d", code or "")
    return match.group(1) if match else code


def dedupe_curriculum_families(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        code = str((row.get("metadata") or {}).get("curriculum_code") or "")
        family = curriculum_family_key(code)
        if not family or family in seen:
            continue
        seen.add(family)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


@lru_cache(maxsize=1)
def get_retriever() -> Retriever | HybridRetriever:
    if settings.retrieval_backend == "hybrid":
        try:
            return HybridRetriever(
                settings.index_dir,
                embedding_model=settings.embedding_model,
                reranker_model=settings.reranker_model,
                embedding_device=settings.embedding_device,
                dense_weight=settings.hybrid_dense_weight,
                sparse_weight=settings.hybrid_sparse_weight,
                candidate_k=settings.hybrid_candidate_k,
                rerank_k=settings.hybrid_rerank_k,
                use_reranker=settings.hybrid_use_reranker,
            )
        except FileNotFoundError:
            pass
    return Retriever(settings.index_dir)


def get_ollama() -> OllamaClient:
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
        api_key=settings.ollama_api_key,
    )


def _snippet(text: str, limit: int = 700) -> str:
    text = " ".join(clean_text(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def source_payload(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for idx, item in enumerate(results, start=1):
        sources.append(
            {
                "source_id": f"S{idx}",
                "score": item["score"],
                "id": item["id"],
                "title": clean_text(item.get("title") or ""),
                "doc_type": item.get("doc_type"),
                "source": item.get("source"),
                "metadata": item.get("metadata") or {},
                "snippet": _snippet(item.get("text") or ""),
            }
        )
    return sources


def build_messages(question: str, results: list[dict[str, Any]]) -> list[dict[str, str]]:
    context_blocks = []
    for idx, item in enumerate(results, start=1):
        metadata = ", ".join(
            f"{key}={value}"
            for key, value in (item.get("metadata") or {}).items()
            if value not in (None, "")
        )
        context_blocks.append(
            "\n".join(
                [
                    f"[S{idx}] {item.get('title') or item.get('id')}",
                    f"Type: {item.get('doc_type')}; Source: {item.get('source')}; Metadata: {metadata}",
                    clean_text(item.get("text") or ""),
                ]
            )
        )

    system_prompt = (
        "You are a Vietnamese RAG assistant for FPT University SAGE curriculum and syllabus data. "
        "Answer only from the provided CONTEXT. If the context is not enough, say that the current data is insufficient. "
        "Reply in Vietnamese, be concise and accurate. Cite sources with [S1], [S2] when giving facts. "
        "If the context does not specify the number of credits or says they are unavailable, do not assume or say that the course has 0 credits. "
        "However, if the context explicitly states that a course has 0 credits or is non-credit (e.g. '0 tín chỉ', 'không tính tín chỉ', 'môn học điều kiện'), you must report that accurately. "
        "If asked which course is most important and the context has no explicit importance label, make a cautious "
        "inference from major relevance, project/core course signals, and semester placement; clearly say it is an inference. "
        "If a question matches multiple curricula but the provided context contains the requested semester or facts "
        "for each matched curriculum, answer for each curriculum instead of refusing. Ask the user to specify a "
        "curriculum code or cohort only when the context is insufficient or the answer would otherwise be ambiguous."
    )
    user_prompt = (
        "CONTEXT:\n"
        + "\n\n---\n\n".join(context_blocks)
        + "\n\nQUESTION:\n"
        + question
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def normalize_question(question: str) -> str:
    return clean_text(repair_mojibake(question)).strip()


def _semester_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    metadata = item.get("metadata") or {}
    semester = str(metadata.get("semester") or "")
    try:
        semester_no = int(semester)
    except ValueError:
        semester_no = 999
    return semester_no, item.get("id") or ""


def _session_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    metadata = item.get("metadata") or {}
    try:
        session_start = int(metadata.get("session_start") or 999)
    except (TypeError, ValueError):
        session_start = 999
    return session_start, item.get("id") or ""


def _result_for_doc(row: dict[str, Any], score: float = 1.0) -> dict[str, Any]:
    return {
        "score": score,
        "id": row["id"],
        "source": row.get("source"),
        "doc_type": row.get("doc_type"),
        "title": row.get("title"),
        "metadata": row.get("metadata") or {},
        "text": row.get("text") or "",
    }


def semester_major_results(question: str, retriever: Retriever) -> list[dict[str, Any]] | None:
    semesters = sorted(semester_numbers_in_question(question))
    if not semesters:
        return None
    if not major_terms_for_query(question):
        return None

    semester = semesters[0]
    overviews = matching_curriculum_overviews(question, retriever)
    if not overviews:
        return None

    selected: list[dict[str, Any]] = []
    for overview in overviews[:4]:
        code = str((overview.get("metadata") or {}).get("curriculum_code") or "")
        semester_rows = [
            row
            for row in retriever.documents
            if (row.get("metadata") or {}).get("curriculum_code") == code
            and str((row.get("metadata") or {}).get("semester") or "") == semester
            and row.get("doc_type") in {"curriculum_semester", "curriculum_subject"}
        ]
        if not semester_rows:
            continue

        selected.append(overview)
        semester_docs = [
            _result_for_doc(row, 1.0)
            for row in semester_rows
            if row.get("doc_type") == "curriculum_semester"
        ]
        subject_docs = sorted(
            [
                _result_for_doc(row, 0.9)
                for row in semester_rows
                if row.get("doc_type") == "curriculum_subject"
            ],
            key=lambda item: str((item.get("metadata") or {}).get("subject_code") or ""),
        )
        selected.extend(semester_docs[:1])
        selected.extend(subject_docs[:8])

    return selected or overviews


def _subject_display_name(title: str) -> str:
    if " / " in title:
        title = title.split(" / ", 1)[1]
    title = re.sub(r"^[A-Z]{2,}\d+[A-Z0-9]*\s*-\s*", "", title)
    return title


def _important_subjects(subjects: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    scored: list[tuple[int, dict[str, Any], str]] = []
    for subject in subjects:
        title = _fold(subject.get("title") or "")
        code = str((subject.get("metadata") or {}).get("subject_code") or "")
        score = 0
        reason = "liên quan trực tiếp tới chuyên ngành"
        if any(term in title for term in ["project", "du an"]):
            score = 100
            reason = "môn dự án/tích hợp, dùng để áp dụng kiến thức chuyên ngành"
        elif any(term in title for term in ["deep learning", "hoc sau"]):
            score = 95
            reason = "môn lõi của AI hiện đại"
        elif any(term in title for term in ["quantum machine learning", "luong tu hoc may"]):
            score = 76
            reason = "môn AI nâng cao/chuyên sâu"
        elif any(term in title for term in ["machine learning", "hoc may"]):
            score = 92
            reason = "môn nền tảng quan trọng của AI"
        elif any(term in title for term in ["computer vision", "thi giac may tinh"]):
            score = 82
            reason = "môn chuyên ngành AI ứng dụng"
        elif any(term in title for term in ["data mining", "khai pha du lieu"]):
            score = 78
            reason = "môn nền tảng xử lý/khai thác dữ liệu cho AI"
        elif any(term in title for term in ["software engineering", "ky thuat phan mem"]):
            score = 45
            reason = "môn hỗ trợ xây dựng hệ thống phần mềm"
        elif code.startswith(("JPD", "SSG")):
            score = 20
            reason = "môn kỹ năng/ngôn ngữ hỗ trợ"
        scored.append((score, subject, reason))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(subject, reason) for score, subject, reason in scored[:2] if score > 0]


def semester_major_direct_response(question: str, retriever: Retriever) -> dict[str, Any] | None:
    results = semester_major_results(question, retriever)
    if not results:
        return None

    folded = _fold(question)
    asks_courses = any(term in folded for term in ["mon nao", "nhung mon", "cac mon", "hoc mon"])
    asks_important = any(term in folded for term in ["quan trong", "trong tam", "important", "core"])
    if not (asks_courses or asks_important):
        return None

    grouped: dict[str, dict[str, Any]] = {}
    for item in results:
        metadata = item.get("metadata") or {}
        code = str(metadata.get("curriculum_code") or "")
        if not code:
            continue
        group = grouped.setdefault(code, {"overview": None, "semester": None, "subjects": []})
        if item.get("doc_type") == "curriculum_overview":
            group["overview"] = item
        elif item.get("doc_type") == "curriculum_semester":
            group["semester"] = item
        elif item.get("doc_type") == "curriculum_subject":
            group["subjects"].append(item)

    semesters = sorted(semester_numbers_in_question(question))
    semester_str = semesters[0] if semesters else "?"

    if len(grouped) == 1:
        lines = [
            f"Dưới đây là danh sách môn học Học kỳ {semester_str} của chương trình **{next(iter(grouped.keys()))}**:",
        ]
    else:
        if curriculum_codes_in_question(question) or cohort_codes_in_question(question):
            lines = [
                f"Dưới đây là danh sách môn học Học kỳ {semester_str} của các chương trình phù hợp:",
            ]
        else:
            lines = [
                "Mình tìm thấy các chương trình phù hợp với ngành AI trong dữ liệu. Vì bạn chưa nêu mã chương trình/khoá cụ thể, mình liệt kê theo từng chương trình phù hợp:",
            ]

    for code, group in grouped.items():
        overview = group["overview"] or {}
        metadata = (overview.get("metadata") or {}) if overview else {}
        cohort = metadata.get("cohort") or ((group["semester"] or {}).get("metadata") or {}).get("cohort")
        title = clean_text(overview.get("title") or code)
        subjects = sorted(
            group["subjects"],
            key=lambda item: str((item.get("metadata") or {}).get("subject_code") or ""),
        )
        lines.append("")
        lines.append(f"**{code}**" + (f" ({cohort})" if cohort else ""))
        lines.append(f"{title}")
        lines.append(f"Các môn học kỳ {semester_str}:")
        for subject in subjects:
            subject_code = (subject.get("metadata") or {}).get("subject_code")
            lines.append(f"- **{subject_code}** - {_subject_display_name(clean_text(subject.get('title') or ''))}")

        if asks_important:
            important = _important_subjects(subjects)
            if important:
                lines.append("Môn quan trọng nhất nên ưu tiên:")
                for subject, reason in important:
                    subject_code = (subject.get("metadata") or {}).get("subject_code")
                    lines.append(
                        f"- **{subject_code}** - {_subject_display_name(clean_text(subject.get('title') or ''))}: {reason}."
                    )
                lines.append("Lưu ý: phần 'quan trọng nhất' là suy luận từ tên môn/vai trò môn trong học kỳ, vì dữ liệu không gắn nhãn chính thức môn nào quan trọng nhất.")

    return {
        "question": question,
        "answer": "\n".join(lines),
        "sources": source_payload(results),
        "model": settings.ollama_model,
    }


def subject_learning_path_results(question: str, retriever: Retriever) -> list[dict[str, Any]] | None:
    codes = sorted(subject_codes_in_question(question))
    if not codes:
        return None
    if not is_subject_path_question(question) and _fold(question).strip() not in {_fold(code) for code in codes}:
        return None

    selected_code = codes[0]
    subject_rows = [
        row
        for row in retriever.documents
        if str((row.get("metadata") or {}).get("subject_code") or "").upper() == selected_code
    ]
    if not subject_rows:
        return None

    overviews = [
        _result_for_doc(row, 1.0)
        for row in subject_rows
        if row.get("doc_type") == "syllabus_overview"
    ]
    schedules = sorted(
        [
            _result_for_doc(row, 0.95)
            for row in subject_rows
            if row.get("doc_type") == "syllabus_schedule"
        ],
        key=_session_sort_key,
    )
    learning_outcomes = [
        _result_for_doc(row, 0.85)
        for row in subject_rows
        if row.get("doc_type") == "syllabus_learning_outcomes"
    ]
    curriculum_subjects = sorted(
        [
            _result_for_doc(row, 0.75)
            for row in subject_rows
            if row.get("doc_type") == "curriculum_subject"
        ],
        key=_semester_sort_key,
    )

    if schedules:
        return overviews[:1] + schedules[:6] + learning_outcomes[:1]
    return overviews[:1] + learning_outcomes[:1] + curriculum_subjects[:6]


def subject_comparison_results(question: str, retriever: Retriever) -> list[dict[str, Any]] | None:
    codes = sorted(subject_codes_in_question(question))
    if len(codes) < 2 or not is_subject_comparison_question(question):
        return None

    selected: list[dict[str, Any]] = []
    wanted_types = [
        ("syllabus_overview", 1.0),
        ("syllabus_learning_outcomes", 0.92),
        ("syllabus_assessments", 0.88),
        ("syllabus_materials", 0.78),
    ]
    for code in codes[:4]:
        subject_rows = [
            row
            for row in retriever.documents
            if str((row.get("metadata") or {}).get("subject_code") or "").upper() == code
        ]
        for doc_type, score in wanted_types:
            match = next((row for row in subject_rows if row.get("doc_type") == doc_type), None)
            if match is not None:
                selected.append(_result_for_doc(match, score))
    return selected or None


def subject_clarification_response(question: str, retriever: Retriever) -> dict[str, Any] | None:
    if subject_codes_in_question(question):
        return None

    prefixes = sorted(subject_prefixes_in_question(question))
    if not prefixes:
        return None

    prefix = prefixes[0]
    codes = subject_codes_for_prefix(prefix, retriever)
    if len(codes) <= 1:
        return None

    option_rows = subject_option_rows(codes[:8], retriever)
    options = []
    for row in option_rows:
        metadata = row.get("metadata") or {}
        options.append(f"- {metadata.get('subject_code')}: {clean_text(row.get('title') or '')}")

    answer = (
        f"Mình thấy bạn đang hỏi về nhóm mã môn {prefix}, nhưng có nhiều môn phù hợp. "
        "Bạn muốn hỏi môn nào?\n"
        + "\n".join(options)
        + "\n\nBạn chỉ cần trả lời bằng mã môn, ví dụ: "
        + ", ".join(codes[:3])
        + "."
    )
    return {
        "question": question,
        "answer": answer,
        "sources": source_payload(option_rows),
        "model": settings.ollama_model,
    }


def _curriculum_name(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    return clean_text(str(metadata.get("curriculum_name") or result.get("title") or ""))


def _same_program_family(left: str, right: str) -> bool:
    left_folded = _fold(left)
    right_folded = _fold(right)
    family_terms = ["business adminstration", "business administration", "quan tri kinh doanh"]
    return any(term in left_folded and term in right_folded for term in family_terms)


def plan_results(question: str, retriever: Retriever) -> list[dict[str, Any]] | None:
    if not is_study_plan_question(question):
        return None

    codes = curriculum_codes_in_question(question)
    if codes:
        db_codes = {
            str((row.get("metadata") or {}).get("curriculum_code") or "")
            for row in retriever.documents
            if row.get("doc_type") == "curriculum_overview"
        }
        matched_codes = find_matching_curriculum_codes(codes, db_codes)
        rows = [
            _result_for_doc(row, 1.0)
            for row in retriever.documents
            if str((row.get("metadata") or {}).get("curriculum_code") or "") in matched_codes
            and row.get("doc_type") in {"curriculum_overview", "curriculum_semester"}
        ]
        overviews = [row for row in rows if row.get("doc_type") == "curriculum_overview"]
        semesters = sorted(
            [row for row in rows if row.get("doc_type") == "curriculum_semester"],
            key=_semester_sort_key,
        )
        return overviews[:1] + semesters[:12]

    if has_specific_major(question):
        major_overviews = [
            _result_for_doc(row, 0.9)
            for row in retriever.documents
            if row.get("doc_type") == "curriculum_overview" and doc_matches_major(row, question)
        ]
        q_cohorts = cohort_codes_in_question(question)
        if q_cohorts:
            major_overviews = [
                row
                for row in major_overviews
                if str((row.get("metadata") or {}).get("cohort") or "").upper() in q_cohorts
            ]
        major_families = dedupe_curriculum_families(major_overviews, limit=8)
        if len(major_families) == 1:
            selected_code = str((major_families[0].get("metadata") or {}).get("curriculum_code") or "")
            rows = [
                _result_for_doc(row, 1.0)
                for row in retriever.documents
                if (row.get("metadata") or {}).get("curriculum_code") == selected_code
                and row.get("doc_type") in {"curriculum_overview", "curriculum_semester"}
            ]
            overviews = [row for row in rows if row.get("doc_type") == "curriculum_overview"]
            semesters = sorted(
                [row for row in rows if row.get("doc_type") == "curriculum_semester"],
                key=_semester_sort_key,
            )
            return overviews[:1] + semesters[:12]
        if major_families:
            return major_families

    if is_business_admin_query(question) and not has_specific_major(question):
        business_overviews = [
            _result_for_doc(row, 0.8)
            for row in retriever.documents
            if row.get("doc_type") == "curriculum_overview" and is_business_admin_doc(row)
        ]
        return dedupe_curriculum_families(business_overviews, limit=8)

    candidates = retriever.search(question, top_k=20)
    overview_candidates = [item for item in candidates if item.get("doc_type") == "curriculum_overview"]
    semester_candidates = [item for item in candidates if item.get("doc_type") == "curriculum_semester"]

    if is_business_admin_query(question):
        overview_candidates = [item for item in overview_candidates if is_business_admin_doc(item)]
        semester_candidates = [item for item in semester_candidates if is_business_admin_doc(item)]

    if not overview_candidates:
        overview_candidates = [
            _result_for_doc(row, 0.5)
            for row in retriever.documents
            if row.get("doc_type") == "curriculum_overview"
            and (
                is_business_admin_doc(row)
                if is_business_admin_query(question)
                else "business" in _fold(row.get("text") or row.get("title") or "")
            )
        ][:8]

    if len(overview_candidates) >= 2:
        first_name = _curriculum_name(overview_candidates[0])
        different_family = [
            item
            for item in overview_candidates[1:]
            if not _same_program_family(first_name, _curriculum_name(item))
        ]
        if different_family or "major" not in _fold(question):
            return overview_candidates[:8]

    if semester_candidates:
        top_code = (semester_candidates[0].get("metadata") or {}).get("curriculum_code")
        rows = [
            _result_for_doc(row, 1.0)
            for row in retriever.documents
            if (row.get("metadata") or {}).get("curriculum_code") == top_code
            and row.get("doc_type") in {"curriculum_overview", "curriculum_semester"}
        ]
        overviews = [row for row in rows if row.get("doc_type") == "curriculum_overview"]
        semesters = sorted(
            [row for row in rows if row.get("doc_type") == "curriculum_semester"],
            key=_semester_sort_key,
        )
        return overviews[:1] + semesters[:12]

    return candidates


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict[str, Any]:
    question = normalize_question(request.question)
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    retriever = get_retriever()
    try:
        clarification = subject_clarification_response(question, retriever)
        if clarification is not None:
            return clarification

        # Suggestions for non-existent subject codes
        q_subject_codes = subject_codes_in_question(question)
        if q_subject_codes:
            db_subject_codes = {
                str((row.get("metadata") or {}).get("subject_code") or "").upper()
                for row in retriever.documents
                if (row.get("metadata") or {}).get("subject_code")
            }
            missing_codes = [c for c in q_subject_codes if c not in db_subject_codes]
            if missing_codes:
                suggestions = []
                for code in missing_codes[:2]:
                    sims = find_similar_subject_codes(code, db_subject_codes)
                    if sims:
                        suggestions.append(
                            f"Môn **{code}** không có trong dữ liệu hiện tại, nhưng có nhóm môn tương tự: "
                            + ", ".join(f"**{s}**" for s in sims)
                            + "."
                        )
                if suggestions:
                    answer = (
                        "Mình không tìm thấy thông tin chi tiết cho môn học bạn yêu cầu.\n\n"
                        + "\n".join(suggestions)
                        + "\n\nBạn có muốn tìm học phần hoặc lộ trình của các môn học trên không?"
                    )
                    sim_codes_set = set()
                    for code in missing_codes[:2]:
                        sim_codes_set.update(find_similar_subject_codes(code, db_subject_codes))
                    
                    option_rows = []
                    for code in sorted(sim_codes_set)[:4]:
                        overview = next(
                            (
                                row
                                for row in retriever.documents
                                if row.get("doc_type") == "syllabus_overview"
                                and str((row.get("metadata") or {}).get("subject_code") or "").upper() == code
                            ),
                            None,
                        )
                        if overview:
                            option_rows.append(_result_for_doc(overview, 0.9))
                    return {
                        "question": question,
                        "answer": answer,
                        "sources": source_payload(option_rows),
                        "model": settings.ollama_model,
                    }

        semester_major_answer = semester_major_direct_response(question, retriever)
        if semester_major_answer is not None:
            return semester_major_answer

        results = subject_comparison_results(question, retriever)
        if results is None:
            results = subject_learning_path_results(question, retriever)
        if results is None:
            results = semester_major_results(question, retriever)
        if results is None:
            results = plan_results(question, retriever)
        if results is None:
            results = retriever.search(question, top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = source_payload(results)
    if not results:
        return {
            "question": question,
            "answer": NO_CONTEXT_MESSAGE,
            "sources": [],
            "model": settings.ollama_model,
        }

    messages = build_messages(question, results)
    try:
        answer = get_ollama().chat(
            messages,
            temperature=settings.rag_temperature,
        )
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "question": question,
        "answer": repair_mojibake(clean_text(answer)),
        "sources": sources,
        "model": settings.ollama_model,
    }


class JDAnalysisRequest(BaseModel):
    jd_text: str = Field(..., min_length=10)
    top_k: int = Field(default=6, ge=1, le=15)


class JDAnalysisResponse(BaseModel):
    jd_text: str
    analysis: str
    suggested_subjects: list[dict[str, Any]]
    model: str


@app.post("/analyze-jd", response_model=JDAnalysisResponse)
def analyze_jd(request: JDAnalysisRequest) -> dict[str, Any]:
    jd_text = clean_text(request.jd_text).strip()
    if not jd_text:
        raise HTTPException(status_code=422, detail="jd_text must not be empty")

    retriever = get_retriever()
    
    # Search for top matching documents based on the JD text
    search_results = retriever.search(jd_text, top_k=35)
    
    subject_docs = []
    seen_codes = set()
    for item in search_results:
        metadata = item.get("metadata") or {}
        subject_code = str(metadata.get("subject_code") or "").upper().strip()
        if not subject_code:
            continue
        if subject_code in seen_codes:
            continue
        
        # We only want subject overview, curriculum subject, or learning outcomes
        if item.get("doc_type") in {"syllabus_overview", "curriculum_subject", "syllabus_learning_outcomes"}:
            seen_codes.add(subject_code)
            subject_docs.append(item)
            if len(subject_docs) >= request.top_k:
                break
                
    # Fallback to general syllabus overviews if not enough found
    if len(subject_docs) < 3:
        for row in retriever.documents:
            if row.get("doc_type") == "syllabus_overview":
                metadata = row.get("metadata") or {}
                subject_code = str(metadata.get("subject_code") or "").upper().strip()
                if subject_code and subject_code not in seen_codes:
                    seen_codes.add(subject_code)
                    subject_docs.append(_result_for_doc(row, 0.5))
                    if len(subject_docs) >= request.top_k:
                        break

    context_blocks = []
    for idx, item in enumerate(subject_docs, start=1):
        metadata = ", ".join(
            f"{key}={value}"
            for key, value in (item.get("metadata") or {}).items()
            if value not in (None, "")
        )
        context_blocks.append(
            "\n".join(
                [
                    f"[Subject {idx}] Code: {item.get('metadata', {}).get('subject_code') or item.get('id')}; Title: {item.get('title')}",
                    f"Type: {item.get('doc_type')}; Metadata: {metadata}",
                    clean_text(item.get("text") or ""),
                ]
            )
        )
        
    system_prompt = (
        "You are an expert academic advisor and career counselor at FPT University. "
        "Analyze the provided Job Description (JD) and suggest relevant courses from the CONTEXT. "
        "Explain in detail how each suggested course matches the skills, technologies, or concepts required in the JD. "
        "Highlight the importance of these courses for the job. "
        "Write a structured, professional report in Vietnamese. Use markdown tables, lists, and bold text for readability. "
        "Do not invent facts or courses not present in the context. "
    )
    user_prompt = (
        "JOB DESCRIPTION (JD):\n"
        + jd_text
        + "\n\n"
        + "CONTEXT COURSES:\n"
        + "\n\n---\n\n".join(context_blocks)
        + "\n\n"
        + "Write a comprehensive analysis matching courses to the JD."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        answer = get_ollama().chat(
            messages,
            temperature=settings.rag_temperature,
        )
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "jd_text": jd_text,
        "analysis": repair_mojibake(clean_text(answer)),
        "suggested_subjects": source_payload(subject_docs),
        "model": settings.ollama_model,
    }

