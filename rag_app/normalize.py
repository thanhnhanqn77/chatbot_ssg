from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rag_app.config import get_settings
    from rag_app.encoding import clean_text, clean_value
else:
    from .config import get_settings
    from .encoding import clean_text, clean_value


JsonDict = dict[str, Any]


def load_json(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as file:
        return clean_value(json.load(file))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def scalar(value: Any) -> str:
    if not nonempty(value):
        return ""
    return clean_text(value)


def normalized_credit(value: Any) -> str:
    credit = scalar(value)
    if credit in {"0", "0.0", "0.00"}:
        return ""
    return credit


def labeled_line(label: str, value: Any, empty_text: str | None = None) -> str:
    text = scalar(value)
    if text:
        return f"{label}: {text}"
    if empty_text is not None:
        return f"{label}: {empty_text}"
    return ""


def compact_json(value: Any) -> str:
    if not nonempty(value):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def pair_lines(mapping: JsonDict, fields: list[tuple[str, str]]) -> list[str]:
    lines: list[str] = []
    for label, key in fields:
        value = mapping.get(key)
        if nonempty(value):
            lines.append(f"{label}: {scalar(value)}")
    return lines


def join_lines(lines: list[str]) -> str:
    return clean_text("\n".join(line for line in lines if nonempty(line)))


def safe_id(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9._:-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "item"


def extract_subject_code(subject: JsonDict) -> str:
    if nonempty(subject.get("subjectCode")):
        return scalar(subject["subjectCode"]).upper()

    for option in subject.get("syllabusOptions") or []:
        if nonempty(option.get("subjectCode")):
            return scalar(option["subjectCode"]).upper()

    url = scalar(subject.get("syllabusListUrl"))
    if url:
        parsed = urlparse(url)
        sub_code = parse_qs(parsed.query).get("subCode")
        if sub_code and sub_code[0]:
            return clean_text(sub_code[0]).upper()

    metadata = (subject.get("syllabusList") or {}).get("metadata") or {}
    subject_label = scalar(metadata.get("subject"))
    match = re.match(r"([A-Za-z]{2,}\d+[A-Za-z0-9]*)\b", subject_label)
    return match.group(1).upper() if match else ""


def normalize_curriculum(raw: JsonDict) -> JsonDict:
    list_info = raw.get("listInfo") or {}
    detail = raw.get("detail") or {}
    code = scalar(detail.get("curriculumCode") or list_info.get("code")).upper()

    subjects: list[JsonDict] = []
    for subject in raw.get("subjects") or []:
        options = []
        for option in subject.get("syllabusOptions") or []:
            options.append(
                {
                    "subject_code": scalar(option.get("subjectCode")).upper(),
                    "syllabus_name": scalar(option.get("syllabusName")),
                    "sylid": scalar(option.get("sylid")),
                    "url": scalar(option.get("url")),
                    "decision_no": scalar(option.get("decisionNo")),
                    "note": scalar(option.get("note")),
                    "is_active": option.get("isActive"),
                    "is_approved": option.get("isApproved"),
                }
            )

        subjects.append(
            {
                "subject_code": extract_subject_code(subject),
                "subject_name": scalar(subject.get("subjectName")),
                "semester": scalar(subject.get("semester")),
                "credits": normalized_credit(subject.get("noCredit")),
                "raw_credits": scalar(subject.get("noCredit")),
                "prerequisite": scalar(subject.get("preRequisite")),
                "syllabus_list_url": scalar(subject.get("syllabusListUrl")),
                "syllabus_options": options,
            }
        )

    return {
        "curriculum_code": code,
        "cohort": scalar(list_info.get("cohort")),
        "name": scalar(detail.get("name") or list_info.get("name")),
        "english_name": scalar(detail.get("englishName")),
        "description": scalar(detail.get("description") or list_info.get("description")),
        "decision": scalar(list_info.get("decision")),
        "total_credit": scalar(list_info.get("totalCredit")),
        "url": scalar(list_info.get("url")),
        "plos": [
            {
                "name": scalar(plo.get("ploName")),
                "description": scalar(plo.get("ploDescription")),
            }
            for plo in raw.get("plos") or []
        ],
        "subjects": subjects,
    }


def normalize_syllabus(subject_code: str, raw: JsonDict) -> JsonDict:
    detail = raw.get("detail") or {}
    code = scalar(detail.get("subjectCode") or raw.get("subjectCode") or subject_code).upper()

    return {
        "subject_code": code,
        "sylid": scalar(raw.get("sylid") or detail.get("syllabusId")),
        "url": scalar(raw.get("url")),
        "detail": {
            "syllabus_id": scalar(detail.get("syllabusId")),
            "syllabus_name": scalar(detail.get("syllabusName")),
            "syllabus_english": scalar(detail.get("syllabusEnglish")),
            "credits": normalized_credit(detail.get("noCredit")),
            "raw_credits": scalar(detail.get("noCredit")),
            "degree_level": scalar(detail.get("degreeLevel")),
            "time_allocation": scalar(detail.get("timeAllocation")),
            "prerequisite": scalar(detail.get("prerequisite")),
            "description": scalar(detail.get("description")),
            "student_tasks": scalar(detail.get("studentTasks")),
            "tools": scalar(detail.get("tools")),
            "scoring_scale": scalar(detail.get("scoringScale")),
            "decision_no": scalar(detail.get("decisionNo")),
            "is_approved": scalar(detail.get("isApproved")),
            "note": scalar(detail.get("note")),
            "min_average_mark_to_pass": scalar(detail.get("minAverageMarkToPass")),
            "is_active": scalar(detail.get("isActive")),
            "approved_date": scalar(detail.get("approveddate")),
        },
        "materials": raw.get("materials") or [],
        "learning_outcomes": raw.get("learningOutcomes") or [],
        "schedule": raw.get("schedule") or [],
        "constructive_questions": raw.get("constructiveQuestions") or [],
        "assessments": raw.get("assessments") or [],
        "provenance": raw.get("provenance") or {},
    }


class ChunkBuilder:
    def __init__(self) -> None:
        self.rows: list[JsonDict] = []
        self._ids: Counter[str] = Counter()

    def add(
        self,
        *,
        doc_id: str,
        source: str,
        doc_type: str,
        title: str,
        text: str,
        metadata: JsonDict,
    ) -> None:
        text = clean_text(text)
        if len(text) < 20:
            return

        base_id = safe_id(doc_id)
        self._ids[base_id] += 1
        final_id = base_id if self._ids[base_id] == 1 else f"{base_id}-{self._ids[base_id]}"

        self.rows.append(
            {
                "id": final_id,
                "source": source,
                "doc_type": doc_type,
                "title": clean_text(title),
                "text": text,
                "metadata": metadata,
            }
        )


def curriculum_chunks(builder: ChunkBuilder, curriculum: JsonDict) -> None:
    code = curriculum["curriculum_code"]
    title = f"{code} - {curriculum['name']}"

    overview = pair_lines(
        curriculum,
        [
            ("Mã chương trình", "curriculum_code"),
            ("Khóa", "cohort"),
            ("Tên chương trình", "name"),
            ("Tên tiếng Anh", "english_name"),
            ("Quyết định", "decision"),
            ("Tổng tín chỉ", "total_credit"),
            ("URL", "url"),
            ("Mô tả", "description"),
        ],
    )
    if curriculum.get("plos"):
        overview.append("Chuẩn đầu ra chương trình:")
        for plo in curriculum["plos"]:
            overview.append(f"- {plo.get('name')}: {plo.get('description')}")

    builder.add(
        doc_id=f"curriculum:{code}:overview",
        source="flm-crawl",
        doc_type="curriculum_overview",
        title=title,
        text=join_lines(overview),
        metadata={
            "curriculum_code": code,
            "cohort": curriculum.get("cohort"),
            "url": curriculum.get("url"),
        },
    )

    by_semester: dict[str, list[JsonDict]] = defaultdict(list)
    for subject in curriculum.get("subjects") or []:
        semester = subject.get("semester") or "unknown"
        by_semester[semester].append(subject)

        option_lines = []
        for option in subject.get("syllabus_options") or []:
            label = option.get("syllabus_name") or option.get("sylid") or option.get("subject_code")
            option_lines.append(
                f"- {label}; sylid={option.get('sylid')}; active={option.get('is_active')}; "
                f"approved={option.get('is_approved')}; url={option.get('url')}"
            )

        text = join_lines(
            [
                f"Mã chương trình: {code}",
                f"Tên chương trình: {curriculum.get('name')}",
                f"Mã môn: {subject.get('subject_code')}",
                f"Tên môn: {subject.get('subject_name')}",
                f"Học kỳ: {subject.get('semester')}",
                labeled_line(
                    "Số tín chỉ",
                    subject.get("credits"),
                    "Không có dữ liệu tín chỉ trong curriculum",
                ),
                labeled_line("Điều kiện tiên quyết", subject.get("prerequisite")),
                f"URL danh sách syllabus: {subject.get('syllabus_list_url')}",
                "Các syllabus liên quan:",
                *option_lines,
            ]
        )

        builder.add(
            doc_id=f"curriculum:{code}:subject:{subject.get('subject_code') or subject.get('subject_name')}",
            source="flm-crawl",
            doc_type="curriculum_subject",
            title=f"{code} / {subject.get('subject_code')} - {subject.get('subject_name')}",
            text=text,
            metadata={
                "curriculum_code": code,
                "curriculum_name": curriculum.get("name"),
                "cohort": curriculum.get("cohort"),
                "subject_code": subject.get("subject_code"),
                "semester": subject.get("semester"),
            },
        )

    for semester, subjects in sorted(by_semester.items(), key=lambda item: item[0]):
        lines = [
            f"Mã chương trình: {code}",
            f"Tên chương trình: {curriculum.get('name')}",
            f"Học kỳ: {semester}",
            "Danh sách môn:",
        ]
        for subject in subjects:
            lines.append(
                f"- {subject.get('subject_code')} - {subject.get('subject_name')}; "
                f"tín chỉ={subject.get('credits') or 'không có dữ liệu'}; "
                f"tiên quyết={subject.get('prerequisite') or 'không có dữ liệu'}"
            )

        builder.add(
            doc_id=f"curriculum:{code}:semester:{semester}",
            source="flm-crawl",
            doc_type="curriculum_semester",
            title=f"{code} - Học kỳ {semester}",
            text=join_lines(lines),
            metadata={
                "curriculum_code": code,
                "curriculum_name": curriculum.get("name"),
                "cohort": curriculum.get("cohort"),
                "semester": semester,
            },
        )


def syllabus_overview_chunks(builder: ChunkBuilder, syllabus: JsonDict) -> None:
    code = syllabus["subject_code"]
    detail = syllabus["detail"]
    title = f"{code} - {detail.get('syllabus_name')}"
    lines = [
        f"Mã môn: {code}",
        f"Syllabus ID: {detail.get('syllabus_id') or syllabus.get('sylid')}",
        f"Tên syllabus: {detail.get('syllabus_name')}",
        f"Tên tiếng Anh: {detail.get('syllabus_english')}",
        labeled_line(
            "Số tín chỉ",
            detail.get("credits"),
            "Không có dữ liệu tín chỉ trong syllabus",
        ),
        f"Bậc đào tạo: {detail.get('degree_level')}",
        f"Phân bổ thời gian: {detail.get('time_allocation')}",
        f"Điều kiện tiên quyết: {detail.get('prerequisite')}",
        f"Thang điểm: {detail.get('scoring_scale')}",
        f"Điểm trung bình tối thiểu để qua: {detail.get('min_average_mark_to_pass')}",
        f"Quyết định: {detail.get('decision_no')}",
        f"Đã duyệt: {detail.get('is_approved')}",
        f"Đang active: {detail.get('is_active')}",
        f"Ngày duyệt: {detail.get('approved_date')}",
        f"Ghi chú: {detail.get('note')}",
        f"Công cụ/phần mềm: {detail.get('tools')}",
        f"Nhiệm vụ sinh viên: {detail.get('student_tasks')}",
        f"Mô tả: {detail.get('description')}",
        f"URL: {syllabus.get('url')}",
    ]

    builder.add(
        doc_id=f"syllabus:{code}:overview",
        source="flm-syllabi",
        doc_type="syllabus_overview",
        title=title,
        text=join_lines(lines),
        metadata={
            "subject_code": code,
            "sylid": syllabus.get("sylid"),
            "url": syllabus.get("url"),
        },
    )


def syllabus_list_chunk(
    builder: ChunkBuilder,
    *,
    syllabus: JsonDict,
    key: str,
    doc_type: str,
    title_suffix: str,
    formatter,
) -> None:
    code = syllabus["subject_code"]
    items = syllabus.get(key) or []
    if not items:
        return

    lines = [f"Mã môn: {code}", title_suffix + ":"]
    for idx, item in enumerate(items, start=1):
        lines.extend(formatter(idx, item))

    builder.add(
        doc_id=f"syllabus:{code}:{key}",
        source="flm-syllabi",
        doc_type=doc_type,
        title=f"{code} - {title_suffix}",
        text=join_lines(lines),
        metadata={
            "subject_code": code,
            "sylid": syllabus.get("sylid"),
            "url": syllabus.get("url"),
        },
    )


def material_lines(idx: int, item: JsonDict) -> list[str]:
    return [
        f"- Tài liệu {idx}: {item.get('materialDescription')}",
        f"  Tác giả: {item.get('author')}; NXB: {item.get('publisher')}; "
        f"Năm: {item.get('publishedDate')}; Edition: {item.get('edition')}; ISBN: {item.get('isbn')}",
        f"  Main={item.get('isMainMaterial')}; HardCopy={item.get('isHardCopy')}; "
        f"Online={item.get('isOnline')}; Note={item.get('note')}",
    ]


def outcome_lines(idx: int, item: JsonDict) -> list[str]:
    return [
        f"- CLO {idx}: {item.get('cloName')}",
        f"  Chi tiết CLO: {item.get('cloDetails')}",
        f"  LO liên quan: {item.get('loDetails')}",
    ]


def question_lines(idx: int, item: JsonDict) -> list[str]:
    return [
        f"- Câu hỏi {idx}: session={item.get('sessionNo')}; name={item.get('name')}",
        f"  Nội dung: {item.get('details')}",
    ]


def assessment_lines(idx: int, item: JsonDict) -> list[str]:
    return [
        f"- Đánh giá {idx}: {item.get('category')} / {item.get('type')}; part={item.get('part')}",
        f"  Trọng số: {item.get('weight')}; tiêu chí hoàn thành: {item.get('completionCriteria')}; "
        f"thời lượng: {item.get('duration')}; CLO: {item.get('clo')}",
        f"  Dạng câu hỏi: {item.get('questionType')}; số câu/phần thi: {item.get('noQuestion')}",
        f"  Kiến thức/kỹ năng: {item.get('knowledgeAndSkill')}",
        f"  Hướng dẫn chấm: {item.get('gradingGuide')}",
        f"  Ghi chú: {item.get('note')}",
    ]


def syllabus_schedule_chunks(builder: ChunkBuilder, syllabus: JsonDict, sessions_per_chunk: int = 10) -> None:
    code = syllabus["subject_code"]
    schedule = syllabus.get("schedule") or []
    if not schedule:
        return

    for start in range(0, len(schedule), sessions_per_chunk):
        part = schedule[start : start + sessions_per_chunk]
        lines = [f"Mã môn: {code}", f"Lịch học sessions {start + 1}-{start + len(part)}:"]
        for item in part:
            lines.extend(
                [
                    f"- Session {item.get('session')}: {item.get('topic')}",
                    f"  Hình thức: {item.get('learningTeachingType')}; LO: {item.get('lo')}; ITU: {item.get('itu')}",
                    f"  Tài liệu: {item.get('studentMaterials')}",
                    f"  Việc sinh viên cần làm: {item.get('studentsTasks')}",
                    f"  Downloads: {item.get('studentMaterialDownloads')}; URLs: {item.get('urls')}",
                ]
            )

        builder.add(
            doc_id=f"syllabus:{code}:schedule:{start + 1}-{start + len(part)}",
            source="flm-syllabi",
            doc_type="syllabus_schedule",
            title=f"{code} - Lịch học {start + 1}-{start + len(part)}",
            text=join_lines(lines),
            metadata={
                "subject_code": code,
                "sylid": syllabus.get("sylid"),
                "url": syllabus.get("url"),
                "session_start": start + 1,
                "session_end": start + len(part),
            },
        )


def syllabus_chunks(builder: ChunkBuilder, syllabus: JsonDict) -> None:
    syllabus_overview_chunks(builder, syllabus)
    syllabus_list_chunk(
        builder=builder,
        syllabus=syllabus,
        key="learning_outcomes",
        doc_type="syllabus_learning_outcomes",
        title_suffix="Chuẩn đầu ra môn học",
        formatter=outcome_lines,
    )
    syllabus_list_chunk(
        builder=builder,
        syllabus=syllabus,
        key="materials",
        doc_type="syllabus_materials",
        title_suffix="Tài liệu học tập",
        formatter=material_lines,
    )
    syllabus_list_chunk(
        builder=builder,
        syllabus=syllabus,
        key="constructive_questions",
        doc_type="syllabus_constructive_questions",
        title_suffix="Câu hỏi constructive alignment",
        formatter=question_lines,
    )
    syllabus_list_chunk(
        builder=builder,
        syllabus=syllabus,
        key="assessments",
        doc_type="syllabus_assessments",
        title_suffix="Đánh giá môn học",
        formatter=assessment_lines,
    )
    syllabus_schedule_chunks(builder, syllabus)


def normalize_all(crawl_path: Path, syllabi_path: Path, output_dir: Path) -> JsonDict:
    crawl = load_json(crawl_path)
    syllabi_payload = load_json(syllabi_path)

    curricula = [normalize_curriculum(item) for item in crawl.get("curricula") or []]
    syllabi = [
        normalize_syllabus(subject_code, raw)
        for subject_code, raw in sorted((syllabi_payload.get("syllabi") or {}).items())
    ]

    builder = ChunkBuilder()
    for curriculum in curricula:
        curriculum_chunks(builder, curriculum)
    for syllabus in syllabi:
        syllabus_chunks(builder, syllabus)

    chunks = builder.rows
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "crawl": str(crawl_path),
            "syllabi": str(syllabi_path),
        },
        "curricula_count": len(curricula),
        "syllabi_count": len(syllabi),
        "chunks_count": len(chunks),
        "chunks_by_type": dict(Counter(row["doc_type"] for row in chunks)),
    }

    write_json(output_dir / "curricula.json", curricula)
    write_json(output_dir / "syllabi.json", syllabi)
    write_jsonl(output_dir / "knowledge_chunks.jsonl", chunks)
    write_json(output_dir / "stats.json", stats)
    return stats


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Normalize FLM crawl/syllabi JSON into RAG knowledge chunks.")
    parser.add_argument("--crawl", type=Path, default=settings.root_dir / "flm-crawl.json")
    parser.add_argument("--syllabi", type=Path, default=settings.root_dir / "flm-syllabi.json")
    parser.add_argument("--out", type=Path, default=settings.normalized_dir)
    args = parser.parse_args()

    stats = normalize_all(args.crawl, args.syllabi, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
