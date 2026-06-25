from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sklearn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rag_app.config import get_settings
else:
    from .config import get_settings


JsonDict = dict[str, Any]


def load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def searchable_text(row: JsonDict) -> str:
    metadata = row.get("metadata") or {}
    metadata_text = " ".join(str(value) for value in metadata.values() if value not in (None, ""))
    return "\n".join(
        [
            str(row.get("title") or ""),
            str(row.get("doc_type") or ""),
            metadata_text,
            str(row.get("text") or ""),
        ]
    )


def build_index(chunks_path: Path, index_dir: Path) -> JsonDict:
    rows = load_jsonl(chunks_path)
    if not rows:
        raise ValueError(f"No chunks found in {chunks_path}")

    texts = [searchable_text(row) for row in rows]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_features=200_000,
        token_pattern=r"(?u)\b[\w./+-]+\b",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    index_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(chunks_path, index_dir / "documents.jsonl")

    payload = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "doc_ids": [row["id"] for row in rows],
        "versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (index_dir / "tfidf.pkl").open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    stats = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chunks_path": str(chunks_path),
        "documents": len(rows),
        "features": len(vectorizer.vocabulary_),
        "index_file": str(index_dir / "tfidf.pkl"),
        "documents_file": str(index_dir / "documents.jsonl"),
        "versions": payload["versions"],
    }
    with (index_dir / "stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return stats


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build a local TF-IDF retrieval index from normalized chunks.")
    parser.add_argument("--chunks", type=Path, default=settings.normalized_dir / "knowledge_chunks.jsonl")
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    args = parser.parse_args()

    stats = build_index(args.chunks, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
