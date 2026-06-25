from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rag_app.config import get_settings
    from rag_app.hybrid_index import build_hybrid_index
    from rag_app.index import build_index
    from rag_app.normalize import normalize_all
else:
    from .hybrid_index import build_hybrid_index
    from .config import get_settings
    from .index import build_index
    from .normalize import normalize_all


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Normalize FLM datasets and build a local RAG index.")
    parser.add_argument("--crawl", type=Path, default=settings.root_dir / "flm-crawl.json")
    parser.add_argument("--syllabi", type=Path, default=settings.root_dir / "flm-syllabi.json")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--normalized-out", type=Path, default=settings.normalized_dir)
    parser.add_argument("--index-out", type=Path, default=settings.index_dir)
    parser.add_argument("--backend", choices=["hybrid", "tfidf", "both"], default=settings.retrieval_backend)
    parser.add_argument("--skip-normalize", action="store_true")
    args = parser.parse_args()

    normalize_stats = None
    if args.skip_normalize:
        chunks_path = args.chunks or args.normalized_out / "knowledge_chunks.jsonl"
    else:
        normalize_stats = normalize_all(args.crawl, args.syllabi, args.normalized_out)
        chunks_path = args.normalized_out / "knowledge_chunks.jsonl"

    index_stats = {}
    if args.backend in {"hybrid", "both"}:
        index_stats["hybrid"] = build_hybrid_index(
            chunks_path,
            args.index_out,
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            max_chars=settings.chunk_max_chars,
            overlap=settings.chunk_overlap,
        )
    if args.backend in {"tfidf", "both"}:
        tfidf_dir = args.index_out if args.backend == "tfidf" else args.index_out / "tfidf"
        index_stats["tfidf"] = build_index(chunks_path, tfidf_dir)

    print(
        json.dumps(
            {
                "normalized": normalize_stats,
                "index": index_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
