from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rag_app.config import get_settings
    from rag_app.encoding import clean_text
else:
    from .config import get_settings
    from .encoding import clean_text


JsonDict = dict[str, Any]


def load_jsonl(path: Path | str) -> list[JsonDict]:
    path = Path(path)
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def searchable_text(row: JsonDict) -> str:
    metadata = row.get("metadata") or {}
    metadata_text = " ".join(str(value) for value in metadata.values() if value not in (None, ""))
    return clean_text(
        "\n".join(
            [
                str(row.get("title") or ""),
                str(row.get("doc_type") or ""),
                metadata_text,
                str(row.get("text") or ""),
            ]
        )
    )


def _split_long_paragraph(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?。！？])\s+", paragraph)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            for start in range(0, len(sentence), max_chars - overlap):
                parts.append(sentence[start : start + max_chars].strip())
            current = ""
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current.strip())
            current = current[-overlap:] if overlap and len(current) > overlap else ""
        current = f"{current} {sentence}".strip()
    if current:
        parts.append(current.strip())
    return parts


def semantic_chunks(text: str, max_chars: int = 2200, overlap: int = 250) -> list[str]:
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n|(?<=\n)- ", text):
        paragraph = clean_text(paragraph)
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_paragraph(paragraph, max_chars, overlap))
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap and len(current) > overlap else ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if len(chunk) >= 20]


def organize_rows(rows: list[JsonDict], max_chars: int = 2200, overlap: int = 250) -> list[JsonDict]:
    organized: list[JsonDict] = []
    for row in rows:
        base = {
            "source": row.get("source"),
            "doc_type": row.get("doc_type"),
            "title": clean_text(row.get("title") or ""),
            "metadata": dict(row.get("metadata") or {}),
        }
        parts = semantic_chunks(row.get("text") or "", max_chars=max_chars, overlap=overlap)
        if len(parts) <= 1:
            organized.append({**row, **base, "text": parts[0] if parts else clean_text(row.get("text") or "")})
            continue

        for idx, part in enumerate(parts, start=1):
            metadata = dict(base["metadata"])
            metadata.update(
                {
                    "parent_id": row.get("id"),
                    "chunk_index": idx,
                    "chunk_count": len(parts),
                }
            )
            organized.append(
                {
                    **base,
                    "id": f"{row.get('id')}:part:{idx}",
                    "title": f"{base['title']} (part {idx}/{len(parts)})",
                    "text": part,
                    "metadata": metadata,
                }
            )
    return organized


class BGEM3Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str = "auto",
        use_fp16: bool = True,
        batch_size: int = 12,
        max_length: int = 8192,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.is_m3 = "bge-m3" in model_name.lower()

        try:
            if self.is_m3:
                from FlagEmbedding import BGEM3FlagModel
                wrapper_class = BGEM3FlagModel
            else:
                from FlagEmbedding import FlagModel
                wrapper_class = FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "Embedding requires FlagEmbedding. Install requirements.txt first."
            ) from exc

        kwargs: dict[str, Any] = {"use_fp16": use_fp16}
        if device and device != "auto":
            kwargs["device"] = device
        try:
            self.model = wrapper_class(model_name, **kwargs)
        except TypeError:
            kwargs.pop("device", None)
            self.model = wrapper_class(model_name, **kwargs)

    @staticmethod
    def _normalize_dense(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    @staticmethod
    def _clean_sparse(vector: dict[Any, Any]) -> dict[str, float]:
        return {str(key): float(value) for key, value in (vector or {}).items() if float(value) != 0.0}

    def encode(self, texts: list[str]) -> tuple[np.ndarray, list[dict[str, float]]]:
        if self.is_m3:
            output = self.model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense = self._normalize_dense(np.asarray(output["dense_vecs"], dtype=np.float32))
            sparse = [self._clean_sparse(item) for item in output.get("lexical_weights", [])]
            return dense, sparse
        else:
            dense = self.model.encode(texts)
            dense = self._normalize_dense(np.asarray(dense, dtype=np.float32))
            sparse = [{} for _ in texts]
            return dense, sparse

    def encode_query(self, query: str) -> tuple[np.ndarray, dict[str, float]]:
        dense, sparse = self.encode([query])
        return dense[0], sparse[0] if sparse else {}


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", *, use_fp16: bool = True) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        if use_fp16 and torch.cuda.is_available():
            self.model = self.model.half()
        self.model.eval()

    def score(self, query: str, documents: list[str]) -> list[float]:
        import torch
        
        if not documents:
            return []
            
        pairs = [[query, doc] for doc in documents]
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            outputs = self.model(**inputs)
            scores = outputs.logits.view(-1).float().cpu().numpy()
            
            # Sigmoid normalization
            scores = 1.0 / (1.0 + np.exp(-scores))
            
            if len(documents) == 1:
                return [float(scores[0])]
            return [float(s) for s in scores]


def _add_sparse_index(index: dict[str, list[tuple[int, float]]], row_idx: int, sparse: dict[str, float]) -> None:
    for term, weight in sparse.items():
        index.setdefault(term, []).append((row_idx, float(weight)))


def build_hybrid_index(
    chunks_path: Path,
    index_dir: Path,
    *,
    model_name: str = "BAAI/bge-m3",
    device: str = "auto",
    batch_size: int = 12,
    max_chars: int = 2200,
    overlap: int = 250,
) -> JsonDict:
    source_rows = load_jsonl(chunks_path)
    rows = organize_rows(source_rows, max_chars=max_chars, overlap=overlap)
    texts = [searchable_text(row) for row in rows]
    embedder = BGEM3Embedder(model_name, device=device, batch_size=batch_size)

    dense_batches: list[np.ndarray] = []
    sparse_vectors: list[dict[str, float]] = []
    for start in range(0, len(texts), batch_size):
        dense, sparse = embedder.encode(texts[start : start + batch_size])
        dense_batches.append(dense)
        sparse_vectors.extend(sparse)

    dense_matrix = np.vstack(dense_batches).astype(np.float32)
    sparse_index: dict[str, list[tuple[int, float]]] = {}
    for idx, sparse in enumerate(sparse_vectors):
        _add_sparse_index(sparse_index, idx, sparse)

    index_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(index_dir / "documents.jsonl", rows)
    np.save(index_dir / "dense.npy", dense_matrix)
    with (index_dir / "hybrid.pkl").open("wb") as file:
        pickle.dump(
            {
                "sparse_index": sparse_index,
                "model_name": model_name,
                "doc_ids": [row["id"] for row in rows],
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    shutil.copyfile(chunks_path, index_dir / "source_knowledge_chunks.jsonl")

    stats = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "backend": "bge-m3-hybrid-local",
        "source_chunks_path": str(chunks_path),
        "source_documents": len(source_rows),
        "documents": len(rows),
        "dense_shape": list(dense_matrix.shape),
        "sparse_terms": len(sparse_index),
        "embedding_model": model_name,
        "documents_file": str(index_dir / "documents.jsonl"),
        "dense_file": str(index_dir / "dense.npy"),
        "hybrid_file": str(index_dir / "hybrid.pkl"),
    }
    with (index_dir / "stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return stats


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build a BGE-M3 dense+sparse hybrid RAG index.")
    parser.add_argument("--chunks", type=Path, default=settings.normalized_dir / "knowledge_chunks.jsonl")
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--device", default=settings.embedding_device)
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    parser.add_argument("--max-chars", type=int, default=settings.chunk_max_chars)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    args = parser.parse_args()

    stats = build_hybrid_index(
        args.chunks,
        args.out,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
