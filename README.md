# FLM Course Hybrid RAG API

API local cho chatbot RAG trên dữ liệu FLM curriculum/syllabus. Pipeline hiện tại:

1. Document loader từ `knowledge_chunks.jsonl`
2. Text cleaning và metadata preservation
3. Semantic chunking cho các chunk quá dài
4. BGE-M3 embedding
   - dense vector
   - sparse lexical weights
5. Local vector database trên disk
   - `documents.jsonl`
   - `dense.npy`
   - `hybrid.pkl`
   - `stats.json`
6. Query normalize
7. BGE-M3 query embedding
8. Hybrid retrieval
9. Merge và deduplicate
10. BGE reranker
11. Context builder
12. LLM answer generation qua Ollama-compatible `/api/chat`

## Cài Đặt

```powershell
.\chatbot_ssg\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Lần đầu chạy BGE-M3 sẽ tải model từ Hugging Face, nên cần mạng và đủ dung lượng.

## Build Hybrid Knowledge Index

Dùng trực tiếp dataset đã normalize:

```powershell
.\chatbot_ssg\Scripts\Activate.ps1
python scripts/build_hybrid_index.py --chunks D:\ssg\data\normalized\knowledge_chunks.jsonl --out D:\ssg\data\index
```

Hoặc normalize lại từ JSON gốc rồi build hybrid:

```powershell
.\chatbot_ssg\Scripts\Activate.ps1
python scripts/prepare_rag.py --backend hybrid
```

Nếu muốn build cả hybrid và TF-IDF fallback:

```powershell
python scripts/prepare_rag.py --backend both
```

## Chạy API

```powershell
.\chatbot_ssg\Scripts\Activate.ps1
python scripts/serve_api.py
```

Mặc định API chạy tại `http://127.0.0.1:8000`.

## Endpoint

`POST /chat`

```json
{
  "question": "OTP101 học mấy tín chỉ?",
  "top_k": 6
}
```

Test bằng PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"OTP101 học mấy tín chỉ?","top_k":6}'
```

## Cấu Hình Chính

Các biến trong `.env.example`:

- `RAG_RETRIEVAL_BACKEND=hybrid`
- `RAG_EMBEDDING_MODEL=BAAI/bge-m3`
- `RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3`
- `RAG_EMBEDDING_BATCH_SIZE=12`
- `RAG_CHUNK_MAX_CHARS=2200`
- `RAG_CHUNK_OVERLAP=250`
- `RAG_HYBRID_DENSE_WEIGHT=0.55`
- `RAG_HYBRID_SPARSE_WEIGHT=0.45`
- `RAG_HYBRID_USE_RERANKER=true`
