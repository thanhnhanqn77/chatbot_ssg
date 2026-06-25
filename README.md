# FLM Course Hybrid RAG Chatbot & API

Hệ thống Chatbot RAG (Retrieval-Augmented Generation) hỗ trợ trả lời thông tin về khung chương trình (curriculum) và đề cương môn học (syllabus) ngành SAGE/FLM của Đại học FPT. 

Hệ thống kết hợp tìm kiếm lai (**Hybrid Retrieval** - Dense & Sparse) sử dụng model **BGE-M3** để nhúng văn bản và xếp hạng lại với **BGE Reranker**, kết hợp tích hợp với mô hình ngôn ngữ lớn (LLM) qua API tương thích **Ollama**.

---

## 🛠️ Hướng Dẫn Cài Đặt và Chạy Dự Án

### 1. Clone Code Về Máy Local

Mở terminal (PowerShell, Command Prompt, hoặc Terminal trên macOS/Linux) và chạy lệnh sau để clone mã nguồn:

```bash
git clone https://github.com/thanhnhanqn77/chatbot_ssg.git
cd chatbot_ssg
```

### 2. Khởi Tạo và Kích Hoạt Môi Trường Ảo (Virtual Environment)

Nên sử dụng môi trường ảo Python (khuyên dùng Python 3.9 - 3.11) để tránh xung đột thư viện:

* **Trên Windows (PowerShell):**
  ```powershell
  python -m venv chatbot_ssg
  .\chatbot_ssg\Scripts\Activate.ps1
  ```
* **Trên Windows (CMD):**
  ```cmd
  python -m venv chatbot_ssg
  .\chatbot_ssg\Scripts\activate.bat
  ```
* **Trên macOS/Linux:**
  ```bash
  python3 -m venv chatbot_ssg
  source chatbot_ssg/bin/activate
  ```

### 3. Cài Đặt Các Thư Viện Cần Thiết

Cài đặt tất cả dependencies từ file `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Lưu ý: Lần đầu tiên chạy hoặc build chỉ mục, thư viện Hugging Face sẽ tự động tải các mô hình `BAAI/bge-m3` và `BAAI/bge-reranker-v2-m3` về máy, quá trình này cần kết nối internet ổn định).*

### 4. Cấu Hình Biến Môi Trường (`.env`)

Sao chép file cấu hình mẫu `.env.example` thành `.env`:

```bash
# Trên Windows (PowerShell)
cp .env.example .env

# Trên macOS/Linux hoặc CMD
cp .env.example .env
```

Mở file `.env` bằng trình soạn thảo và điền thông tin kết nối tới Ollama / LLM của bạn:

```env
OLLAMA_HOST=https://ollama.com              # URL của Ollama API (hoặc endpoint tương thích)
OLLAMA_MODEL=gpt-oss:120b-cloud             # Tên model LLM sử dụng
OLLAMA_API_KEY=your_api_key_here            # API key kết nối (nếu dùng cloud/remote host)
RAG_EMBEDDING_MODEL=BAAI/bge-m3             # Model dùng để nhúng văn bản (dense/sparse)
```

---

## 🏗️ Tạo Chỉ Mục Tìm Kiếm (Build Knowledge Index)

Nếu bạn chưa có sẵn dữ liệu index trong thư mục `data/index`, hãy tạo dữ liệu và build index bằng cách chạy tập lệnh chuẩn bị:

```bash
# Tạo và build hybrid index từ dữ liệu normalized
python scripts/prepare_rag.py --backend hybrid
```

Lệnh này sẽ xử lý các chunk từ dữ liệu thô, nhúng vector bằng model BGE-M3 (dense và sparse) và lưu trữ cục bộ vào đĩa dưới thư mục `data/index/`.

---

## 🚀 Chạy Web Chatbot và API Server

Khởi động máy chủ API tích hợp giao diện web bằng lệnh sau:

```bash
python scripts/serve_api.py
```

Mặc định máy chủ sẽ khởi chạy tại: **`http://127.0.0.1:8000`**

### 💻 Truy Cập Giao Diện Web Chatbot
* Mở trình duyệt web và truy cập địa chỉ: **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**
* Giao diện chat trực quan sẽ hiển thị. Bạn có thể nhập các câu hỏi liên quan đến môn học, chương trình học, số tín chỉ, lộ trình học để kiểm tra và nhận câu trả lời từ RAG Chatbot.

### 🧪 Test API Bằng Câu Lệnh (Command Line)
Bạn có thể test trực tiếp cổng API `/chat` bằng công cụ dòng lệnh:

* **Bằng PowerShell (Windows):**
  ```powershell
  Invoke-RestMethod http://127.0.0.1:8000/chat `
    -Method Post `
    -ContentType "application/json" `
    -Body '{"question": "Ngành Trí tuệ nhân tạo học mấy học kỳ?", "top_k": 6}'
  ```

* **Bằng cURL (Terminal macOS/Linux/Git Bash):**
  ```bash
  curl -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "Ngành Trí tuệ nhân tạo học mấy học kỳ?", "top_k": 6}'
  ```

---

## 📂 Cấu Trúc Dự Án

* `rag_app/`: Chứa mã nguồn cốt lõi của RAG pipeline (FastAPI app, retrievers, normalizers).
  * `rag_app/static/index.html`: Giao diện Web Chatbot (HTML/JS/CSS).
  * `rag_app/api.py`: Các endpoints API và logic tích hợp.
* `scripts/`: Chứa các script chạy độc lập để build index và chạy server.
* `requirements.txt`: Danh sách thư viện Python phụ thuộc.
* `.gitignore`: File quy định các thư mục/file không được đẩy lên Git (tránh lộ API key và đẩy các file nhúng vector siêu nặng).
