# Codebase Bug-Fix Agent

LangGraph agent có khả năng hiểu codebase và fix bug, sử dụng:
- **tree-sitter-languages** — AST chunking
- **ChromaDB** + **OpenAI Embeddings** — vector search
- **rank_bm25** — BM25 keyword search
- **NetworkX** — call graph context
- **LangGraph** — agent orchestration
- **Streamlit** — giao diện web

---

## Cấu trúc project

```
agent-codebase/
├── src/
│   ├── core/                 # Nhân hệ thống
│   │   ├── agent.py          # LangGraph agent graph
│   │   ├── models.py         # CodeChunk data model
│   │   └── tools.py          # Agent tools (search, callers, callees, context)
│   ├── parsing/              # Phân tích mã nguồn
│   │   ├── chunker.py        # AST parser & chunking (tree-sitter)
│   │   └── code_graph.py     # Call graph (NetworkX)
│   └── search/               # Tìm kiếm & lưu trữ vector
│       ├── hybrid_search.py  # Hybrid search: BM25 + Semantic + RRF
│       └── vector_store.py   # ChromaDB wrapper
├── scripts/
│   ├── index.py              # Index codebase → ChromaDB
│   ├── chat.py               # Chat CLI với agent
│   ├── inspect_db.py         # Xem nội dung ChromaDB
│   └── inspect_graph.py      # Xem call graph relationships
├── streamlit_app.py           # Giao diện web Streamlit
├── sample_project/            # Codebase mẫu đơn giản
├── complex_project/           # Codebase mẫu phức tạp (3 tầng)
├── requirements.txt
└── .env.example
```

---

## Setup (5 phút)

### 1. Tạo virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# hoặc: .venv\Scripts\activate   # Windows
```

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

### 3. Tạo file `.env`

```bash
cp .env.example .env
```

Mở `.env` và điền API key:

```
OPENAI_API_KEY=sk-...your-key-here...
```

> Hệ thống hiện đang cấu hình sử dụng LM Studio local tại `http://[IP_ADDRESS]/v1`.

---

## Chạy

### Bước 1 — Index codebase

```bash
# Index sample project có sẵn
python scripts/index.py --dir ./sample_project

# Hoặc index project phức tạp hơn
python scripts/index.py --dir ./complex_project

# Hoặc index project của bạn
python scripts/index.py --dir /path/to/your/project

# Index lại từ đầu (xóa cache cũ)
python scripts/index.py --dir ./sample_project --reindex
```

### Bước 2 — Chat với agent

**CLI:**
```bash
python scripts/chat.py --dir ./complex_project
```

**Web UI (Streamlit):**
```bash
streamlit run streamlit_app.py
```

---

## Công cụ chẩn đoán

### Xem nội dung ChromaDB

```bash
# Xem thống kê
python scripts/inspect_db.py

# Xuất toàn bộ ra file
python scripts/inspect_db.py --full --output export.md
```

### Xem call graph

```bash
# Xem toàn bộ quan hệ
python scripts/inspect_graph.py --dir ./complex_project

# Xem chi tiết 1 hàm
python scripts/inspect_graph.py --dir ./complex_project --node save_user
```

---

## Câu hỏi mẫu để test

Sau khi chạy với `complex_project`, thử hỏi:

```
You: phân tích hàm handle_buy_items

You: hàm lưu user có vấn đề gì không?

You: tìm tất cả chỗ có thể gây lỗi

You: hàm nào gọi đến save_order?

You: fix bug trong create_checkout
```

---

## Hybrid Search

Agent sử dụng **Hybrid Search** kết hợp hai phương pháp:

| Phương pháp | Ưu điểm |
|---|---|
| **Semantic Search** (ChromaDB) | Tìm code theo ý nghĩa, hiểu ngữ cảnh |
| **BM25 Keyword** (rank_bm25) | Tìm chính xác tên hàm, biến, class |

Kết quả được kết hợp bằng **Reciprocal Rank Fusion (RRF)** để đạt độ chính xác cao nhất.

---

## Mở rộng

### Thêm ngôn ngữ mới
Trong `src/parsing/chunker.py`, thêm vào `EXTENSION_MAP` và `CHUNK_NODE_TYPES`:
```python
EXTENSION_MAP[".kt"] = "kotlin"
CHUNK_NODE_TYPES["kotlin"] = {"function_declaration", "class_declaration"}
```

### Thêm tool mới cho agent
Trong `src/core/tools.py`, thêm function với decorator `@tool` bên trong `make_tools()`:
```python
@tool
def search_by_file(file_path: str) -> str:
    """Lấy tất cả chunks trong một file cụ thể."""
    results = vector_store.search(file_path, filter_file=file_path, k=20)
    ...
```

---

## Troubleshooting

### `ModuleNotFoundError: tree_sitter_languages`
```bash
pip install tree-sitter-languages
```

### `ChromaDB error: collection not found`
Index lại:
```bash
python scripts/index.py --dir ./sample_project --reindex
```

### Dimension mismatch khi đổi embedding model
Xóa thư mục `.chroma/` rồi chạy index lại:
```bash
rm -rf .chroma
python scripts/index.py --dir ./complex_project --reindex
```

### Agent trả lời chung chung, không tìm thấy code
- Kiểm tra index đã chạy chưa: thư mục `.chroma/` phải tồn tại
- Chạy `python scripts/index.py` trước khi chat
