# Codebase Bug-Fix Agent

LangGraph agent có khả năng hiểu codebase và fix bug, sử dụng:
- **tree-sitter-languages** — AST chunking
- **ChromaDB** + **OpenAI Embeddings** — vector search
- **NetworkX** — call graph context
- **LangGraph** — agent orchestration

---

## Cấu trúc project

```
codebase-agent/
├── src/
│   ├── models.py        # CodeChunk, AgentState
│   ├── chunker.py       # AST parser & chunking
│   ├── vector_store.py  # ChromaDB wrapper
│   ├── code_graph.py    # Call graph (NetworkX)
│   └── agent.py         # LangGraph agent
├── scripts/
│   ├── index.py         # Index codebase → ChromaDB
│   └── chat.py          # Chat CLI với agent
├── sample_project/
│   └── app.py           # Codebase mẫu có bug để test
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

> Nếu không có OpenAI key, xem phần **Chạy offline** bên dưới.

---

## Chạy

### Bước 1 — Index codebase

```bash
# Index sample project có sẵn
python scripts/index.py --dir ./sample_project

# Hoặc index project của bạn
python scripts/index.py --dir /path/to/your/project

# Index lại từ đầu (xóa cache cũ)
python scripts/index.py --dir ./sample_project --reindex
```

Output mẫu:
```
━━━━━━━━━━━━━━ Codebase Indexer ━━━━━━━━━━━━━━
  Target dir : /path/to/sample_project
  ChromaDB   : .chroma
  Reindex    : False

  Language    Count
  python         12

  Type        Count
  function        8
  class           3
  method          1

  ✓ Call graph: 12 nodes, 7 call edges
  ✓ Done! 12 chunks in ChromaDB → .chroma

Index complete! Bạn có thể chạy agent ngay bây giờ.
```

### Bước 2 — Chat với agent

```bash
python scripts/chat.py --dir ./sample_project
```

---

## Câu hỏi mẫu để test

Sau khi chạy với `sample_project`, thử hỏi:

```
You: có bug gì trong hàm remove_item không?

You: hàm get_order_by_id có vấn đề gì?

You: fix bug trong calculate_total

You: tìm tất cả chỗ có thể gây KeyError

You: process_order có logic lỗi gì?

You: hàm nào gọi đến get_discount?
```

---

## Options

### Đổi model

```bash
# Dùng GPT-4o thay vì GPT-4o-mini (chính xác hơn, đắt hơn)
python scripts/chat.py --dir ./sample_project --model gpt-4o

# Dùng GPT-3.5 (nhanh hơn, rẻ hơn)
python scripts/chat.py --dir ./sample_project --model gpt-3.5-turbo
```

### Chạy offline (không cần OpenAI key)

Thay embedding model bằng Ollama:

1. Cài [Ollama](https://ollama.ai) và pull model:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

2. Sửa `src/vector_store.py` — đổi embedding function:
   ```python
   from langchain_ollama import OllamaEmbeddings
   
   self.embedding_fn = OllamaEmbeddings(model="nomic-embed-text")
   ```

3. Sửa `src/agent.py` — đổi LLM:
   ```python
   from langchain_ollama import ChatOllama
   
   llm = ChatOllama(model="llama3.2", temperature=0).bind_tools(tools)
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

### Agent trả lời chung chung, không tìm thấy code
- Kiểm tra index đã chạy chưa: thư mục `.chroma/` phải tồn tại
- Chạy `python scripts/index.py` trước khi chat

### `RateLimitError` từ OpenAI
Giảm batch size trong `scripts/index.py`:
```python
store.index_chunks(batch, batch_size=20)  # giảm từ 50 → 20
```

---

## Mở rộng

### Thêm ngôn ngữ mới
Trong `src/chunker.py`, thêm vào `EXTENSION_MAP` và `CHUNK_NODE_TYPES`:
```python
EXTENSION_MAP[".kt"] = "kotlin"
CHUNK_NODE_TYPES["kotlin"] = {"function_declaration", "class_declaration"}
```

### Thêm tool mới cho agent
Trong `src/agent.py`, thêm function với decorator `@tool`:
```python
@tool
def search_by_file(file_path: str) -> str:
    """Lấy tất cả chunks trong một file cụ thể."""
    results = vector_store.search(file_path, filter_file=file_path, k=20)
    ...
```

### Lưu conversation history
Dùng LangGraph checkpointing:
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)

# Mỗi session có thread_id riêng
config = {"configurable": {"thread_id": "session-001"}}
app.invoke({"messages": [...]}, config=config)
```
