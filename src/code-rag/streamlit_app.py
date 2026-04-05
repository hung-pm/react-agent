#!/usr/bin/env python3
"""
scripts/app.py — Giao diện Streamlit cho Codebase Bug-Fix Agent

Usage:
    streamlit run scripts/app.py
"""
import sys
from pathlib import Path

# File ở thư mục gốc project, thêm chính nó vào path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

from src.parsing.chunker import chunk_directory
from src.search.vector_store import CodebaseVectorStore
from src.parsing.code_graph import CodeGraph
from src.core.agent import create_agent


# ------------------------------------------------------------------ #
#  PAGE CONFIG                                                         #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="🔍 Codebase Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  CUSTOM CSS                                                          #
# ------------------------------------------------------------------ #

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        color: white !important;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 16px;
        padding: 12px 18px;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .status-ready {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
    }
    .status-loading {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
    }
    .status-idle {
        background: rgba(255,255,255,0.1);
        color: #aaa;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2em;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card p {
        margin: 4px 0 0;
        font-size: 0.85em;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  SIDEBAR                                                             #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.markdown("## 🤖 Codebase Agent")
    st.markdown("---")

    # Input đường dẫn project
    project_dir = st.text_input(
        "📁 Đường dẫn Project",
        value="/Users/nguyenvanquan/Quan/Dev/ai/agent-codebase/complex_project",
        placeholder="/Users/nguyenvanquan/Quan/Dev/ai/your-project",
        help="Nhập đường dẫn đầy đủ (full path) tới thư mục codebase cần phân tích",
    )

    chroma_dir = ".chroma"
    model_name = "lmstudio-community/gemma-4-e2b-it"

    st.markdown("---")

    # Nút Load / Reload
    load_btn = st.button("🚀 Load Codebase", use_container_width=True, type="primary")
    reindex_btn = st.button("🔄 Re-index", use_container_width=True)

    st.markdown("---")

    # Hiển thị trạng thái
    if "agent_ready" in st.session_state and st.session_state.agent_ready:
        st.markdown('<span class="status-badge status-ready">● Agent Sẵn Sàng</span>', unsafe_allow_html=True)

        stats = st.session_state.get("stats", {})
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats.get('chunks', 0)}</h3>
                <p>Code Chunks</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats.get('nodes', 0)}</h3>
                <p>Graph Nodes</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">○ Chưa Load</span>', unsafe_allow_html=True)
        st.caption("Nhập đường dẫn project và bấm **Load Codebase**")


# ------------------------------------------------------------------ #
#  LOAD / INDEX LOGIC                                                  #
# ------------------------------------------------------------------ #

def do_load(force_reindex=False):
    """Load codebase, build graph, connect vector store."""
    target = Path(project_dir).resolve()
    if not target.exists():
        st.error(f"❌ Thư mục không tồn tại: `{target}`")
        return

    with st.spinner("⏳ Đang phân tích codebase..."):
        chunks = chunk_directory(str(target))

    if not chunks:
        st.warning("⚠️ Không tìm thấy code chunks. Kiểm tra lại đường dẫn.")
        return

    with st.spinner("🔗 Đang xây dựng Call Graph..."):
        graph = CodeGraph()
        graph.build(chunks)
        graph_stats = graph.summary()

    store = CodebaseVectorStore(persist_dir=chroma_dir)

    if force_reindex:
        with st.spinner("🗑️ Đang xóa index cũ..."):
            store.clear()

    db_stats = store.get_stats()
    if db_stats["total_chunks"] == 0:
        with st.spinner("📊 Đang embedding & indexing..."):
            store.index_chunks(chunks)

    with st.spinner("🤖 Đang khởi tạo Agent..."):
        app = create_agent(store, graph, chunks=chunks, model=model_name)

    # Lưu vào session
    st.session_state.app = app
    st.session_state.agent_ready = True
    st.session_state.messages = []
    st.session_state.stats = {
        "chunks": len(chunks),
        "nodes": graph_stats["nodes"],
        "edges": graph_stats["edges"],
    }
    st.session_state.project_path = str(target)


if load_btn:
    do_load(force_reindex=False)
    st.rerun()

if reindex_btn:
    do_load(force_reindex=True)
    st.rerun()


# ------------------------------------------------------------------ #
#  MAIN CHAT AREA                                                      #
# ------------------------------------------------------------------ #

# Header
st.markdown("# 🔍 Codebase Bug-Fix Agent")

if "agent_ready" not in st.session_state or not st.session_state.agent_ready:
    st.info("👈 Nhập đường dẫn project ở thanh bên trái và bấm **Load Codebase** để bắt đầu.")
    st.stop()

st.caption(f"📂 Project: `{st.session_state.get('project_path', '')}`")

# Initialize chat history  
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Hỏi về codebase hoặc mô tả bug..."):
    # Hiển thị user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi agent bằng stream để hiển thị realtime
    with st.chat_message("assistant"):
        # Build LangGraph messages
        lc_messages = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))

        response_text = ""
        step_count = 0

        try:
            # st.status hiển thị realtime từng bước
            with st.status("🧠 Agent đang suy nghĩ...", expanded=True) as status:
                for event in st.session_state.app.stream({"messages": lc_messages}):
                    for node_name, node_output in event.items():
                        msgs = node_output.get("messages", [])
                        for msg in msgs:
                            # --- AI Message: tool call hoặc final answer ---
                            if isinstance(msg, AIMessage):
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        step_count += 1
                                        args_preview = str(tc["args"])
                                        if len(args_preview) > 200:
                                            args_preview = args_preview[:200] + "..."

                                        status.update(label=f"🔧 Step {step_count}: Gọi {tc['name']}...")
                                        st.markdown(f"**Step {step_count}** — 🔧 Gọi tool `{tc['name']}`")
                                        st.code(args_preview, language="json")

                                if msg.content and isinstance(msg.content, str) and msg.content.strip():
                                    response_text = msg.content

                            # --- Tool Message: kết quả trả về từ tool ---
                            elif hasattr(msg, "name") and hasattr(msg, "content"):
                                step_count += 1
                                tool_result = msg.content if isinstance(msg.content, str) else str(msg.content)

                                status.update(label=f"📋 Step {step_count}: Nhận kết quả {getattr(msg, 'name', 'tool')}...")
                                st.markdown(f"**Step {step_count}** — 📋 Kết quả `{getattr(msg, 'name', 'tool')}`")
                                result_display = tool_result
                                if len(result_display) > 1500:
                                    result_display = result_display[:1500] + "\n... (truncated)"
                                st.code(result_display, language="markdown")

                # Cập nhật trạng thái hoàn tất
                if step_count > 0:
                    status.update(label=f"✅ Hoàn tất {step_count} bước xử lý", state="complete", expanded=False)
                else:
                    status.update(label="✅ Trả lời trực tiếp", state="complete", expanded=False)

            # Hiển thị response chính
            if response_text:
                st.markdown(response_text)
            else:
                response_text = "Agent không trả lời được. Hãy thử lại với câu hỏi khác."
                st.warning(response_text)

        except Exception as e:
            response_text = f"❌ Lỗi: {str(e)}"
            st.error(response_text)

        # Lưu vào history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
