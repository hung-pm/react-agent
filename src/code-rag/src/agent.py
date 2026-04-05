"""
agent.py — LangGraph agent cho việc hiểu codebase và fix bug

Graph:
  START → retrieve → analyze → generate_fix → END

Tools:
  - search_code: vector search trong codebase
  - get_callers: tìm hàm gọi đến function
  - get_callees: tìm hàm được gọi bởi function
"""
import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.tools import tool

from .models import CodeChunk
from .vector_store import CodebaseVectorStore
from .code_graph import CodeGraph

# ------------------------------------------------------------------ #
#  STATE                                                               #
# ------------------------------------------------------------------ #

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ------------------------------------------------------------------ #
#  TOOLS                                                               #
# ------------------------------------------------------------------ #

def make_tools(
    vector_store: CodebaseVectorStore,
    code_graph: CodeGraph,
    hybrid_engine=None,
):
    """
    Tạo tools, inject vector_store và code_graph vào closure.
    """

    @tool
    def search_code(query: str, top_k: int = 6) -> str:
        """
        Tìm kiếm các đoạn code liên quan đến query trong codebase.
        Dùng hybrid search (semantic + keyword) để tìm chính xác hơn.
        Dùng khi cần tìm function, class, hoặc logic liên quan đến bug.

        Args:
            query: Mô tả điều cần tìm (vd: "user authentication logic")
            top_k: Số kết quả trả về (mặc định 6)
        """
        engine = hybrid_engine or vector_store

        if hybrid_engine is not None:
            results_raw = hybrid_engine.search(query, k=top_k)
            # results_raw = [(chunk, score, source), ...]
            display = []
            for chunk, score, source in results_raw:
                header = (
                    f"[{chunk.chunk_type.upper()}] {chunk.name} "
                    f"— {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})"
                    f"  score={score:.4f}  [{source}]"
                )
                display.append(f"{header}\n```{chunk.language}\n{chunk.content}\n```")
        else:
            results = vector_store.search(query, k=top_k)
            if not results:
                return "Không tìm thấy code nào liên quan."
            display = []
            for chunk, score in results:
                header = (
                    f"[{chunk.chunk_type.upper()}] {chunk.name} "
                    f"— {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})"
                    f"  score={score:.2f}"
                )
                display.append(f"{header}\n```{chunk.language}\n{chunk.content}\n```")

        if not display:
            return "Không tìm thấy code nào liên quan."
        return "\n\n---\n\n".join(display)

    @tool
    def get_callers(function_name: str) -> str:
        """
        Tìm tất cả hàm/method đang gọi đến function_name.
        Dùng khi cần hiểu ai là người gọi function bị bug.

        Args:
            function_name: Tên chính xác của function cần tra
        """
        callers = code_graph.get_callers(function_name)
        if not callers:
            return f"Không tìm thấy hàm nào gọi `{function_name}`."

        result_parts = [f"Các hàm gọi đến `{function_name}`:"]
        for name in callers:
            chunk = code_graph.get_chunk(name)
            if chunk:
                result_parts.append(
                    f"- `{name}` ({chunk.chunk_type}) tại {chunk.file_path}:{chunk.start_line}"
                )
            else:
                result_parts.append(f"- `{name}`")

        return "\n".join(result_parts)

    @tool
    def get_callees(function_name: str) -> str:
        """
        Tìm tất cả hàm/method được function_name gọi đến.
        Dùng khi cần hiểu function bị bug đang phụ thuộc vào những gì.

        Args:
            function_name: Tên chính xác của function cần tra
        """
        callees = code_graph.get_callees(function_name)
        if not callees:
            return f"`{function_name}` không gọi hàm nào đã biết."

        result_parts = [f"Hàm `{function_name}` đang gọi:"]
        for name in callees:
            chunk = code_graph.get_chunk(name)
            if chunk:
                result_parts.append(
                    f"- `{name}` ({chunk.chunk_type}) tại {chunk.file_path}:{chunk.start_line}"
                )
            else:
                result_parts.append(f"- `{name}`")

        return "\n".join(result_parts)

    @tool
    def get_graph_context(function_name: str) -> str:
        """
        Lấy toàn bộ ngữ cảnh graph (callers + callees + code) cho một function.
        Dùng khi cần hiểu sâu về function trước khi fix bug.

        Args:
            function_name: Tên function cần lấy context
        """
        main_chunk = code_graph.get_chunk(function_name)
        context_chunks = code_graph.get_context_chunks(function_name, radius=1)

        parts = []

        if main_chunk:
            parts.append(
                f"=== {function_name} ({main_chunk.file_path}:{main_chunk.start_line}) ===\n"
                f"```{main_chunk.language}\n{main_chunk.content}\n```"
            )
        else:
            # Thử tìm qua vector store
            results = vector_store.search_by_name(function_name)
            if results:
                c = results[0]
                parts.append(
                    f"=== {function_name} ({c.file_path}:{c.start_line}) ===\n"
                    f"```{c.language}\n{c.content}\n```"
                )

        if context_chunks:
            parts.append(f"\n--- Context ({len(context_chunks)} related chunks) ---")
            for chunk in context_chunks[:4]:   # giới hạn 4 để không quá dài
                parts.append(
                    f"[{chunk.chunk_type}] {chunk.name} — {chunk.file_path}:{chunk.start_line}\n"
                    f"```{chunk.language}\n{chunk.content[:500]}...\n```"
                )

        callers = code_graph.get_callers(function_name)
        callees = code_graph.get_callees(function_name)
        if callers:
            parts.append(f"\nCallers: {', '.join(callers)}")
        if callees:
            parts.append(f"Callees: {', '.join(callees)}")

        return "\n\n".join(parts) if parts else f"Không tìm thấy `{function_name}`."

    return [search_code, get_callers, get_callees, get_graph_context]


# ------------------------------------------------------------------ #
#  AGENT GRAPH                                                         #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """Bạn là một AI agent chuyên phân tích codebase và fix bug.

Bạn có các tools sau:
- `search_code`: tìm code liên quan đến vấn đề
- `get_callers`: tìm hàm nào đang gọi function đang xem xét
- `get_callees`: tìm hàm nào được function gọi đến
- `get_graph_context`: lấy toàn bộ context graph của một function

WORKFLOW khi fix bug:
1. Dùng `search_code` để tìm đoạn code liên quan đến bug
2. Dùng `get_graph_context` hoặc `get_callers`/`get_callees` để hiểu context
3. Phân tích nguyên nhân gốc rễ
4. Đề xuất fix cụ thể với code diff rõ ràng

Khi trả lời:
- Luôn chỉ rõ FILE PATH và LINE NUMBER của bug
- Giải thích RÕ RÀNG tại sao đó là bug
- Cung cấp CODE FIX cụ thể (before/after)
- Nêu các hàm liên quan có thể bị ảnh hưởng

Trả lời bằng tiếng Việt nếu người dùng hỏi tiếng Việt."""


def create_agent(
    vector_store: CodebaseVectorStore,
    code_graph: CodeGraph,
    chunks: list | None = None,
    model: str = "lmstudio-community/gemma-4-e2b-it",
    openai_api_key: str | None = None,
):
    """
    Tạo LangGraph agent với hybrid search (semantic + BM25 keyword).

    Args:
        vector_store: Đã được index codebase
        code_graph: Đã được build từ chunks
        chunks: Danh sách chunks để build BM25 index (nếu None → semantic only)
        model: OpenAI model name
        openai_api_key: API key (hoặc lấy từ env)

    Returns:
        Compiled LangGraph app
    """
    hybrid_engine = None
    # if chunks:
    #     hybrid_engine = HybridSearchEngine(vector_store, chunks, alpha=0.7)

    tools = make_tools(vector_store, code_graph, hybrid_engine)

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", "lm-studio"),
        base_url=os.environ.get("LM_STUDIO_BASE_URL", "https://hung-pm.nport.link/v1"),
    ).bind_tools(tools)

    # Node: agent gọi LLM
    def agent_node(state: AgentState):
        messages = state["messages"]
        # Inject system prompt nếu chưa có
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        
        print("\n\n" + "="*50)
        print("🔍 CONTEXT GỬI LÊN LLM MÔ HÌNH:")
        for idx, msg in enumerate(messages):
            role = msg.__class__.__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(f"[{idx}] {role}: {content[:500]}..." if len(content) > 500 else f"[{idx}] {role}: {content}")
        print("="*50 + "\n\n")

        response = llm.invoke(messages)
        return {"messages": [response]}

    # Build graph
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile()
