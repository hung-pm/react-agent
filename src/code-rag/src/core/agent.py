"""
agent.py — LangGraph agent cho việc hiểu codebase và fix bug

Graph:
  START → retrieve → analyze → generate_fix → END
"""
import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .models import CodeChunk
from .tools import make_tools
from src.search.vector_store import CodebaseVectorStore
from src.parsing.code_graph import CodeGraph
from src.search.hybrid_search import HybridSearchEngine

# ------------------------------------------------------------------ #
#  STATE                                                               #
# ------------------------------------------------------------------ #

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ------------------------------------------------------------------ #
#  AGENT GRAPH                                                         #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """Bạn là một AI agent chuyên phân tích codebase và fix bug.

Bạn có các tools sau:
- `search_code`: tìm code liên quan đến vấn đề
- `get_callers`: tìm hàm nào đang gọi function đang xem xét
- `get_callees`: tìm hàm nào được function gọi đến
- `get_graph_context`: lấy toàn bộ context graph của một function

WORKFLOW:
1. Dùng `search_code` để tìm đoạn code liên quan đến bug/yêu cầu
2. Dùng `get_graph_context` hoặc `get_callers`/`get_callees` để hiểu context
3. Phân tích code và đưa ra nguyên nhân gốc rễ của bug/yêu cầu
4. Đề xuất fix cụ thể với code diff rõ ràng nếu là fix bug

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
    if chunks:
        hybrid_engine = HybridSearchEngine(vector_store, chunks, rrf_k=60)

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