"""
tools.py — Các tools cho LangGraph agent

Tools:
  - search_code: vector search trong codebase
  - get_callers: tìm hàm gọi đến function
  - get_callees: tìm hàm được gọi bởi function
  - get_graph_context: lấy toàn bộ context graph
"""
from langchain_core.tools import tool

from src.search.vector_store import CodebaseVectorStore
from src.parsing.code_graph import CodeGraph
from src.search.hybrid_search import HybridSearchEngine


def make_tools(
    vector_store: CodebaseVectorStore,
    code_graph: CodeGraph,
    hybrid_engine: HybridSearchEngine | None = None,
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
