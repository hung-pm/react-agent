"""
code_graph.py — Xây dựng call graph từ codebase dùng NetworkX

Phân tích AST để tìm quan hệ gọi hàm giữa các chunk.
"""
import re
from collections import defaultdict

import networkx as nx

from src.core.models import CodeChunk


class CodeGraph:
    """
    Đồ thị có hướng biểu diễn quan hệ giữa các function/class trong codebase.

    Node: tên function/class
    Edge: A → B nghĩa là A gọi B (calls), hoặc A import B (imports)
    """

    def __init__(self):
        self.G = nx.DiGraph()
        # Map tên → chunk để tra cứu nhanh
        self._name_to_chunk: dict[str, CodeChunk] = {}

    # ------------------------------------------------------------------ #
    #  BUILD                                                               #
    # ------------------------------------------------------------------ #

    def build(self, chunks: list[CodeChunk]):
        """Xây dựng graph từ danh sách chunks."""
        # Thêm tất cả nodes trước
        for chunk in chunks:
            self.G.add_node(
                chunk.name,
                file_path=chunk.file_path,
                chunk_type=chunk.chunk_type,
                start_line=chunk.start_line,
                language=chunk.language,
            )
            self._name_to_chunk[chunk.name] = chunk

        # Xây dựng set tên đã biết để filter false positive
        known_names = set(self._name_to_chunk.keys())

        # Thêm edges dựa trên call analysis
        for chunk in chunks:
            callees = self._extract_calls(chunk, known_names)
            for callee in callees:
                if callee != chunk.name:  # tránh self-loop
                    self.G.add_edge(chunk.name, callee, edge_type="calls")

    def _extract_calls(
        self, chunk: CodeChunk, known_names: set[str]
    ) -> set[str]:
        """
        Trích các tên hàm được gọi trong chunk bằng regex.
        Đây là heuristic đơn giản — đủ tốt cho demo.
        """
        calls = set()
        content = chunk.content

        # Pattern: tên_hàm( hoặc self.tên_hàm( hoặc obj.tên_hàm(
        # Bắt tên hàm trước dấu (
        pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        for match in pattern.finditer(content):
            name = match.group(1)
            if name in known_names:
                calls.add(name)

        # Python: method calls dạng self.method_name(
        self_pattern = re.compile(r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
        for match in self_pattern.finditer(content):
            name = match.group(1)
            if name in known_names:
                calls.add(name)

        return calls

    # ------------------------------------------------------------------ #
    #  QUERY                                                               #
    # ------------------------------------------------------------------ #

    def get_callers(self, func_name: str) -> list[str]:
        """Trả về danh sách hàm gọi func_name (predecessors)."""
        if func_name not in self.G:
            return []
        return list(self.G.predecessors(func_name))

    def get_callees(self, func_name: str) -> list[str]:
        """Trả về danh sách hàm được func_name gọi (successors)."""
        if func_name not in self.G:
            return []
        return list(self.G.successors(func_name))

    def get_context_names(self, func_name: str, radius: int = 1) -> list[str]:
        """
        Trả về tên tất cả node trong vùng lân cận bán kính `radius`.
        Dùng để thu thập ngữ cảnh cho LLM.
        """
        if func_name not in self.G:
            return []
        subgraph = nx.ego_graph(
            self.G, func_name, radius=radius, undirected=True
        )
        # Loại bỏ chính func_name
        return [n for n in subgraph.nodes if n != func_name]

    def get_impact(self, func_name: str) -> list[str]:
        """
        Tìm tất cả hàm bị ảnh hưởng nếu thay đổi func_name.
        (Tất cả hàm có thể dẫn đến func_name qua call chain)
        """
        if func_name not in self.G:
            return []
        return list(nx.ancestors(self.G, func_name))

    def get_chunk(self, name: str) -> CodeChunk | None:
        return self._name_to_chunk.get(name)

    def get_context_chunks(
        self, func_name: str, radius: int = 1
    ) -> list[CodeChunk]:
        """
        Trả về CodeChunk của tất cả node lân cận.
        Đây là context bổ sung được đưa vào LLM cùng với chunk chính.
        """
        names = self.get_context_names(func_name, radius)
        chunks = []
        for name in names:
            chunk = self.get_chunk(name)
            if chunk:
                chunks.append(chunk)
        return chunks

    def summary(self) -> dict:
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
        }
