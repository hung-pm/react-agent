"""
models.py — Data models dùng xuyên suốt project
"""
from dataclasses import dataclass, field
from typing import Optional
import hashlib


@dataclass
class CodeChunk:
    """Một đơn vị code có nghĩa (function, class, module-level)"""

    content: str
    chunk_type: str          # "function" | "class" | "method" | "module"
    file_path: str
    start_line: int
    end_line: int
    name: str
    language: str

    # Optional metadata
    signature: str = ""
    docstring: str = ""
    parent_class: Optional[str] = None

    # Auto-generated
    id: str = field(default="", init=False)

    def __post_init__(self):
        # ID là hash ổn định để tránh index lại khi content không đổi
        raw = f"{self.file_path}:{self.name}:{self.start_line}"
        self.id = hashlib.md5(raw.encode()).hexdigest()[:16]

    def to_embed_text(self) -> str:
        """
        Văn bản dùng để tạo embedding.
        Không embed raw code — dùng mô tả ngôn ngữ tự nhiên kết hợp code.
        """
        parts = [
            f"File: {self.file_path}",
            f"Type: {self.chunk_type}  Name: {self.name}",
        ]
        if self.parent_class:
            parts.append(f"Class: {self.parent_class}")
        if self.signature:
            parts.append(f"Signature: {self.signature}")
        if self.docstring:
            parts.append(f"Description: {self.docstring}")
        # Chỉ lấy 30 dòng đầu để không vượt token limit của embedding model
        preview_lines = self.content.splitlines()[:30]
        parts.append("\n".join(preview_lines))
        return "\n".join(parts)

    def to_metadata(self) -> dict:
        """Metadata lưu vào ChromaDB để filter sau này"""
        return {
            "file_path": self.file_path,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent_class": self.parent_class or "",
            "signature": self.signature,
        }


@dataclass
class AgentState:
    """State của LangGraph agent — truyền qua tất cả các node"""
    messages: list = field(default_factory=list)
    query: str = ""
    retrieved_chunks: list = field(default_factory=list)   # List[CodeChunk]
    bug_analysis: str = ""
    proposed_fix: str = ""
    final_answer: str = ""
