"""
chunker.py — AST-based code chunker dùng tree-sitter-languages

Hỗ trợ: Python, TypeScript, JavaScript, Go, Java, Rust, C, C++
"""
import os
from pathlib import Path
from typing import Iterator

from tree_sitter_languages import get_parser, get_language

from .models import CodeChunk

# Map extension → tên ngôn ngữ tree-sitter
EXTENSION_MAP = {
    ".py":   "python",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".go":   "go",
    ".java": "java",
    ".rs":   "rust",
    ".c":    "c",
    ".cpp":  "cpp",
    ".cc":   "cpp",
}

# Node types cần extract theo từng ngôn ngữ
CHUNK_NODE_TYPES = {
    "python": {
        "function_definition",
        "async_function_definition",
        "class_definition",
    },
    "typescript": {
        "function_declaration",
        "method_definition",
        "class_declaration",
        "arrow_function",
        "lexical_declaration",   # const fn = () => {}
    },
    "javascript": {
        "function_declaration",
        "method_definition",
        "class_declaration",
        "arrow_function",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "java": {
        "method_declaration",
        "class_declaration",
        "constructor_declaration",
    },
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
    },
    "c":   {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier"},
}


def _get_node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_docstring_python(node, source_bytes: bytes) -> str:
    """Trích docstring từ function/class Python."""
    try:
        body = next(
            (c for c in node.children if c.type == "block"), None
        )
        if body:
            first_stmt = next(
                (c for c in body.children if c.type == "expression_statement"),
                None,
            )
            if first_stmt:
                string_node = next(
                    (c for c in first_stmt.children if c.type == "string"),
                    None,
                )
                if string_node:
                    raw = _get_node_text(string_node, source_bytes)
                    return raw.strip('"""').strip("'''").strip('"').strip("'").strip()
    except Exception:
        pass
    return ""


def _get_name(node, source_bytes: bytes) -> str:
    """Lấy tên của function hoặc class."""
    name_node = next(
        (c for c in node.children if c.type == "identifier"), None
    )
    if name_node:
        return _get_node_text(name_node, source_bytes)
    return "<anonymous>"


def _get_signature(node, source_bytes: bytes, language: str) -> str:
    """Lấy dòng signature (dòng đầu tiên của function/class)."""
    text = _get_node_text(node, source_bytes)
    first_line = text.splitlines()[0].strip()
    # Giới hạn độ dài
    return first_line[:200]


def _find_parent_class(node) -> str | None:
    """Tìm class cha nếu node là method."""
    parent = node.parent
    while parent:
        if parent.type in ("class_definition", "class_declaration", "impl_item"):
            # Tìm tên class
            name_node = next(
                (c for c in parent.children if c.type == "identifier"), None
            )
            if name_node:
                return name_node.text.decode("utf-8", errors="replace")
        parent = parent.parent
    return None


def _extract_chunks_from_tree(
    root_node,
    source_bytes: bytes,
    file_path: str,
    language: str,
    target_types: set,
) -> list[CodeChunk]:
    """Duyệt AST và extract tất cả node thuộc target_types."""
    chunks = []

    def walk(node, depth=0):
        if node.type in target_types:
            name = _get_name(node, source_bytes)
            content = _get_node_text(node, source_bytes)
            signature = _get_signature(node, source_bytes, language)
            docstring = (
                _get_docstring_python(node, source_bytes)
                if language == "python"
                else ""
            )
            parent_class = _find_parent_class(node)

            # Xác định chunk_type
            t = node.type
            if "class" in t:
                chunk_type = "class"
            elif "method" in t:
                chunk_type = "method"
            else:
                chunk_type = "function"

            chunk = CodeChunk(
                content=content,
                chunk_type=chunk_type,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                name=name,
                language=language,
                signature=signature,
                docstring=docstring,
                parent_class=parent_class,
            )
            chunks.append(chunk)

        for child in node.children:
            walk(child, depth + 1)

    walk(root_node)
    return chunks


def chunk_file(file_path: str) -> list[CodeChunk]:
    """
    Parse một file và trả về danh sách CodeChunk.
    Trả về [] nếu không hỗ trợ ngôn ngữ.
    """
    ext = Path(file_path).suffix.lower()
    language = EXTENSION_MAP.get(ext)
    if not language:
        return []

    target_types = CHUNK_NODE_TYPES.get(language, set())
    if not target_types:
        return []

    try:
        source_bytes = Path(file_path).read_bytes()
    except (OSError, PermissionError):
        return []

    try:
        parser = get_parser(language)
        tree = parser.parse(source_bytes)
    except Exception as e:
        print(f"  [warn] parse error {file_path}: {e}")
        return []

    return _extract_chunks_from_tree(
        tree.root_node, source_bytes, file_path, language, target_types
    )


def chunk_directory(
    directory: str,
    exclude_dirs: set[str] | None = None,
) -> list[CodeChunk]:
    """
    Duyệt toàn bộ thư mục và chunk tất cả file code được hỗ trợ.

    Args:
        directory: Đường dẫn thư mục gốc
        exclude_dirs: Tên thư mục cần bỏ qua (mặc định: node_modules, .git, ...)

    Returns:
        Danh sách tất cả CodeChunk từ codebase
    """
    if exclude_dirs is None:
        exclude_dirs = {
            ".git", ".svn", "node_modules", "__pycache__",
            ".venv", "venv", "env", ".env",
            "dist", "build", ".next", ".nuxt",
            "coverage", ".pytest_cache", ".mypy_cache",
        }

    all_chunks: list[CodeChunk] = []
    root = Path(directory).resolve()

    for path in root.rglob("*"):
        # Bỏ qua thư mục bị exclude
        if any(part in exclude_dirs for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in EXTENSION_MAP:
            continue

        rel_path = str(path.relative_to(root))
        file_chunks = chunk_file(str(path))

        for chunk in file_chunks:
            chunk.file_path = rel_path  # dùng relative path

        all_chunks.extend(file_chunks)

    return all_chunks
