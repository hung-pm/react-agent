"""
vector_store.py — Indexing và retrieval với ChromaDB + OpenAI embeddings

Lưu vector index xuống file .chroma/ — không cần re-index mỗi lần khởi động.
"""
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.core.models import CodeChunk


COLLECTION_NAME = "codebase"


class CodebaseVectorStore:
    """
    Wrapper quanh ChromaDB cho việc index và search code chunks.
    """

    def __init__(
        self,
        persist_dir: str = ".chroma",
        embedding_model: str = "text-embedding-e5-large-v2",
        openai_api_key: str | None = None,
    ):
        self.persist_dir = persist_dir
        self.embedding_fn = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", "lm-studio"),
            openai_api_base=os.environ.get("LM_STUDIO_BASE_URL", "https://hung-pm.nport.link/v1"),
            check_embedding_ctx_length=False,
        )
        self._vectordb: Chroma | None = None

    def _get_db(self) -> Chroma:
        if self._vectordb is None:
            self._vectordb = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                persist_directory=self.persist_dir,
            )
        return self._vectordb

    # ------------------------------------------------------------------ #
    #  INDEXING                                                            #
    # ------------------------------------------------------------------ #

    def index_chunks(
        self,
        chunks: list[CodeChunk],
        batch_size: int = 100,
        force_reindex: bool = False,
    ) -> int:
        """
        Index danh sách CodeChunk vào ChromaDB.

        Args:
            chunks: Danh sách chunk cần index
            batch_size: Số chunk mỗi lần gọi API embedding (tránh timeout)
            force_reindex: Xóa collection cũ và index lại toàn bộ

        Returns:
            Số chunk đã được index
        """
        if force_reindex:
            self.clear()

        db = self._get_db()

        # Kiểm tra chunk nào chưa có trong DB (tránh index duplicate)
        existing_ids = set()
        try:
            result = db.get(include=[])  # chỉ lấy IDs
            existing_ids = set(result["ids"])
        except Exception:
            pass

        new_chunks = [c for c in chunks if c.id not in existing_ids]
        if not new_chunks:
            return 0

        # Index theo batch
        total = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            db.add_texts(
                texts=[c.to_embed_text() for c in batch],
                metadatas=[c.to_metadata() for c in batch],
                ids=[c.id for c in batch],
                # documents = full source code (để retrieve sau)
                # LangChain Chroma lưu `texts` làm documents
            )
            total += len(batch)

        return total

    def get_stats(self) -> dict:
        """Thống kê collection hiện tại."""
        try:
            db = self._get_db()
            count = db._collection.count()
            return {"total_chunks": count, "persist_dir": self.persist_dir}
        except Exception:
            return {"total_chunks": 0, "persist_dir": self.persist_dir}

    def clear(self):
        """Xóa toàn bộ collection."""
        try:
            db = self._get_db()
            db.delete_collection()
            self._vectordb = None
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  RETRIEVAL                                                           #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        k: int = 8,
        filter_language: str | None = None,
        filter_type: str | None = None,
        filter_file: str | None = None,
    ) -> list[tuple[CodeChunk, float]]:
        """
        Tìm kiếm chunk liên quan đến query.

        Args:
            query: Câu hỏi hoặc mô tả bug
            k: Số kết quả trả về
            filter_language: Lọc theo ngôn ngữ (vd: "python")
            filter_type: Lọc theo loại chunk (vd: "function")
            filter_file: Lọc theo tên file (partial match)

        Returns:
            List of (CodeChunk, score) sorted by relevance
        """
        db = self._get_db()

        # Xây dựng metadata filter nếu có
        where = {}
        if filter_language:
            where["language"] = filter_language
        if filter_type:
            where["chunk_type"] = filter_type

        try:
            results = db.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=where if where else None,
            )
        except Exception as e:
            print(f"[search error] {e}")
            return []

        chunks_with_scores = []
        for doc, score in results:
            # Reconstruct CodeChunk từ metadata + document content
            meta = doc.metadata
            chunk = CodeChunk(
                content=doc.page_content,
                chunk_type=meta.get("chunk_type", "function"),
                file_path=meta.get("file_path", ""),
                start_line=meta.get("start_line", 0),
                end_line=meta.get("end_line", 0),
                name=meta.get("name", ""),
                language=meta.get("language", ""),
                signature=meta.get("signature", ""),
                docstring="",
                parent_class=meta.get("parent_class") or None,
            )
            # filter_file là partial match
            if filter_file and filter_file not in chunk.file_path:
                continue
            chunks_with_scores.append((chunk, score))

        return chunks_with_scores

    def search_by_name(self, name: str) -> list[CodeChunk]:
        """Tìm chunk theo tên function/class chính xác."""
        db = self._get_db()
        try:
            results = db.get(
                where={"name": name},
                include=["documents", "metadatas"],
            )
            chunks = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                chunk = CodeChunk(
                    content=doc,
                    chunk_type=meta.get("chunk_type", "function"),
                    file_path=meta.get("file_path", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    name=meta.get("name", ""),
                    language=meta.get("language", ""),
                    signature=meta.get("signature", ""),
                    parent_class=meta.get("parent_class") or None,
                )
                chunks.append(chunk)
            return chunks
        except Exception:
            return []
