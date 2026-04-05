import re
from typing import Any
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản thành list lowercase words."""
    return re.findall(r'\w+', text.lower())

class HybridSearchEngine:
    def __init__(self, vector_store, chunks, rrf_k=60):
        """
        Khởi tạo HybridSearchEngine bằng VectorStore và nội dung Chunk để tạo BM25 index.
        """
        self.vector_store = vector_store
        self.chunks = chunks
        self.rrf_k = rrf_k
        
        # Tiền xử lý tokenization
        tokenized_corpus = [tokenize(chunk.content) for chunk in self.chunks]
        self.bm25_model = BM25Okapi(tokenized_corpus)
        
    def _bm25_search(self, query: str, top_k: int) -> list[tuple[Any, float]]:
        """Lexical search trả về top_k theo bm25 object"""
        tokenized_query = tokenize(query)
        scores = self.bm25_model.get_scores(tokenized_query)
        
        # Ghép chunk và điểm, rồi sắp xếp giảm dần
        chunk_score = list(zip(self.chunks, scores))
        chunk_score.sort(key=lambda x: x[1], reverse=True)
        return chunk_score[:top_k]
        
    def search(self, query: str, k: int = 5) -> list[tuple[Any, float, str]]:
        """
        Thực hiện tìm kiếm hybrid và dung hợp qua RRF.
        Trả về kết quả chuẩn của Hybrid Search với meta 'source'.
        """
        limit = k * 2  # Lấy sâu hơn để scale RRF tốt hơn
        
        # 1. Semantic Search
        semantic_results = self.vector_store.search(query, k=limit)
        
        # 2. Keyword/BM25 Search
        bm25_results = self._bm25_search(query, top_k=limit)
        
        # 3. Reciprocal Rank Fusion
        # Tính RRF score: 1 / (k + rank)
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Any] = {}
        source_map: dict[str, str] = {}
        
        # Đánh giá list semantic
        for rank, (chunk, score) in enumerate(semantic_results):
            # Cần 1 định danh duy nhất (trong project này ta dùng hash nội dung làm id, hoặc tạo id ảo)
            uid = hash(chunk.content + chunk.name)
            chunk_map[uid] = chunk
            source_map[uid] = "semantic"
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            
        # Đánh giá list BM25 
        for rank, (chunk, score) in enumerate(bm25_results):
            uid = hash(chunk.content + chunk.name)
            chunk_map[uid] = chunk
            # Nếu đã có trong map thì gắn cờ hybrid
            if uid in source_map:
                source_map[uid] = "hybrid"
            else:
                source_map[uid] = "bm25"
                
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            
        # Sort object đã RRF
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Trích dẫn lại top_k kết quả sau khi fusion
        final_results = []
        for uid, rrf_score in sorted_rrf[:k]:
            final_results.append(
                (chunk_map[uid], rrf_score, source_map[uid])
            )
            
        return final_results
