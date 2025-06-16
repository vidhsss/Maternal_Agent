from sentence_transformers import CrossEncoder
from typing import List

class CrossEncoderReranker:
    """
    Reranks documents using a cross-encoder model.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[str], top_k: int = 7) -> List[int]:
        """
        Returns indices of top_k reranked documents.
        """
        pairs = [(query, doc) for doc in docs]
        scores = self.model.predict(pairs)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted_indices[:top_k] 