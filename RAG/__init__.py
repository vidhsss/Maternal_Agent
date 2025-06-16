from .core import MedicalRAG, EnhancedMedicalRAG
from .chunking import SemanticChunker, HierarchicalChunker
from .vector_store import VectorStore
from .reranking import CrossEncoderReranker

# ─── NEW: helper wrapper ────────────────────────────────────────────────────────
from langchain.embeddings import HuggingFaceEmbeddings

class E5Embeddings(HuggingFaceEmbeddings):
    """
    Adds the 'query: ' / 'passage: ' prefixes that E5‑instruct models expect.
    Everything else (batch‑ing, normalisation, device etc.) is inherited.
    """
    def __init__(self, device: str = "cpu",
                 model_name: str = "intfloat/multilingual-e5-large-instruct",
                 batch_size: int = 32):
        super().__init__(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": batch_size
            },
        )
        self._query_prefix   = "query: "
        self._passage_prefix = "passage: "

    # LangChain will call these two methods internally  ↓
    def embed_query(self, text: str):
        return super().embed_query(self._query_prefix + text)

    def embed_documents(self, texts):
        texts = [self._passage_prefix + t for t in texts]
        return super().embed_documents(texts)
# ────────────────────────────────────────────────────────────────────────────────
