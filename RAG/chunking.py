"""
chunking.py
-----------
Provides chunking strategies for text, including semantic and hierarchical chunkers.
These chunkers are used throughout the codebase for splitting documents into meaningful pieces for embedding and retrieval.

Classes:
    - SemanticChunker: Splits text into semantically meaningful chunks.
    - HierarchicalChunker: Splits text into hierarchical sections and then into smaller chunks.
Functions:
    - get_default_embeddings: Returns a default HuggingFace embedding model.
    - retrieve_top_k: Retrieves top-k relevant documents and computes similarity metrics.
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import numpy as np

class SemanticChunker:
    """
    Splits text into semantically meaningful chunks using a transformer model.
    """
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Add model initialization if needed

    def chunk(self, text: str) -> List[str]:
        # Simple split by sentences for demonstration
        sentences = text.split('.')
        chunks = []
        current = []
        for sent in sentences:
            if len(' '.join(current + [sent]).split()) > self.chunk_size:
                chunks.append(' '.join(current).strip())
                current = [sent]
            else:
                current.append(sent)
        if current:
            chunks.append(' '.join(current).strip())
        return [c for c in chunks if c]

class HierarchicalChunker:
    """
    Splits text into hierarchical sections and then into smaller chunks.
    """
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        # Simple split by paragraphs for demonstration
        paragraphs = text.split('\n\n')
        chunks = []
        for para in paragraphs:
            words = para.split()
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk = ' '.join(words[i:i+self.chunk_size])
                if chunk:
                    chunks.append(chunk)
        return chunks

def get_default_embeddings():
    """Return a default HuggingFace embedding model for use in chunking/retrieval."""
    return HuggingFaceEmbeddings(
        model_name="google/muril-base-cased",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def retrieve_top_k(query: str, vectorstore: FAISS, embedding_model, k: int = 3) -> Dict[str, Any]:
    """Retrieve top-k relevant documents and compute similarity metrics."""
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})
    results = retriever.get_relevant_documents(query)
    query_embedding = embedding_model.embed_query(query)
    chunk_embeddings = [embedding_model.embed_query(doc.page_content) for doc in results]
    sims = [cosine_similarity([query_embedding], [chunk_emb])[0][0] for chunk_emb in chunk_embeddings]
    avg_sim = np.mean(sims)
    total_tokens = sum(len(doc.page_content.split()) for doc in results)
    return {
        "retrieved_docs": results,
        "avg_cosine_sim": avg_sim,
        "total_tokens": total_tokens,
        "texts": [doc.page_content for doc in results]
    } 