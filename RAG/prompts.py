"""
prompts.py
----------
Provides prompt templates for RAG systems, including the main RAG prompt for medical QA.

Variables:
    - RAG_PROMPT: PromptTemplate for medical AI assistant.
"""

from langchain.prompts import PromptTemplate

# Main RAG prompt template for medical QA
RAG_PROMPT = PromptTemplate(
    template=""" You are a medical AI assistant. Answer the medical question based only on the following context.\nIf you don't know the answer based on the context, admit that you don't know rather than making up information.\nAlways maintain patient confidentiality and provide evidence-based answers when possible.\n\nContext:\n{context}\n\nMedical Question:\n{question}\n\nAnswer:\n""",
    input_variables=["context", "question"]
) 