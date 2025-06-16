"""
rag_core.py
-----------
Minimal RAG pipeline for medical QA.

Usage:
    python rag_core.py --question "Your question here" --vector_store_path <path>
"""

from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional
import argparse

class MedicalRAG:
    """RAG system for medical documents with advanced filtering and retrieval."""
    def __init__(self, vector_store: FAISS, model_name: str = "gpt-4-turbo", temperature: float = 0.0):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.rag_prompt = PromptTemplate(
            template=""" You are a medical AI assistant. Answer the medical question based only on the following context.\nIf you don't know the answer based on the context, admit that you don't know rather than making up information.\nAlways maintain patient confidentiality and provide evidence-based answers when possible.\n\nContext:\n{context}\n\nMedical Question:\n{question}\n\nAnswer:\n""",
            input_variables=["context", "question"]
        )
        self.retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.rag_prompt}
        )

    def query(self, question: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the RAG system with optional metadata filtering."""
        if filters:
            self.retriever.search_kwargs["filter"] = filters
        result = self.qa_chain({"query": question})
        sources = []
        for doc in result.get("source_documents", []):
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        return {
            "question": question,
            "answer": result["result"],
            "sources": sources
        }

    def query_recommendations(self, question: str, evidence_level: str = None) -> Dict[str, Any]:
        """Query specifically for medical recommendations with optional evidence filtering."""
        filters = {"chunk_type": "recommendation"}
        if evidence_level:
            filters["evidence_level"] = {"$regex": f"{evidence_level.lower()}.*evidence"}
        return self.query(question, filters)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run minimal MedicalRAG query.")
    parser.add_argument('--question', type=str, required=True, help='User question to query the RAG system')
    parser.add_argument('--vector_store_path', type=str, required=True, help='Path to the FAISS vector store directory')
    args = parser.parse_args()

    vector_store = FAISS.load_local(folder_path=args.vector_store_path, embeddings=None, allow_dangerous_deserialization=True)
    rag = MedicalRAG(vector_store)
    result = rag.query(args.question)
    print("\nANSWER:\n" + result['answer'])
    print("\nSOURCES:")
    for i, source in enumerate(result['sources']):
        print(f"\n{i+1}. {source['metadata'].get('title', 'Unknown document')}") 