# Basic RAG system with similarity search 
# used this for initial results 
"""
basic_rag_gpt.py
---------------
Basic RAG system using OpenAI GPT models for medical documents.

Usage:
    python basic_rag_gpt.py --question "Your question here" --vector_store_path <path>
"""

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from typing import Dict, Any
import argparse

class MedicalRAG:
    """RAG system for medical documents with advanced filtering and retrieval."""
    
    def __init__(
        self,
        vector_store: FAISS,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.0,
    ):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Create RAG prompt template
        self.rag_prompt = PromptTemplate(
            template="""You are a medical information assistant that helps healthcare professionals by providing evidence-based information from medical guidelines and literature.

Context information from medical documents:
{context}

Question: {question}

Instructions:
1. Answer based only on the provided context. If the information isn't in the context, say "I don't have enough information to answer this question based on the provided medical guidelines."
2. Cite specific recommendations, evidence levels, and document sources when available.
3. Be concise but comprehensive.
4. If multiple recommendations or conflicting guidance exists in the context, present all perspectives.
5. When quoting recommendations, preserve their exact wording.

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Setup retrieval chain
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        
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
            # Apply metadata filters to the retriever
            self.retriever.search_kwargs["filter"] = filters
        
        # Get answer
        result = self.qa_chain({"query": question})
        
        # Format response
        sources = []
        for doc in result.get("source_documents", []):
            # Add source information
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
            # Filter by evidence level (e.g., "high", "moderate", "low")
            filters["evidence_level"] = {"$regex": f"{evidence_level.lower()}.*evidence"}
        
        return self.query(question, filters)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run basic GPT MedicalRAG query.")
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

    # rag = MedicalRAG(vector_store)
        
    #     # # Run query if provided
    #     # if args.query:
    #     #     if args.filter_recommendations:
    #     #         result = rag.query_recommendations(args.query, args.evidence_level)
    #     #     else:
    # query="How does folic acid help during pregnancy?"
    # result = rag.query(query)
            
    # print("\n" + "="*50)
    # print("QUESTION:")
    # print(result["question"])
    # print("\nANSWER:")
    # print(result["answer"])
    # print("\nSOURCES:")
    # for i, source in enumerate(result["sources"]):
    #     print(f"\n{i+1}. {source['metadata'].get('document_title', 'Unknown document')}")
    #     if "section_path" in source["metadata"]:
    #         print(f"   Section: {' > '.join(source['metadata']['section_path'])}")
    #     if "chunk_type" in source["metadata"]:
    #         print(f"   Type: {source['metadata']['chunk_type']}")
    #     if "evidence_level" in source["metadata"]:
    #         print(f"   Evidence level: {source['metadata']['evidence_level']}")
