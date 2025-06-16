"""
basic_rag_huggingface.py
------------------------
Basic RAG system using HuggingFace models for medical documents.

Usage:
    python basic_rag_huggingface.py --question "Your question here" --vector_store_path <path> --model_name <hf_model_name>
"""

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import argparse

class MedicalRAG:
    """RAG system for medical documents with advanced filtering and retrieval."""
    
    def __init__(
        self,
        vector_store: FAISS,
        model_name: str =  "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct",
        temperature: float = 0.2,
        model=None,
        tokenizer=None
    ):
        self.vector_store = vector_store
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=temperature,  # Lower temperature for medical accuracy
            top_p=0.95
        )
        self.device = "cuda"
        # Create LangChain wrapper
        self.llm = HuggingFacePipeline(pipeline=pipe)
        # Create RAG prompt template
        self.template=""" You are a medical AI assistant. Answer the medical question based only on the following context.
        If you don't know the answer based on the context, admit that you don't know rather than making up information.
        Always maintain patient confidentiality and provide evidence-based answers when possible.
        
        Context:
        {context}
        
        Medical Question:
        {question}
        
        Answer:
        
        """
        self.rag_prompt = PromptTemplate(
            template=self.template,
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
        answer = result["result"]
        # Find "Answer:" and return everything after it
        answer_start = answer.lower().find("answer:")
        if answer_start != -1:
            answer = answer[answer_start:].strip()
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
            "answer": answer,
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
    parser = argparse.ArgumentParser(description="Run basic HuggingFace MedicalRAG query.")
    parser.add_argument('--question', type=str, required=True, help='User question to query the RAG system')
    parser.add_argument('--vector_store_path', type=str, required=True, help='Path to the FAISS vector store directory')
    parser.add_argument('--model_name', type=str, default='/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct', help='HuggingFace model name (default: Llama-3.2-3B-Instruct)')
    args = parser.parse_args()

    embeddings = HuggingFaceEmbeddings(model_name=args.model_name)
    vector_store = FAISS.load_local(folder_path=args.vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to("cuda")
    rag = MedicalRAG(vector_store, model_name=args.model_name, model=model, tokenizer=tokenizer)
    result = rag.query(args.question)
    print("\nANSWER:\n" + result['answer'])
    print("\nSOURCES:")
    for i, source in enumerate(result['sources']):
        print(f"\n{i+1}. {source['metadata'].get('title', 'Unknown document')}")
    


