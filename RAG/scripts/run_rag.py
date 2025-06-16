"""
Script to run RAG queries.
"""

import argparse
from rag.core import MedicalRAG
from rag.vector_store import VectorStore

def main():
    parser = argparse.ArgumentParser(description="Run a RAG query with selectable model.")
    parser.add_argument('--question', type=str, required=True, help='User question to query the RAG system')
    parser.add_argument('--vector_store_path', type=str, required=True, help='Path to the FAISS vector store directory')
    parser.add_argument('--model', type=str, default='medical', choices=['medical', 'enhanced'], help='Which RAG model to use: medical or enhanced')
    parser.add_argument('--hf_model_name', type=str, default='google/muril-base-cased', help='HuggingFace model name for enhanced RAG (if used)')
    args = parser.parse_args()

    if args.model == 'medical':
        vector_store = VectorStore.load(args.vector_store_path)
        rag = MedicalRAG(vector_store=vector_store)
        result = rag.query(args.question)
    else:
        # Dynamic imports for enhanced model
        from rag.new_enhancedrag import EnhancedMedicalRAG
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        from transformers import AutoTokenizer
        from transformers import AutoModelForCausalLM

        embeddings = HuggingFaceEmbeddings(model_name=args.hf_model_name)
        vector_store = FAISS.load_local(folder_path=args.vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_name)
        rag = EnhancedMedicalRAG(vector_store, model, tokenizer)
        result = rag.query(args.question)

    print("\nANSWER:\n" + result['answer'])
    print("\nSOURCES:")
    for i, source in enumerate(result['sources']):
        print(f"\n{i+1}. {source['metadata'].get('title', 'Unknown document')}")

if __name__ == "__main__":
    main() 