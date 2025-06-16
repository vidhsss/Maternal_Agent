from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import LLMChain

class MedicalRAG:
    """RAG system for medical documents with filtering, query translation, and relationship awareness."""

    def __init__(
        self,
        vector_store: FAISS,
        model,
        tokenizer,
        temperature: float = 0.25,
    ):
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=temperature,  # Lower temperature for medical accuracy
            top_p=0.95
        )
        self.vector_store = vector_store
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Updated prompt template to include both original and translated questions
        self.rag_prompt = PromptTemplate(
            template="""You are a medical information assistant that helps patients by providing evidence-based information from medical guidelines and literature, who don't have any medical knowledge.

Context information from medical documents:
{context}

Question: {question}


Instructions:
1. Respond based only on the provided context. If the information isn't in the context, say "I don't have enough information to answer this question based on the provided medical guidelines."
2. Cite specific recommendations, evidence levels, and document sources when available.
3. Be concise but comprehensive.
4. If multiple recommendations or conflicting guidance exists in the context, present all perspectives.
5. When quoting recommendations, preserve their exact wording.
6. Respond in the language of the original question, ie if the original question is in Hindi, respond in Hindi.

Answer:""",
            input_variables=["context", "question"]
        )

        # Simple retriever without reranking
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        self.qa_prompt_chain = LLMChain(
    llm=self.llm,
    prompt=self.rag_prompt
)

    def translate_query_to_english(self, query: str) -> str:
        translation_prompt = PromptTemplate(
            template="""Translate the following user query to English, preserving its medical meaning:

Query: {query}

Translation:""",
            input_variables=["query"]
        )
        llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
        response = llm(translation_prompt.format_prompt(query=query).to_messages())
        return response.content.strip()

    def query(self, question: str, filters: Dict[str, Any] = None, translate: bool = True) -> Dict[str, Any]:
        """Query the RAG system with translation support."""

        original_question = question
        translated_question = question

        if translate:
            translated_question = self.translate_query_to_english(question)

        if filters:
            self.retriever.search_kwargs["filter"] = filters

        # Retrieve with translated question
        retrieved_docs = self.retriever.get_relevant_documents(translated_question)

        # Join contexts
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Generate answer with original question
        answer = self.qa_prompt_chain.run({
            "context": context,
            "question": original_question
        })
        answer_start = answer.lower().find("answer:")
        if answer_start != -1:
            answer = answer[answer_start:].strip()
        sources = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in retrieved_docs
        ]

        return {
            "question_original": original_question,
            "question_translated": translated_question,
            "answer": answer,
            "sources": sources
        }
    def query_without_rag(self, question: str, translate: bool = True) -> Dict[str, Any]:
        """Query GPT model directly without using retrieval (no RAG)."""

        original_question = question
        translated_question = question
        if translate:
            translated_question = self.translate_query_to_english(question)

        # Define a prompt that doesn't use context
        direct_prompt = PromptTemplate(
            template="""You are a medical information assistant helping patients by providing evidence-based information.

Question: {question}

Instructions:
1. Answer based on your existing medical knowledge.
2. If unsure, say "I'm not sure based on my current knowledge."
3. Be concise but clear.
4. Answer in the language of the question (e.g., answer in Hindi if question is in Hindi).

Answer:""",
            input_variables=["question"]
        )

        qa_direct_chain = LLMChain(
            llm=self.llm,
            prompt=direct_prompt
        )

        answer = qa_direct_chain.run({
            "question": original_question
        })
        answer_start = answer.lower().find("answer:")
        if answer_start != -1:
            answer = answer[answer_start:].strip()
        return {
            "question_original": original_question,
            "question_translated": translated_question,
            "answer": answer,
            "sources": []  # No sources in non-RAG mode
        }

"""
Basic_rag_multilingual_with_no_rag.py
-------------------------------------
Basic multilingual RAG system for medical documents.

Usage:
    python Basic_rag_multilingual_with_no_rag.py --question "Your question here" --vector_store_path <path> --model_name <hf_model_name>
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Basic Multilingual MedicalRAG query.")
    parser.add_argument('--question', type=str, required=True, help='User question to query the RAG system')
    parser.add_argument('--vector_store_path', type=str, required=True, help='Path to the FAISS vector store directory')
    parser.add_argument('--model_name', type=str, default='google/muril-base-cased', help='HuggingFace model name (default: google/muril-base-cased)')
    args = parser.parse_args()

    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from transformers import AutoTokenizer, AutoModelForCausalLM
    embeddings = HuggingFaceEmbeddings(model_name=args.model_name)
    vector_store = FAISS.load_local(folder_path=args.vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    rag = MedicalRAG(vector_store, model=model, tokenizer=tokenizer)
    result = rag.query(args.question)
    print("\nANSWER:\n" + result['answer'])
    print("\nSOURCES:")
    for i, source in enumerate(result['sources']):
        print(f"\n{i+1}. {source['metadata'].get('title', 'Unknown document')}")

     