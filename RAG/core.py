"""
core.py
-------
Main RAG pipeline classes for medical and maternal health document QA.

Usage:
    python core.py --question "Your question here" --vector_store_path <path> --model medical|enhanced

Dependencies:
- chunking.py: for chunking strategies
- vector_store.py: for vector store management
- reranking.py: for reranking logic
- multilingual.py: for language detection/translation
- document_processing.py: for PDF/text processing

Public API:
- MedicalRAG
- EnhancedMedicalRAG
"""

# Standard library
from functools import lru_cache
from typing import List, Dict, Any, Optional
import re

# Third-party
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline

# --- MedicalRAG: Basic RAG pipeline ---
class MedicalRAG:
    """
    Basic RAG system for medical documents with context-based retrieval and answering.
    """
    def __init__(self, vector_store: FAISS, model_name: str = "gpt-4-turbo", temperature: float = 0.0):
        """
        Args:
            vector_store: FAISS vector store instance
            model_name: Name of the LLM to use
            temperature: LLM temperature
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.rag_prompt = PromptTemplate(
            template="""You are a medical information assistant that helps healthcare professionals by providing evidence-based information from medical guidelines and literature.\n\nContext information from medical documents:\n{context}\n\nQuestion: {question}\n\nInstructions:\n1. Answer based only on the provided context. If the information isn't in the context, say \"I don't have enough information to answer this question based on the provided medical guidelines.\"\n2. Cite specific recommendations, evidence levels, and document sources when available.\n3. Be concise but comprehensive.\n4. If multiple recommendations or conflicting guidance exists in the context, present all perspectives.\n5. When quoting recommendations, preserve their exact wording.\n\nAnswer:""",
            input_variables=["context", "question"]
        )
        self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.rag_prompt}
        )

    def query(self, question: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the RAG system with optional metadata filtering.
        Args:
            question: User question
            filters: Optional metadata filters
        Returns:
            dict with question, answer, and sources
        """
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

    def query_recommendations(self, question: str, evidence_level: Optional[str] = None) -> Dict[str, Any]:
        """
        Query specifically for medical recommendations with optional evidence filtering.
        Args:
            question: User question
            evidence_level: Optional evidence level filter
        Returns:
            dict with question, answer, and sources
        """
        filters = {"chunk_type": "recommendation"}
        if evidence_level:
            filters["evidence_level"] = {"$regex": f"{evidence_level.lower()}.*evidence"}
        return self.query(question, filters)

# --- EnhancedMedicalRAG: Advanced, multilingual, stage-aware RAG ---
class EnhancedMedicalRAG:
    """
    Enhanced RAG system for maternal health with multilingual support, stage-aware retrieval, and cross-encoder reranking.
    """
    def __init__(self, vector_store: FAISS, model, tokenizer, temperature: float = 0.25, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", use_reranking: bool = True):
        """
        Args:
            vector_store: FAISS vector store instance
            model: LLM model or model name
            tokenizer: Tokenizer for the model
            temperature: LLM temperature
            reranker_model: Cross-encoder model for reranking
            use_reranking: Whether to use reranking
        """
        self.vector_store = vector_store
        if model == "gpt-4-turbo":
            self.llm = ChatOpenAI(model=model, temperature=0.1)
        else:
            from transformers import pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                device_map="auto",
                temperature=temperature,
                top_p=0.95,
                batch_size=1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
        self.use_reranking = use_reranking
        if use_reranking:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(reranker_model)
            except Exception as e:
                print(f"Failed to load reranker model: {e}")
                self.use_reranking = False
        self.language_patterns = {
            'hindi': re.compile(r'[\u0900-\u097F]'),
            'assamese': re.compile(r'[\u0980-\u09FF]'),
            'english': re.compile(r'[a-zA-Z]')
        }
        self.stage_patterns = {
            'week': re.compile(r'(?:week|weeks)\s+(\d+(?:-\d+)?)|(\d+)(?:st|nd|rd|th)?\s+week', re.IGNORECASE),
            'month': re.compile(r'(?:month|months)\s+(\d+(?:-\d+)?)|(\d+)(?:st|nd|rd|th)?\s+month', re.IGNORECASE),
            'trimester': re.compile(r'(first|second|third)\s+trimester', re.IGNORECASE),
            'stage': re.compile(r'(preconception|prenatal|antenatal|perinatal|intrapartum|postpartum|postnatal)', re.IGNORECASE)
        }
        self.concern_patterns = {
            'pain': re.compile(r'pain|ache|hurt|sore|discomfort', re.IGNORECASE),
            'bleeding': re.compile(r'bleed|blood|spotting|hemorrhage', re.IGNORECASE),
            'movement': re.compile(r'movement|kick|moving|motion|active', re.IGNORECASE),
            'nutrition': re.compile(r'food|eat|diet|nutrition|hungry|appetite', re.IGNORECASE),
            'sleep': re.compile(r'sleep|insomnia|tired|fatigue|rest', re.IGNORECASE),
            'emotional': re.compile(r'depress|anxiety|worry|stress|mood|emotion', re.IGNORECASE)
        }
        self.rag_prompt = PromptTemplate(
            template="""You are a maternal health assistant providing information to pregnant women and new mothers.\n\nContext information from trusted maternal health guidelines:\n{context}\n\nUser Question: {question}\n\nInstructions:\n1. Respond based only on the provided context. If the information isn't in the context, say \"I don't have enough information to answer this question based on the provided guidelines.\"\n2. Be reassuring but accurate - don't minimize legitimate health concerns.\n3. If discussing warning signs, clearly distinguish between normal variations and concerning symptoms.\n4. Remind users to consult healthcare providers for personal medical advice when appropriate.\n5. Use clear, non-technical language that is easy to understand.\n6. Respond in the same language as the original question, if the language is hindi written in english, do the same. \n\nAnswer:""",
            input_variables=["context", "question"]
        )
        self.qa_prompt_chain = LLMChain(llm=self.llm, prompt=self.rag_prompt)

    @lru_cache(maxsize=100)
    def translate_query_to_english(self, query: str) -> str:
        """
        Translate query to English if needed, with caching for efficiency.
        """
        translation_prompt = PromptTemplate(
            template="""Translate the following maternal health query to English, preserving all medical terms and meaning:\n\nQuery: {query}\n\nTranslation:""",
            input_variables=["query"]
        )
        try:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            response = llm(translation_prompt.format_prompt(query=query).to_messages())
            return response.content.strip()
        except Exception as e:
            print(f"Translation failed: {e}")
            return query

    def _detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.
        """
        counts = {lang: len(pattern.findall(text)) for lang, pattern in self.language_patterns.items()}
        total = sum(counts.values())
        if total == 0:
            return 'english'
        lang_ratios = {lang: count/total for lang, count in counts.items()}
        primary_lang = max(lang_ratios.items(), key=lambda x: x[1])[0]
        if lang_ratios[primary_lang] < 0.7:
            return 'mixed'
        return primary_lang

    def detect_pregnancy_stage(self, query: str) -> Optional[str]:
        """
        Extract pregnancy stage information from the query.
        """
        query_lower = query.lower()
        stage_mapping = {
            'first_trimester': ['first trimester', 'early pregnancy', '1st trimester'],
            'second_trimester': ['second trimester', 'mid pregnancy', '2nd trimester'],
            'third_trimester': ['third trimester', 'late pregnancy', '3rd trimester'],
            'labor_delivery': ['labor', 'delivery', 'birth', 'intrapartum'],
            'postpartum': ['postpartum', 'after birth', 'post delivery', 'postnatal']
        }
        for stage, keywords in stage_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                return stage
        for match in self.stage_patterns['week'].finditer(query_lower):
            week_str = match.group(1) or match.group(2)
            if week_str:
                try:
                    week = int(week_str.split('-')[0])
                    if 1 <= week <= 12:
                        return 'first_trimester'
                    elif 13 <= week <= 26:
                        return 'second_trimester'
                    elif 27 <= week <= 42:
                        return 'third_trimester'
                except ValueError:
                    pass
        for match in self.stage_patterns['month'].finditer(query_lower):
            month_str = match.group(1) or match.group(2)
            if month_str:
                try:
                    month = int(month_str.split('-')[0])
                    if 1 <= month <= 3:
                        return 'first_trimester'
                    elif 4 <= month <= 6:
                        return 'second_trimester'
                    elif 7 <= month <= 10:
                        return 'third_trimester'
                except ValueError:
                    pass
        return None

    def detect_concerns(self, query: str) -> List[str]:
        """
        Detect maternal health concerns from query.
        """
        query_lower = query.lower()
        concerns = []
        for concern, pattern in self.concern_patterns.items():
            if pattern.search(query_lower):
                concerns.append(concern)
        return concerns

    def hybrid_retrieve(self, query: str, k: int = 10) -> List[Any]:
        """
        Retrieve documents using both dense and metadata-based approaches.
        """
        stage = self.detect_pregnancy_stage(query)
        concerns = self.detect_concerns(query)
        search_kwargs = {"k": k}
        if stage:
            stage_filter = lambda metadata: (
                stage in metadata.get('pregnancy_stages', {}).get('standard_stages', [])
                if isinstance(metadata.get('pregnancy_stages', {}), dict) else False
            ) or (
                not (metadata.get('pregnancy_stages', {}).get('standard_stages', [])
                    if isinstance(metadata.get('pregnancy_stages', {}), dict) else [])
            )
            search_kwargs["filter"] = stage_filter
        initial_docs = self.vector_store.as_retriever(search_kwargs=search_kwargs).get_relevant_documents(query)
        if concerns:
            concern_addressed = False
            for doc in initial_docs[:3]:
                doc_concerns = doc.metadata.get('patient_concerns', [])
                if any(concern in doc_concerns for concern in concerns):
                    concern_addressed = True
                    break
            if not concern_addressed:
                concern_filter = lambda metadata: any(
                    concern in metadata.get("patient_concerns", []) 
                    for concern in concerns
                )
                concern_kwargs = search_kwargs.copy()
                concern_kwargs["filter"] = concern_filter
                concern_kwargs["k"] = 3
                concern_docs = self.vector_store.as_retriever(
                    search_kwargs=concern_kwargs
                ).get_relevant_documents(query)
                existing_ids = {doc.metadata.get('chunk_id', '') for doc in initial_docs}
                for doc in concern_docs:
                    if doc.metadata.get('chunk_id', '') not in existing_ids:
                        initial_docs.append(doc)
        return initial_docs

    def rerank_documents(self, query: str, docs: List[Any], top_k: int = 7) -> List[Any]:
        """
        Rerank documents using a cross-encoder if available.
        """
        if not self.use_reranking or not docs:
            return docs[:top_k]
        try:
            query_doc_pairs = [(query, doc.page_content) for doc in docs]
            scores = self.reranker.predict(query_doc_pairs)
            doc_score_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc, _ in doc_score_pairs]
            return reranked_docs[:top_k]
        except Exception as e:
            print(f"Reranking failed: {e}")
            return docs[:top_k]

    def query(self, question: str, translate: bool = True) -> Dict[str, Any]:
        """
        Query the enhanced RAG system with multilingual support and reranking.
        Args:
            question: User question
            translate: Whether to translate the question to English
        Returns:
            dict with question_original, question_translated, answer, sources, detected_info
        """
        original_question = question
        detected_language = self._detect_language(question)
        if translate:
            translated_question = self.translate_query_to_english(question)
        else:
            translated_question = question
        retrieved_docs = self.hybrid_retrieve(translated_question)
        if self.use_reranking:
            reranked_docs = self.rerank_documents(translated_question, retrieved_docs)
        else:
            reranked_docs = retrieved_docs[:7]
        context = "\n\n".join(doc.page_content for doc in reranked_docs)
        answer = self.qa_prompt_chain.run({
            "context": context,
            "question": original_question
        })
        answer_start = answer.lower().find("answer:")
        if answer_start != -1:
            answer = answer[answer_start + 7:].strip()
        sources = [
            {
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": {
                    k: v for k, v in doc.metadata.items() 
                    if k in ['title', 'source', 'pregnancy_stages', 'content_type']
                }
            }
            for doc in reranked_docs
        ]
        detected_info = {
            "language": detected_language,
            "stage": self.detect_pregnancy_stage(question),
            "concerns": self.detect_concerns(question)
        }
        return {
            "question_original": original_question,
            "question_translated": translated_question,
            "answer": answer,
            "sources": sources,
            "detected_info": detected_info
        }

    def query_without_rag(self, question: str, translate: bool = True) -> Dict[str, Any]:
        """
        Query without retrieval for comparison purposes.
        Args:
            question: User question
            translate: Whether to translate the question to English
        Returns:
            dict with question_original, question_translated, answer, sources, detected_info
        """
        original_question = question
        detected_language = self._detect_language(question)
        if translate and detected_language != 'english':
            translated_question = self.translate_query_to_english(question)
        else:
            translated_question = question
        direct_prompt = PromptTemplate(
            template="""You are a maternal health assistant helping pregnant women and new mothers.\n\nQuestion: {question}\n\nInstructions:\n1. Answer based on your existing medical knowledge.\n2. Be reassuring but accurate - don't minimize legitimate health concerns.\n3. Remind users to consult healthcare providers for personal medical advice.\n4. Use clear, non-technical language that is easy to understand.\n5. Answer in the same language as the question (Hindi, Assamese, or English).\n\nAnswer:""",
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
            answer = answer[answer_start + 7:].strip()
        detected_info = {
            "language": detected_language,
            "stage": self.detect_pregnancy_stage(question),
            "concerns": self.detect_concerns(question)
        }
        return {
            "question_original": original_question,
            "question_translated": translated_question,
            "answer": answer,
            "sources": [],
            "detected_info": detected_info
        }

# --- Public API ---
__all__ = ["MedicalRAG", "EnhancedMedicalRAG"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MedicalRAG or EnhancedMedicalRAG query.")
    parser.add_argument('--question', type=str, required=True, help='User question to query the RAG system')
    parser.add_argument('--vector_store_path', type=str, required=True, help='Path to the FAISS vector store directory')
    parser.add_argument('--model', type=str, default='medical', choices=['medical', 'enhanced'], help='Which RAG model to use: medical or enhanced')
    parser.add_argument('--hf_model_name', type=str, default='google/muril-base-cased', help='HuggingFace model name for enhanced RAG (if used)')
    args = parser.parse_args()

    from .vector_store import VectorStore
    if args.model == 'medical':
        vector_store = VectorStore.load(args.vector_store_path)
        rag = MedicalRAG(vector_store=vector_store)
        result = rag.query(args.question)
    else:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        from transformers import AutoTokenizer, AutoModelForCausalLM
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