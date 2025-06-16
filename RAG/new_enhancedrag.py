"""
new_enhancedrag.py
-----------------
Enhanced RAG system for maternal health with multilingual support, stage-aware retrieval, and cross-encoder reranking.

Usage:
    python new_enhancedrag.py --question "Your question here" --vector_store_path <path> --model_name <hf_model_name>
"""

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import numpy as np
from sentence_transformers import CrossEncoder
import re
from functools import lru_cache
import argparse

class EnhancedMedicalRAG:
    """
    Enhanced RAG system for maternal health with multilingual support,
    stage-aware retrieval, and cross-encoder reranking.
    """

    def __init__(
        self,
        vector_store: FAISS,
        model,
        tokenizer,
        temperature: float = 0.25,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_reranking: bool = True
    ):
        # Set up text generation pipeline
        
        self.vector_store = vector_store

        if model == "gpt-4-turbo":
            self.llm = ChatOpenAI(
                model=model,
                temperature=0.1  # Low temperature for factual responses
            )
        else:
            pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            device_map="auto",
            temperature=temperature,  # Lower temperature for medical accuracy
            top_p=0.95,
            batch_size=1
            )
            self.llm = HuggingFacePipeline(
                pipeline=pipe
            )
        self.use_reranking = use_reranking
        
        # Initialize reranker model if enabled
        if use_reranking:
            try:
                self.reranker = CrossEncoder(reranker_model)
                print(f"Initialized reranker model: {reranker_model}")
            except Exception as e:
                print(f"Failed to load reranker model: {e}")
                self.use_reranking = False
        
        # Language detection model - simple regex patterns for quick language detection
        self.language_patterns = {
            'hindi': re.compile(r'[\u0900-\u097F]'),   # Hindi Unicode range
            'assamese': re.compile(r'[\u0980-\u09FF]'),  # Bengali/Assamese Unicode range
            'english': re.compile(r'[a-zA-Z]')
        }
        
        # Create pregnancy stage patterns for query understanding
        self.stage_patterns = {
            'week': re.compile(r'(?:week|weeks)\s+(\d+(?:-\d+)?)|(\d+)(?:st|nd|rd|th)?\s+week', re.IGNORECASE),
            'month': re.compile(r'(?:month|months)\s+(\d+(?:-\d+)?)|(\d+)(?:st|nd|rd|th)?\s+month', re.IGNORECASE),
            'trimester': re.compile(r'(first|second|third)\s+trimester', re.IGNORECASE),
            'stage': re.compile(r'(preconception|prenatal|antenatal|perinatal|intrapartum|postpartum|postnatal)', re.IGNORECASE)
        }
        
        # Medical concern patterns for better filtering
        self.concern_patterns = {
            'pain': re.compile(r'pain|ache|hurt|sore|discomfort', re.IGNORECASE),
            'bleeding': re.compile(r'bleed|blood|spotting|hemorrhage', re.IGNORECASE),
            'movement': re.compile(r'movement|kick|moving|motion|active', re.IGNORECASE),
            'nutrition': re.compile(r'food|eat|diet|nutrition|hungry|appetite', re.IGNORECASE),
            'sleep': re.compile(r'sleep|insomnia|tired|fatigue|rest', re.IGNORECASE),
            'emotional': re.compile(r'depress|anxiety|worry|stress|mood|emotion', re.IGNORECASE)
        }

        # Define prompt templates
        self.rag_prompt = PromptTemplate(
            template="""You are a supportive maternal health assistant providing information to pregnant women and new mothers.
Use the following retrieved information to answer the question.
Response Guidelines:
1. Answer ONLY using the provided context. Do not speculate or generate information not found in the retrieved data. If the provided context doesn't fully answer the question, state the limitations. Example: "I don't have enough information to answer that accurately."


3. Be reassuring but honest. Do not minimize legitimate concerns. If the topic is serious, clearly explain when to seek medical help.


4. Use clear, non-technical language. Avoid jargon and explain complex terms simply.

5. Warning signs: When discussing symptoms or warning signs, clearly distinguish between normal variations and symptoms that require medical attention. List these separately if appropriate.

6. If you detect any signs of a possible emergency (such as heavy bleeding, severe pain, loss of consciousness, seizures, absent fetal movement, severe swelling, difficulty breathing, fever with rash, premature rupture of membranes, or vision changes):

Do NOT give any advice.

Respond ONLY with:
"IMPORTANT: This appears to be an emergency situation that requires immediate medical attention.

This chatbot cannot provide emergency medical advice. Please:
1. Contact your healthcare provider immediately OR
2. Go to the nearest emergency department OR
3. Call your local emergency number for immediate assistance.

Do not wait or delay seeking professional care."


7. Do not provide: Any specific drug brand or prescription recommendations; Any information related to fetal gender detection or selection; Any instructions or advice for suicidal/self-harm situations. Instead, respond only with the relevant crisis hotline (India: 9152987821, US: 988, UK: 0800 689 5652).

8. Do not answer off topic questions, ie: questions which are not related to  pregnancy, postpartum, or maternal/newborn wellness at all. (if off-topic, say: "I'm sorry, I can only help with pregnancy and maternal-health questions."),

9. Respond in the same language as the original question, if the language is hindi written in english, do the same. 

Context:
 {context}
Question:
 {question}
Answer:
""",

            
# Context information from trusted maternal health guidelines:
# {context}

# User Question: {question}

# Instructions:
# 1. Respond based only on the provided context. If the information isn't in the context, say "I don't have enough information to answer this question based on the provided guidelines."
# 2. Be reassuring but accurate - don't minimize legitimate health concerns.
# 3. If discussing warning signs, clearly distinguish between normal variations and concerning symptoms.
# 4. Remind users to consult healthcare providers for personal medical advice when appropriate.
# 5. Use clear, non-technical language that is easy to understand.
# 6. Respond in the same language as the original question, if the language is hindi written in english, do the same. 

# Answer:""",
            input_variables=["context", "question"]
        )

        # Create LLM chain
        self.qa_prompt_chain = LLMChain(
            llm=self.llm,
            prompt=self.rag_prompt
        )

    @lru_cache(maxsize=100)
    def translate_query_to_english(self, query: str) -> str:
        """Translate query to English if needed, with caching for efficiency."""
        # Skip translation if query already contains mostly English
        
            
        translation_prompt = PromptTemplate(
            template="""Translate the following maternal health query to English, preserving all medical terms and meaning:

Query: {query}

Translation:""",
            input_variables=["query"]
        )
        
        try:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            response = llm(translation_prompt.format_prompt(query=query).to_messages())
            return response.content.strip()
        except Exception as e:
            print(f"Translation failed: {e}")
            return query  # Return original query if translation fails

    def _detect_language(self, text: str) -> str:
        """Detect the primary language of the text."""
        counts = {
            lang: len(pattern.findall(text)) 
            for lang, pattern in self.language_patterns.items()
        }
        
        # If mixed, determine based on ratio
        total = sum(counts.values())
        if total == 0:
            return 'english'  # Default to English if no matches
            
        # Get language with highest ratio
        lang_ratios = {lang: count/total for lang, count in counts.items()}
        primary_lang = max(lang_ratios.items(), key=lambda x: x[1])[0]
        
        # Check if it's mixed (no language has > 70% dominance)
        if lang_ratios[primary_lang] < 0.7:
            return 'mixed'
            
        return primary_lang

    def detect_pregnancy_stage(self, query: str) -> Optional[str]:
        """Extract pregnancy stage information from the query."""
        query_lower = query.lower()
        
        # Map from detected stage references to standardized stages
        stage_mapping = {
            'first_trimester': ['first trimester', 'early pregnancy', '1st trimester'],
            'second_trimester': ['second trimester', 'mid pregnancy', '2nd trimester'],
            'third_trimester': ['third trimester', 'late pregnancy', '3rd trimester'],
            'labor_delivery': ['labor', 'delivery', 'birth', 'intrapartum'],
            'postpartum': ['postpartum', 'after birth', 'post delivery', 'postnatal']
        }
        
        # Check for direct stage mentions
        for stage, keywords in stage_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                return stage
                
        # Check for week mentions
        for match in self.stage_patterns['week'].finditer(query_lower):
            week_str = match.group(1) or match.group(2)
            if week_str:
                try:
                    week = int(week_str.split('-')[0])  # Take first number if range
                    if 1 <= week <= 12:
                        return 'first_trimester'
                    elif 13 <= week <= 26:
                        return 'second_trimester'
                    elif 27 <= week <= 42:
                        return 'third_trimester'
                except ValueError:
                    pass
        
        # Check for month mentions
        for match in self.stage_patterns['month'].finditer(query_lower):
            month_str = match.group(1) or match.group(2)
            if month_str:
                try:
                    month = int(month_str.split('-')[0])  # Take first number if range
                    if 1 <= month <= 3:
                        return 'first_trimester'
                    elif 4 <= month <= 6:
                        return 'second_trimester'
                    elif 7 <= month <= 10:
                        return 'third_trimester'
                except ValueError:
                    pass
                    
        return None  # No stage detected

    def detect_concerns(self, query: str) -> List[str]:
        """Detect maternal health concerns from query."""
        query_lower = query.lower()
        concerns = []
        
        for concern, pattern in self.concern_patterns.items():
            if pattern.search(query_lower):
                concerns.append(concern)
                
        return concerns

    def hybrid_retrieve(self, query: str, k: int = 10) -> List[Document]:
        """Retrieve documents using both dense and metadata-based approaches."""
        stage = self.detect_pregnancy_stage(query)
        concerns = self.detect_concerns(query)
        
        # Default retriever - dense vector similarity
        search_kwargs = {"k": k}
        
        # Add stage filter if detected
        if stage:
            # FIXED: Use a filter function that works with metadata dictionaries
            stage_filter = lambda metadata: (
                # Match documents with the same stage or no specific stage
                stage in metadata.get('pregnancy_stages', {}).get('standard_stages', [])
                if isinstance(metadata.get('pregnancy_stages', {}), dict) else False
            ) or (
                not (metadata.get('pregnancy_stages', {}).get('standard_stages', [])
                    if isinstance(metadata.get('pregnancy_stages', {}), dict) else [])
            )
            search_kwargs["filter"] = stage_filter
        
        # First retrieve based on vector similarity
        initial_docs = self.vector_store.as_retriever(search_kwargs=search_kwargs).get_relevant_documents(query)
        
        # Check if concerns are addressed in top results
        if concerns:
            concern_addressed = False
            for doc in initial_docs[:3]:  # Check top 3 docs
                doc_concerns = doc.metadata.get('patient_concerns', [])
                if any(concern in doc_concerns for concern in concerns):
                    concern_addressed = True
                    break
                    
            # If concerns not addressed, get concern-specific docs
            if not concern_addressed:
                # FIXED: Use a filter function that works with metadata dictionaries
                concern_filter = lambda metadata: any(
                    concern in metadata.get("patient_concerns", []) 
                    for concern in concerns
                )
                
                # Create a temporary retriever with this filter
                concern_kwargs = search_kwargs.copy()
                concern_kwargs["filter"] = concern_filter
                concern_kwargs["k"] = 3  # Just get a few concern-specific docs
                
                concern_docs = self.vector_store.as_retriever(
                    search_kwargs=concern_kwargs
                ).get_relevant_documents(query)
                
                # Add unique concern docs (avoid duplicates)
                existing_ids = {doc.metadata.get('chunk_id', '') for doc in initial_docs}
                for doc in concern_docs:
                    if doc.metadata.get('chunk_id', '') not in existing_ids:
                        initial_docs.append(doc)
                        
        return initial_docs

    def rerank_documents(self, query: str, docs: List[Document], top_k: int = 7) -> List[Document]:
        """Rerank documents using a cross-encoder if available."""
        if not self.use_reranking or not docs:
            return docs[:top_k]
            
        try:
            # Create query-document pairs
            query_doc_pairs = [(query, doc.page_content) for doc in docs]
            
            # Get scores from reranker
            scores = self.reranker.predict(query_doc_pairs)
            
            # Sort by score
            doc_score_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc, _ in doc_score_pairs]
            
            return reranked_docs[:top_k]
        except Exception as e:
            print(f"Reranking failed: {e}")
            return docs[:top_k]  # Return original ordering if reranking fails

    def query(self, question: str, translate: bool = True) -> Dict[str, Any]:
        """Query the enhanced RAG system with multilingual support and reranking."""
        original_question = question
        
        # Detect language
        detected_language = self._detect_language(question)
        print(f"Detected language: {detected_language}")
        # Translate if needed
        if translate :
            translated_question = self.translate_query_to_english(question)
        else:
            translated_question = question
        print(f"Translated question: {translated_question}")
        # Retrieve documents using hybrid approach
        retrieved_docs = self.hybrid_retrieve(translated_question)
        
        # Rerank if enabled
        if self.use_reranking:
            reranked_docs = self.rerank_documents(translated_question, retrieved_docs)
        else:
            reranked_docs = retrieved_docs[:7]  # Limit to top 7 if not reranking
            
        # Join contexts
        context = "\n\n".join(doc.page_content for doc in reranked_docs)
        # print(f"Context: {context}")
        # Generate answer with original question
        answer = self.qa_prompt_chain.run({
            "context": context,
            "question": original_question
        })
        
        # Clean up answer
        answer_start = answer.lower().find("answer:")
        if answer_start != -1:
            answer = answer[answer_start + 7:].strip()  # +7 to skip "answer:"
            
        # Prepare source information
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
        
        # Include detected stage and concerns for debugging/monitoring
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
        """Query without retrieval for comparison purposes."""
        original_question = question
        detected_language = self._detect_language(question)
        
        if translate and detected_language != 'english':
            translated_question = self.translate_query_to_english(question)
        else:
            translated_question = question

        # Define a prompt that doesn't use context
        direct_prompt = PromptTemplate(
            template="""You are a supportive maternal health assistant providing information to pregnant women and new mothers.
Use the following retrieved information to answer the question.
Response Guidelines:
1. Answer based on your knowledge.
3. Be reassuring but honest. Do not minimize legitimate concerns. If the topic is serious, clearly explain when to seek medical help.


4. Use clear, non-technical language. Avoid jargon and explain complex terms simply.

5. Warning signs: When discussing symptoms or warning signs, clearly distinguish between normal variations and symptoms that require medical attention. List these separately if appropriate.

6. If you detect any signs of a possible emergency, answer accordingly:

7. Do not provide: Any specific drug brand or prescription recommendations; Any information related to fetal gender detection or selection; Any instructions or advice for suicidal/self-harm situations. Instead, respond only with the relevant crisis hotline (India: 9152987821, US: 988, UK: 0800 689 5652).

8. Do not answer off topic questions, ie: questions which are not related to  pregnancy, postpartum, or maternal/newborn wellness at all. (if off-topic, say: "I'm sorry, I can only help with pregnancy and maternal-health questions."),

9. Respond in the same language as the original question, if the language is hindi written in english, do the same. 

Question:
 {question}
Answer:

""",
            input_variables=["question"]
        )

        qa_direct_chain = LLMChain(
            llm=self.llm,
            prompt=direct_prompt
        )

        answer = qa_direct_chain.run({
            "question": original_question
        })
        
        # Clean up answer
        answer_start = answer.lower().find("answer:")
        if answer_start != -1:
            answer = answer[answer_start + 7:].strip()
            
        # Include detected info for consistency
        detected_info = {
            "language": detected_language,
            "stage": self.detect_pregnancy_stage(question),
            "concerns": self.detect_concerns(question)
        }
            
        return {
            "question_original": original_question,
            "question_translated": translated_question,
            "answer": answer,
            "sources": [],  # No sources in non-RAG mode
            "detected_info": detected_info
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EnhancedMedicalRAG query.")
    parser.add_argument('--question', type=str, required=True, help='User question to query the RAG system')
    parser.add_argument('--vector_store_path', type=str, required=True, help='Path to the FAISS vector store directory')
    parser.add_argument('--model_name', type=str, default='google/muril-base-cased', help='HuggingFace model name (default: google/muril-base-cased)')
    args = parser.parse_args()

    embeddings = HuggingFaceEmbeddings(model_name=args.model_name)
    vector_store = FAISS.load_local(folder_path=args.vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    # # Load model and tokenizer (for demo, use gpt-4-turbo or HuggingFace model)
    # if args.model_name == 'gpt-4-turbo':
    #     model = args.model_name
    #     tokenizer = None
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #     model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    rag = EnhancedMedicalRAG(vector_store, model, tokenizer)
    result = rag.query(args.question)
    print("\nANSWER:\n" + result['answer'])
    print("\nSOURCES:")
    for i, source in enumerate(result['sources']):
        print(f"\n{i+1}. {source['metadata'].get('title', 'Unknown document')}")