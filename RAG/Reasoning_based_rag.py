"""
Reasoning_based_rag.py
---------------------
RAG system with multi-document reasoning for maternal health and medical documents.

Usage:
    python Reasoning_based_rag.py --question "Your question here" --vector_store_path <path> --model_name <hf_model_name>
"""

import re  # Add import for regex
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional, Tuple
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.schema import Document
import os
import time
from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

class EnhancedMedicalRAG:
    """
    Combined medical RAG system with multi-document reasoning capabilities
    for maternal health and medical documents.
    """

    def __init__(
        self,
        vector_store: FAISS,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        embeddings = None
    ):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.embeddings = embeddings or OpenAIEmbeddings()
        
        # Core prompt template for simple RAG responses
        self.rag_prompt = PromptTemplate(
            template="""You are a medical assistant. Use ONLY the provided context to answer.

Context:
{context}

Question: {question}

Instructions:
1. If context lacks information, respond: "I don't have enough information..."
2. Cite recommendations, evidence levels, sources.
3. Be concise, comprehensive.
4. Present all perspectives if conflicting.
5. Quote recommendations exactly.
6. Answer in the language of the question.

Answer:""",
            input_variables=["context", "question"]
        )

        # Simple retriever setup
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # Setup QA chain for simple retrieval mode
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.rag_prompt}
        )

    def translate_query(self, query: str) -> str:
        """Translate non-English queries to English while preserving medical meaning."""
        translation_prompt = PromptTemplate(
            template="""Translate the following user query to English, preserving its medical meaning:

Query: {query}

Translation:""",
            input_variables=["query"]
        )
        response = self.llm(translation_prompt.format_prompt(query=query).to_messages())
        return response.content.strip()

    def _extract_stage_from_query(self, query: str) -> Dict[str, Any]:
        """Extract pregnancy stage information from the query."""
        # Extract weeks (e.g., "week 12", "12th week", "weeks 12-14")
        week_pattern = re.compile(r'(?:week|weeks)\s+(\d+(?:-\d+)?)|(\d+)(?:st|nd|rd|th)?\s+week', re.IGNORECASE)
        
        # Extract months (e.g., "month 3", "3rd month", "months 4-6")
        month_pattern = re.compile(r'(?:month|months)\s+(\d+(?:-\d+)?)|(\d+)(?:st|nd|rd|th)?\s+month', re.IGNORECASE)
        
        # Extract trimesters (e.g., "first trimester", "second trimester")
        trimester_pattern = re.compile(r'(first|second|third)\s+trimester', re.IGNORECASE)
        
        stages = {
            "weeks": [],
            "months": [],
            "trimesters": []
        }
        
        # Extract weeks
        for match in week_pattern.finditer(query):
            week_str = match.group(1) or match.group(2)
            if '-' in week_str:
                start, end = map(int, week_str.split('-'))
                stages["weeks"].extend(list(range(start, end + 1)))
            else:
                stages["weeks"].append(int(week_str))
        
        # Extract months
        for match in month_pattern.finditer(query):
            month_str = match.group(1) or match.group(2)
            if '-' in month_str:
                start, end = map(int, month_str.split('-'))
                stages["months"].extend(list(range(start, end + 1)))
            else:
                stages["months"].append(int(month_str))
        
        # Extract trimesters
        for match in trimester_pattern.finditer(query):
            trimester = match.group(1).lower()
            stages["trimesters"].append(trimester)
        
        # Remove duplicates
        for key in stages:
            stages[key] = list(set(stages[key]))
        
        return stages

    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent and topics in the query."""
        query_lower = query.lower()
        
        # Detect if query is about normal variations
        is_normal_query = any(term in query_lower for term in 
                             ['normal', 'common', 'usual', 'typically', 'expected'])
        
        # Detect if query is about safety concerns
        is_safety_query = any(term in query_lower for term in 
                             ['safe', 'danger', 'risk', 'harmful', 'warning', 'symptom'])
        
        # Detect if query is about recommendations
        is_recommendation_query = any(term in query_lower for term in 
                                    ['should', 'recommend', 'advised', 'best', 'better'])
        
        # Check if the query is about a medical topic
        maternal_topics = []
        topic_keywords = {
            'fetal_development': ['development', 'growth', 'fetal', 'foetal', 'baby', 'fetus'],
            'maternal_changes': ['mother', 'maternal', 'body', 'changes', 'symptoms'],
            'nutrition': ['diet', 'food', 'nutrition', 'eat', 'meal', 'vitamin'],
            'warning_signs': ['danger', 'warning', 'risk', 'emergency', 'complication'],
            'normal_variations': ['normal', 'common', 'usual', 'typical', 'variant', 'variation'],
            'medical_tests': ['test', 'scan', 'ultrasound', 'screening', 'monitor'],
            'delivery': ['birth', 'labor', 'labour', 'delivery', 'contractions'],
            'postpartum': ['after birth', 'post-delivery', 'postpartum', 'post-natal', 'breastfeeding']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                maternal_topics.append(topic)
        
        return {
            'is_normal_query': is_normal_query,
            'is_safety_query': is_safety_query,
            'is_recommendation_query': is_recommendation_query,
            'topics': maternal_topics
        }

    def _enhanced_retrieval(self, query: str, k: int = 10) -> List[Document]:
        """
        Enhanced retrieval that considers query intent, pregnancy stage,
        and performs filtering and diversity-aware reranking.
        """
        # Extract stage and intent information
        stage_info = self._extract_stage_from_query(query)
        query_intent = self._analyze_query_intent(query)
        
        # Get more documents than needed for filtering
        initial_docs = self.vector_store.similarity_search(query, k=k*2)
        
        # Filter and score documents based on relevance to query
        scored_docs = []
        for doc in initial_docs:
            relevance_score = 0
            
            # Stage relevance
            doc_stages = doc.metadata.get('pregnancy_stages', {})
            
            # Check for stage match
            if stage_info.get('weeks') or stage_info.get('months') or stage_info.get('trimesters'):
                stage_match = False
                
                for week in stage_info.get('weeks', []):
                    if week in doc_stages.get('weeks', []):
                        stage_match = True
                        relevance_score += 3
                        break
                        
                for month in stage_info.get('months', []):
                    if month in doc_stages.get('months', []):
                        stage_match = True
                        relevance_score += 3
                        break
                        
                for trimester in stage_info.get('trimesters', []):
                    if trimester in doc_stages.get('trimesters', []):
                        stage_match = True
                        relevance_score += 3
                        break
            else:
                # No specific stage in query, so no penalty
                stage_match = True
            
            # Topic relevance
            for topic in query_intent.get('topics', []):
                if topic in doc.metadata.get('topics', []) or topic in doc.metadata.get('maternal_topics', []):
                    relevance_score += 2
            
            # Intent relevance
            if query_intent.get('is_normal_query') and doc.metadata.get('discusses_normal_variations', False):
                relevance_score += 2
                
            if query_intent.get('is_safety_query') and doc.metadata.get('contains_warning', False):
                relevance_score += 2
                
            if query_intent.get('is_recommendation_query') and doc.metadata.get('chunk_type') == 'recommendation':
                relevance_score += 2
            
            # Specialized chunks get a boost
            if doc.metadata.get('chunk_type') == 'specialized':
                relevance_score += 1
                
            # Recommendation chunks get a boost for evidence-based medicine
            if doc.metadata.get('chunk_type') == 'recommendation':
                relevance_score += 1
            
            # Add relevance score to document metadata
            doc.metadata['relevance_score'] = relevance_score
            
            # Add document to scored list if it has any relevance or if there was no stage in query
            if relevance_score > 0 or not (stage_info.get('weeks') or stage_info.get('months') or stage_info.get('trimesters')):
                scored_docs.append(doc)
        
        # Ensure diversity in the final set (topics, chunk types, etc.)
        final_docs = self._ensure_diverse_docs(scored_docs, query_intent, k)
        
        return final_docs

    def _ensure_diverse_docs(self, docs: List[Document], query_intent: Dict[str, Any], k: int) -> List[Document]:
        """Ensure diversity in the retrieved documents."""
        # Start with docs sorted by relevance score
        sorted_docs = sorted(docs, key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        
        # Ensure diversity of topics and chunk types
        diverse_docs = []
        seen_topics = set()
        seen_chunk_types = set()
        
        # First add specialized chunks if available and relevant to query topics
        for doc in [d for d in sorted_docs if d.metadata.get('chunk_type') == 'specialized']:
            # Check if this specialized chunk is relevant to the query topics
            doc_topic = doc.metadata.get('topic', '')
            if doc_topic and doc_topic in query_intent.get('topics', []):
                diverse_docs.append(doc)
                seen_topics.add(doc_topic)
        
        # Then add recommendation chunks if query is seeking recommendations
        if query_intent.get('is_recommendation_query'):
            for doc in [d for d in sorted_docs if d.metadata.get('chunk_type') == 'recommendation']:
                if len(diverse_docs) >= k:
                    break
                    
                # Skip if already added
                if doc in diverse_docs:
                    continue
                    
                diverse_docs.append(doc)
                seen_chunk_types.add('recommendation')
        
        # Then add warning chunks if query is about safety
        if query_intent.get('is_safety_query'):
            for doc in [d for d in sorted_docs if doc.metadata.get('contains_warning', False)]:
                if len(diverse_docs) >= k:
                    break
                    
                # Skip if already added
                if doc in diverse_docs:
                    continue
                    
                diverse_docs.append(doc)
                seen_chunk_types.add('warning')
        
        # Finally add remaining docs, prioritizing diversity
        for doc in sorted_docs:
            # Skip if already added
            if doc in diverse_docs:
                continue
                
            # Stop if we have enough docs
            if len(diverse_docs) >= k:
                break
                
            topics = doc.metadata.get('topics', []) or doc.metadata.get('maternal_topics', [])
            chunk_type = doc.metadata.get('chunk_type', 'text')
            
            # Check if this document adds diversity
            adds_diversity = False
            
            # If we haven't seen this chunk type yet
            if chunk_type not in seen_chunk_types:
                adds_diversity = True
                seen_chunk_types.add(chunk_type)
            
            # If it covers topics we haven't seen yet
            for topic in topics:
                if topic not in seen_topics:
                    adds_diversity = True
                    seen_topics.add(topic)
            
            # Add document if it adds diversity or is highly relevant
            if adds_diversity or doc.metadata.get('relevance_score', 0) >= 3:
                diverse_docs.append(doc)
        
        # If we still don't have enough, add the highest scoring remaining docs
        if len(diverse_docs) < k:
            remaining_docs = [doc for doc in sorted_docs if doc not in diverse_docs]
            diverse_docs.extend(remaining_docs[:k-len(diverse_docs)])
        
        return diverse_docs[:k]

    def _analyze_document_relationships(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze relationships between retrieved documents."""
        # Skip if too few documents
        if len(documents) < 2:
            return {
                'contradictions': [],
                'supports': [],
                'extensions': []
            }
            
        # Identify relationships
        contradictions = []
        supports = []
        extensions = []
        
        # Compare each pair of documents
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i+1:], i+1):
                # Skip if documents are from same section (if section path exists)
                if (doc1.metadata.get('section_path') and doc2.metadata.get('section_path') and
                    doc1.metadata.get('section_path') == doc2.metadata.get('section_path')):
                    continue
                
                # Check for contradictions or support using semantic similarity
                similarity = self._compute_semantic_similarity(doc1.page_content, doc2.page_content)
                
                # Get topics for both docs
                topics1 = set(doc1.metadata.get('topics', []) or doc1.metadata.get('maternal_topics', []))
                topics2 = set(doc2.metadata.get('topics', []) or doc2.metadata.get('maternal_topics', []))
                common_topics = topics1 & topics2
                
                # High similarity but different topics might indicate support
                if similarity > 0.85 and topics1 != topics2:
                    supports.append((i, j, similarity))
                
                # Medium similarity with same topic might indicate extension
                elif 0.6 < similarity < 0.85 and common_topics:
                    extensions.append((i, j, similarity))
                
                # Low similarity with same topic might indicate potential contradiction
                elif similarity < 0.4 and common_topics:
                    # This is just a heuristic - real contradiction detection would need more analysis
                    contradictions.append((i, j, similarity))
        
        return {
            'contradictions': contradictions,
            'supports': supports,
            'extensions': extensions
        }
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text passages using embeddings."""
        try:
            # Get embeddings for both texts
            if hasattr(self.embeddings, 'embed_query'):
                # OpenAI embeddings
                embedding1 = self.embeddings.embed_query(text1)
                embedding2 = self.embeddings.embed_query(text2)
            else:
                # Other embedding models
                embedding1 = self.embeddings.embed_documents([text1])[0]
                embedding2 = self.embeddings.embed_documents([text2])[0]
            
            # Compute cosine similarity
            import numpy as np
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
        except Exception as e:
            print(f"Error computing similarity: {e}")
            # Return a moderate similarity as fallback
            return 0.5

    def simple_query(self, question: str, filters: Dict[str, Any] = None, translate: bool = True) -> Dict[str, Any]:
        """Simple query method that uses basic RAG without multi-document reasoning."""
        original_question = question
        
        # Translate if needed
        if translate and not all(ord(c) < 128 for c in question):
            question = self.translate_query(question)
        
        # Apply filters if provided
        if filters:
            self.retriever.search_kwargs["filter"] = filters
        
        # Execute the query with simple RAG
        result = self.qa_chain({"query": question})
        
        # Extract source information
        sources = []
        for doc in result.get("source_documents", []):
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        # Return the result
        return {
            "question_original": original_question,
            "question_translated": question if translate and original_question != question else None,
            "answer": result["result"],
            "sources": sources
        }

    def query_with_reasoning(self, question: str, translate: bool = True) -> Dict[str, Any]:
        """
        Answer a query using multi-document reasoning for more complex questions.
        This approach uses a two-step process that first reasons across documents
        and then synthesizes a comprehensive answer.
        """
        original_question = question
        start_time = time.time()
        
        # Translate if needed
        if translate and not all(ord(c) < 128 for c in question):
            question = self.translate_query(question)
        
        # Retrieve documents with enhanced filtering and diversity
        documents = self._enhanced_retrieval(question, k=8)
        
        # Skip reasoning for simple queries with few documents
        if len(documents) <= 2:
            simple_result = self.simple_query(question, translate=False)
            simple_result["reasoning_method"] = "simple"
            simple_result["execution_time"] = time.time() - start_time
            return simple_result
        
        # Analyze document relationships
        relationships = self._analyze_document_relationships(documents)
        
        # Skip to simple RAG if no interesting relationships found
        if (not relationships['contradictions'] and not relationships['supports'] 
                and not relationships['extensions']):
            simple_result = self.simple_query(question, translate=False)
            simple_result["reasoning_method"] = "simple"
            simple_result["execution_time"] = time.time() - start_time
            return simple_result
        
        # Step 1: Create and execute the reasoning prompt
        reasoning_prompt = self._create_reasoning_prompt(question, documents, relationships)
        reasoning_chain = self.llm(reasoning_prompt.format_prompt().to_messages()).content
        
        # Step 2: Create and execute the synthesis prompt
        synthesis_prompt = self._create_synthesis_prompt(question, reasoning_chain)
        final_answer = self.llm(synthesis_prompt.format_prompt().to_messages()).content
        
        # Extract source information
        sources = []
        for doc in documents:
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        # Return the result
        return {
            "question_original": original_question,
            "question_translated": question if translate and original_question != question else None,
            "answer": final_answer,
            "reasoning_chain": reasoning_chain,
            "sources": sources,
            "reasoning_method": "multi-document",
            "execution_time": time.time() - start_time
        }

    def _create_reasoning_prompt(self, query: str, documents: List[Document], relationships: Dict[str, Any]) -> PromptTemplate:
        """Create a prompt for the reasoning chain."""
        template = """You are a medical reasoning expert focusing on maternal health. 
        Analyze the following documents to answer this query: "{query}"
        
        RETRIEVED DOCUMENTS:
        {documents}
        
        {relationships}
        
        TASK:
        1. Identify the key facts relevant to the query from each document
        2. Resolve any contradictions or inconsistencies
        3. Determine which facts are most relevant and medically accurate
        4. Formulate a coherent understanding that synthesizes information across all documents
        5. Note any important medical caveats or warnings that should be included
        
        Work through this step-by-step:
        """
        
        # Format documents text
        docs_text = ""
        for i, doc in enumerate(documents):
            docs_text += f"\n--- Document {i+1} ---\n"
            docs_text += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            
            if doc.metadata.get('section_path'):
                docs_text += f"Section: {' > '.join(doc.metadata.get('section_path'))}\n"
            elif doc.metadata.get('section_title'):
                docs_text += f"Section: {doc.metadata.get('section_title')}\n"
                
            docs_text += f"Type: {doc.metadata.get('chunk_type', 'text')}\n"
            
            if doc.metadata.get('pregnancy_stages'):
                stages = doc.metadata.get('pregnancy_stages', {})
                stage_info = []
                if stages.get('weeks'):
                    stage_info.append(f"Weeks: {stages.get('weeks')}")
                if stages.get('months'):
                    stage_info.append(f"Months: {stages.get('months')}")
                if stages.get('trimesters'):
                    stage_info.append(f"Trimesters: {stages.get('trimesters')}")
                
                if stage_info:
                    docs_text += f"Stages: {', '.join(stage_info)}\n"
            
            docs_text += f"Content: {doc.page_content}\n"
        
        # Format relationships text
        relationships_text = ""
        if relationships['contradictions']:
            relationships_text += "\nPotential contradictions detected between these documents:\n"
            for i, j, score in relationships['contradictions']:
                relationships_text += f"- Documents {i+1} and {j+1} (similarity: {score:.2f})\n"
        
        if relationships['supports']:
            relationships_text += "\nThese documents appear to support each other:\n"
            for i, j, score in relationships['supports']:
                relationships_text += f"- Documents {i+1} and {j+1} (similarity: {score:.2f})\n"
        
        if relationships['extensions']:
            relationships_text += "\nThese documents appear to extend each other's information:\n"
            for i, j, score in relationships['extensions']:
                relationships_text += f"- Documents {i+1} and {j+1} (similarity: {score:.2f})\n"
        
        return PromptTemplate(
            template=template,
            input_variables=["query"],
            partial_variables={
                "documents": docs_text,
                "relationships": relationships_text
            }
        )
    
    def _create_synthesis_prompt(self, query: str, reasoning: str) -> PromptTemplate:
        """Create a prompt for synthesizing the final response."""
        template = """You are a maternal health expert providing evidence-based information. 
        Based on the query and reasoning below, synthesize a comprehensive, accurate response.
        
        QUERY: {query}
        
        REASONING PROCESS:
        {reasoning}
        
        Using the reasoning above, create a response that:
        1. Directly addresses the user's query with medically accurate information
        2. Clearly ties information to the appropriate pregnancy stage (if relevant)
        3. Acknowledges any limitations in available information
        4. Includes appropriate medical caveats or warnings
        5. Maintains an authoritative but reassuring tone
        6. Cites the source/origin of key information (e.g., "according to medical guidelines...")
        
        RESPONSE:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["query"],
            partial_variables={"reasoning": reasoning}
        )

    def query(self, question: str, filters: Dict[str, Any] = None, use_reasoning: bool = True, translate: bool = True) -> Dict[str, Any]:
        """
        Main query method that chooses between simple RAG and multi-document reasoning
        based on the complexity of the question and user preference.
        """
        if use_reasoning:
            result = self.query_with_reasoning(question, translate=translate)
            
            # Apply filters if needed and re-run with simple RAG
            if filters:
                simple_result = self.simple_query(question, filters=filters, translate=False)
                # If simple result is more specific due to filters, use it
                if len(simple_result["sources"]) < len(result["sources"]):
                    return simple_result
        else:
            result = self.simple_query(question, filters=filters, translate=translate)
        
        return result

    def query_recommendations(self, question: str, evidence_level: str = None, use_reasoning: bool = True) -> Dict[str, Any]:
        """Query specifically for recommendations with optional evidence level filtering."""
        filters = {"chunk_type": "recommendation"}
        if evidence_level:
            filters["evidence_level"] = {"$regex": f"{evidence_level.lower()}.*evidence"}
        
        return self.query(question, filters=filters, use_reasoning=use_reasoning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Reasoning-based EnhancedMedicalRAG query.")
    parser.add_argument('--question', type=str, required=True, help='User question to query the RAG system')
    parser.add_argument('--vector_store_path', type=str, required=True, help='Path to the FAISS vector store directory')
    parser.add_argument('--model_name', type=str, default='google/muril-base-cased', help='HuggingFace model name (default: google/muril-base-cased)')
    args = parser.parse_args()
    
    # rag = EnhancedMedicalRAG( vector_store=vector_store, model_name="gpt-4o")

    # # Example 1: Simple query
    # result = rag.simple_query(
    #     question="What foods should I avoid during pregnancy?"
    # )
    # print(f"Simple Query Answer: {result['answer']}\n")
    # result = rag.query_with_reasoning(
    #     question="What are the signs of preeclampsia and how is it treated?"
    # )
    # print(f"Reasoning Chain: {result['reasoning_chain']}\n")
    # print(f"Final Answer: {result['answer']}\n")

    embeddings = HuggingFaceEmbeddings(model_name=args.model_name)
    vector_store = FAISS.load_local(folder_path=args.vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    rag = EnhancedMedicalRAG(vector_store, model_name=args.model_name, temperature=0.0, embeddings=embeddings)
    result = rag.query(args.question)
    print("\nANSWER:\n" + result['answer'])
    print("\nSOURCES:")
    for i, source in enumerate(result['sources']):
        print(f"\n{i+1}. {source['metadata'].get('title', 'Unknown document')}")