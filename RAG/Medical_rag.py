"""
Medical_rag.py
-------------
Medical RAG pipeline for processing medical PDFs, building a vector store, and running queries.

Usage:
    python Medical_rag.py --pdf_path <input.pdf> --output_db <output_db_dir> --query "Your question here"
"""
#Llama 3.2 3b and pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb embedding in the database

import os
import re
import uuid
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

# Document loading and processing
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Vector database and embeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LLM and generation
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class MedicalPDFProcessor:
    """Process medical PDFs with specialized techniques for handling medical content."""
    
    def __init__(self):
        # Download necessary NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        self.stop_words = set(stopwords.words('english'))
        
        # Medical-specific abbreviations and terms
        self.medical_abbreviations = {
            "pt": "patient", "pts": "patients", "dx": "diagnosis", 
            "tx": "treatment", "hx": "history", "fx": "fracture",
            "sx": "symptoms", "rx": "prescription", "appt": "appointment",
            "vs": "vital signs", "yo": "year old", "y/o": "year old",
            "labs": "laboratory tests", "hpi": "history of present illness",
            "w/": "with", "s/p": "status post", "c/o": "complains of",
            "p/w": "presents with", "h/o": "history of", "f/u": "follow up"
        }
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file with medical-specific preprocessing."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extract text from each page
            for page in reader.pages:
                text += page.extract_text() + "\n"
                
        # Basic cleaning
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize medical text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Expand common medical abbreviations
        for abbr, expansion in self.medical_abbreviations.items():
            # Only replace when it's a whole word (with word boundaries)
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text, flags=re.IGNORECASE)
            
        # Normalize spacing after periods for better sentence splitting
        text = re.sub(r'\.(?! )', '. ', text)
        
        return text
    
    def split_into_sections(self, text: str) -> List[str]:
        """Split medical document into logical sections based on common headers."""
        common_sections = [
            "History", "Physical Examination", "Assessment", "Plan", "Diagnosis",
            "Chief Complaint", "Past Medical History", "Medications", "Allergies",
            "Family History", "Social History", "Review of Systems", "Labs",
            "Imaging", "Discussion", "Conclusion", "Recommendations"
        ]
        
        # Create regex pattern for section headers
        pattern = r'(?i)(?:^|\n)(' + '|'.join(re.escape(s) for s in common_sections) + r')(?::|:)?\s*(?:\n|\s)'
        
        # Find all section headers with their positions
        matches = list(re.finditer(pattern, text))
        
        sections = []
        
        # Extract each section
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i < len(matches) - 1 else len(text)
            
            # Get the section header and content
            header = match.group(1)
            content = text[start:end].strip()
            
            # Add the section
            sections.append(f"{header}:\n{content}")
            
        # If no sections were identified, return the whole text as one section
        if not sections:
            sections = [text]
            
        return sections
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a medical PDF and return LangChain Document objects."""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Try to split into sections if possible
        sections = self.split_into_sections(text)
        
        # Create Document objects
        documents = []
        
        filename = os.path.basename(pdf_path)
        
        for i, section in enumerate(sections):
            # Create metadata to track source and section
            metadata = {
                "source": filename,
                "page": i,  # Using i as a proxy for page if real page info isn't available
                "section": section.split(":", 1)[0] if ":" in section else "General"
            }
            
            documents.append(Document(page_content=section, metadata=metadata))
            
        return documents


class MedicalRAGPipeline:
    """
    Retrieval-Augmented Generation pipeline specialized for medical documents.
    Uses FAISS for vector storage and optimized for medical domain content.
    """

    def __init__(
        self,
        embedding_model: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        llm_model: str = "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct",
        db_directory: str = "medical_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_gpu: bool = None
    ):
        """
        Initialize the Medical RAG pipeline.
        
        Args:
            embedding_model: Hugging Face model for embeddings (preferably biomedical)
            llm_model: Hugging Face model for generation (preferably with medical knowledge)
            db_directory: Directory to store the FAISS database
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            use_gpu: Whether to use GPU. If None, will auto-detect.
        """
        self.embedding_model = embedding_model
        self.llm_model = "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct"
        self.db_directory = db_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Determine device
        if use_gpu is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            
        print(f"Using device: {self.device}")
        
        # Initialize PDF processor
        self.pdf_processor = MedicalPDFProcessor()
        
        # Initialize embeddings - use biomedical-specific models if available
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Initialize text splitter with medical domain settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize LLM - use a model with medical knowledge if possible
        try:
            tokenizer = AutoTokenizer.from_pretrained(llm_model)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model
            )
            
            # Create text generation pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,  # Lower temperature for medical accuracy
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            # Create LangChain wrapper
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("Falling back to smaller model...")
            
            # Fallback to a smaller, more widely compatible model
            try:
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
                model = AutoModelForCausalLM.from_pretrained(
                    "google/flan-t5-base",
                    device_map=self.device,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.1
                )
                
                self.llm = HuggingFacePipeline(pipeline=pipe)
                
            except Exception as inner_e:
                print(f"Error loading fallback model: {inner_e}")
                raise RuntimeError("Could not initialize LLM. Please check model compatibility.")
        
        # Initialize or load vectorstore if it exists
        index_path = os.path.join(db_directory, "index.faiss")
        if os.path.exists(index_path):
            self.db = FAISS.load_local(
                folder_path=db_directory,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.db = None

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Process a medical PDF into Document objects.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of processed Document objects
        """
        # Extract raw documents from PDF
        raw_documents = self.pdf_processor.process_pdf(pdf_path)
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(raw_documents)
        
        print(f"Processed {pdf_path} into {len(chunks)} chunks")
        
        return chunks

    def process_pdf_directory(self, directory_path: str) -> List[Document]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all Document chunks from all PDFs
        """
        all_chunks = []
        
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return all_chunks
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            chunks = self.process_pdf(pdf_path)
            all_chunks.extend(chunks)
            
        return all_chunks

    def store_documents(self, chunks: List[Document], collection_name: Optional[str] = None) -> str:
        """
        Store document chunks in FAISS.
        
        Args:
            chunks: Document chunks to store
            collection_name: Optional name for the collection
            
        Returns:
            Collection name
        """
        # Generate a collection name if not provided
        if collection_name is None:
            collection_name = f"medical_{uuid.uuid4().hex[:8]}"
        
        # Create database directory if it doesn't exist
        index_path = os.path.join(self.db_directory, collection_name)
        os.makedirs(index_path, exist_ok=True)
        
        # Initialize FAISS index from documents
        self.db = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Save to disk
        self.db.save_local(index_path)
        
        print(f"Stored {len(chunks)} chunks in collection '{collection_name}'")
        
        return collection_name

    def retrieve_chunks(
        self, 
        query: str, 
        n_results: int = 5,  # Return more results for medical context
        collection_name: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve relevant document chunks for a medical query.
        
        Args:
            query: The query string
            n_results: Number of chunks to retrieve
            collection_name: Optional collection to search in
            
        Returns:
            List of relevant document chunks
        """
        if self.db is None:
            raise ValueError("No database has been created or loaded.")
            
        # If collection name is provided, load that collection
        if collection_name:
            index_path = os.path.join(self.db_directory, collection_name)
            if os.path.exists(index_path):
                db = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                raise ValueError(f"Collection '{collection_name}' not found.")
        else:
            db = self.db
            
        # Retrieve chunks with MMR for diversity
        chunks = db.max_marginal_relevance_search(
            query, 
            k=n_results,
            fetch_k=n_results*2  # Fetch more candidates for diversity
        )
        
        return chunks

    def query(
        self, 
        query: str, 
        n_results: int = 5,
        collection_name: Optional[str] = None,
        use_mmr: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a medical query using the RAG pipeline.
        
        Args:
            query: The medical query string
            n_results: Number of chunks to retrieve
            collection_name: Optional collection to search in
            use_mmr: Whether to use Maximum Marginal Relevance for diverse results
            
        Returns:
            Dictionary with the query result and relevant chunks
        """
        if self.db is None:
            raise ValueError("No database has been created or loaded.")
            
        # If collection name is provided, load that collection
        if collection_name:
            index_path = os.path.join(self.db_directory, collection_name)
            if os.path.exists(index_path):
                db = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                raise ValueError(f"Collection '{collection_name}' not found.")
        else:
            db = self.db
            
        # Create the appropriate retriever
        search_kwargs = {"k": n_results}
        if use_mmr:
            search_type = "mmr"
            search_kwargs["fetch_k"] = n_results * 2  # Fetch more for diversity
        else:
            search_type = "similarity"
            
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        # Create a medical-specific prompt template
        template = """
        You are a medical AI assistant. Answer the medical question based only on the following context.
        If you don't know the answer based on the context, admit that you don't know rather than making up information.
        Always maintain patient confidentiality and provide evidence-based answers when possible.
        
        Context:
        {context}
        
        Medical Question:
        {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # Create a QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        # Execute the query
        result = qa_chain({"query": query})
        
        # Also retrieve the chunks for reference
        chunks = self.retrieve_chunks(query, n_results, collection_name)
        
        return {
            "query": query,
            "answer": result["result"],
            "chunks": chunks
        }
        
    def ingest_and_store(
        self, 
        pdf_path: str, 
        collection_name: Optional[str] = None
    ) -> str:
        """
        Complete pipeline to ingest, process, and store PDF documents.
        
        Args:
            pdf_path: Path to PDF file or directory containing PDFs
            collection_name: Optional name for the collection
            
        Returns:
            Collection name
        """
        # Check if path is file or directory
        if os.path.isdir(pdf_path):
            # Process directory of PDFs
            chunks = self.process_pdf_directory(pdf_path)
        else:
            # Process single PDF
            chunks = self.process_pdf(pdf_path)
            
        if not chunks:
            raise ValueError(f"No content could be extracted from {pdf_path}")
            
        # Store chunks in FAISS
        collection_name = self.store_documents(chunks, collection_name)
        
        return collection_name


# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process a medical PDF, build a vector store, and run a query.")
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to the input PDF file')
    parser.add_argument('--output_db', type=str, required=True, help='Directory to save the vector store')
    parser.add_argument('--query', type=str, required=True, help='Question to query the RAG system')
    args = parser.parse_args()

    pipeline = MedicalRAGPipeline()
    docs = pipeline.process_pdf(args.pdf_path)
    pipeline.store_documents(docs, collection_name=args.output_db)
    print(f"Processed and stored vector DB at {args.output_db}")
    result = pipeline.query(args.query, collection_name=args.output_db)
    print("\nANSWER:\n" + result['answer'])
    print("\nSOURCES:")
    for i, source in enumerate(result['sources']):
        print(f"\n{i+1}. {source['metadata'].get('document_title', 'Unknown document')}")
    
    # # Initialize the Medical RAG pipeline
    # pipeline = MedicalRAGPipeline(
    #     # Use PubMedBERT or BioBERT embeddings for medical content
    #     embedding_model="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    #     # Use a medically trained LLM when possible
    #     llm_model="/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct",  # Not medical-specific but generally competent
    # )
    
    # # Ingest a directory of medical PDFs
    # collection_name = pipeline.ingest_and_store("/home/vidhij2/nivi/documents")
    
    # # Query the system
    # result = pipeline.query(
    #     "What are the treatment options for Type 2 Diabetes?", 
    #     collection_name=collection_name
    # )
    
    # print("\nQuery:", result["query"])
    # print("\nAnswer:", result["answer"])
    # print("\nRelevant evidence:")
    # for i, chunk in enumerate(result["chunks"][:3]):  # Show top 3 chunks
    #     print(f"\nSource {i+1}: {chunk.metadata.get('source', 'Unknown')}")
    #     print(f"Section: {chunk.metadata.get('section', 'Unknown')}")
    #     print(chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content)