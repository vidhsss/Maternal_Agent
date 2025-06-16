"""
complete_RAG_FAISS.py
---------------------
Provides a complete Retrieval-Augmented Generation (RAG) pipeline with document loading, chunking, embedding, FAISS vector store management, and querying capabilities.

Classes:
    - RAGPipeline: Complete RAG pipeline for document QA with HuggingFace and FAISS.

Usage:
    Can be imported as a module or run as a script for end-to-end RAG workflows.
"""

import os
import uuid
import torch
from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGPipeline:
    """
    A complete Retrieval-Augmented Generation pipeline with document loading, chunking,
    embedding, database storage, and querying capabilities.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        llm_model: str = "google/flan-t5-base",
        db_directory: str = "db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_kwargs: dict = {"device": "cpu"},
        encode_kwargs: dict = {"normalize_embeddings": True},
        use_gpu: bool = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: The Hugging Face model to use for embeddings (can be Llama or any other compatible model)
            llm_model: The Hugging Face model to use for generation
            db_directory: Directory to store the vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            model_kwargs: Additional keyword arguments for the embedding model
            encode_kwargs: Additional keyword arguments for the encoding process
            use_gpu: Whether to use GPU for LLM. If None, will auto-detect GPU.
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
        
        # Initialize embeddings with Hugging Face (compatible with Llama models)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Initialize local LLM using Hugging Face models
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        model = AutoModelForCausalLM.from_pretrained(llm_model
)
        
        # Create text generation pipeline with LLM
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create LangChain wrapper around the pipeline
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Initialize vectorstore if it exists
        if os.path.exists(os.path.join(db_directory, "index.faiss")):
            self.db = FAISS.load_local(
                folder_path=db_directory,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.db = None

    def load_documents(self, file_path: str) -> List[Document]:
        """
        Load documents from a file or directory.
        
        Args:
            file_path: Path to file or directory
            
        Returns:
            List of loaded documents
        """
        if os.path.isdir(file_path):
            # Load from directory
            loader = DirectoryLoader(
                file_path,
                glob="**/*.*",
                loader_cls=self._get_loader_for_extension
            )
            documents = loader.load()
        else:
            # Load single file
            extension = os.path.splitext(file_path)[1].lower()
            loader_cls = self._get_loader_for_extension(extension)
            loader = loader_cls(file_path)
            documents = loader.load()
            
        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents

    def _get_loader_for_extension(self, extension: str):
        """
        Get the appropriate document loader for a file extension.
        
        Args:
            extension: File extension
            
        Returns:
            Document loader class
        """
        if extension == '.pdf':
            return PyPDFLoader
        else:
            # Default to text loader
            return TextLoader

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of document chunks
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks

    def store_documents(self, chunks: List[Document], collection_name: Optional[str] = None):
        """
        Store document chunks in the vector database.
        
        Args:
            chunks: Document chunks to store
            collection_name: Optional name for the collection
        """
        # Generate a collection name if not provided
        if collection_name is None:
            collection_name = f"collection_{uuid.uuid4().hex[:8]}"
            
        # Initialize FAISS index from documents
        self.db = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Create directory if it doesn't exist
        os.makedirs(self.db_directory, exist_ok=True)
        
        # Save the index to disk with collection name in the path
        index_path = os.path.join(self.db_directory, collection_name)
        self.db.save_local(index_path)
        
        print(f"Stored {len(chunks)} chunks in collection '{collection_name}'")
        
        return collection_name

    def retrieve_chunks(
        self, 
        query: str, 
        n_results: int = 4,
        collection_name: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve relevant document chunks for a query.
        
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
            
        # Retrieve chunks
        chunks = db.similarity_search(query, k=n_results)
        return chunks

    def query(
        self, 
        query: str, 
        n_results: int = 4,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a query using the RAG pipeline.
        
        Args:
            query: The query string
            n_results: Number of chunks to retrieve
            collection_name: Optional collection to search in
            
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
            
        # Create a retriever
        retriever = db.as_retriever(search_kwargs={"k": n_results})
        
        # Create a prompt template
        template = """
        You are an assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question:
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

    def ingest_and_store(self, file_path: str, collection_name: Optional[str] = None) -> str:
        """
        Complete pipeline to ingest, process, and store documents.
        
        Args:
            file_path: Path to file or directory to ingest
            collection_name: Optional name for the collection
            
        Returns:
            Collection name
        """
        # Load documents
        documents = self.load_documents(file_path)
        
        # Process into chunks
        chunks = self.process_documents(documents)
        
        # Store in database
        collection_name = self.store_documents(chunks, collection_name)
        
        return collection_name


# Example usage
if __name__ == "__main__":
    # Initialize the RAG pipeline with local models
    pipeline = RAGPipeline(
        # Use smaller embedding model for faster processing
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        # Options for LLM:
        # - "google/flan-t5-base" (smaller, faster)
        # - "google/flan-t5-large" (better quality)
        # - "tiiuae/falcon-7b-instruct" (more capable but requires more GPU memory)
        # - "TheBloke/Llama-2-7B-Chat-GGML" (quantized Llama model, good for CPU)
        llm_model="/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    # Ingest and store documents
    
    collection_name = pipeline.ingest_and_store("/home/vidhij2/nivi/documents/9789240020306-eng.pdf")
    
    # Query the documents
    result = pipeline.query("What is the main topic of these documents?", collection_name=collection_name)
    
    print("\nQuery:", result["query"])
    print("\nAnswer:", result["answer"])
    print("\nRelevant chunks:")
    for i, chunk in enumerate(result["chunks"]):
        print(f"\nChunk {i+1}:")
        print(chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content)

    # llm_model = "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(llm_model)
    # model = AutoModelForCausalLM.from_pretrained(llm_model)
    