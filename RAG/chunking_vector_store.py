"""
chunking_vector_store.py
-----------------------
Provides chunking and vector store creation for medical documents, including custom header splitting, evidence extraction, and table extraction.
This module is used for processing documents into chunks and storing them in a vector database for retrieval.

Classes:
    - MedicalHeaderTextSplitter: Custom text splitter for medical documents that respects section headers.
    - MedicalEvidenceExtractor: Extracts evidence levels and recommendation types from medical text.
    - TableExtractor: Extracts tables as separate documents with metadata.
    - MedicalDocumentProcessor: Processes medical documents into semantically meaningful chunks and creates vector stores.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import argparse
from tqdm import tqdm

# LangChain imports
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.docstore.document import Document
from langchain.schema import BaseDocumentTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

    """Custom text splitter for medical documents that respects section headers."""
    
    def __init__(self):
        headers_to_split_on = [
            ("#", "chapter"),
            ("##", "section"),
            ("###", "subsection"),
            ("####", "recommendation"),
            ("#####", "remarks")
        ]
        super().__init__(headers_to_split_on=headers_to_split_on)
        
    def _add_md_header(self, text):
        """Convert medical document headers to markdown format."""
        # Convert chapter headers
        text = re.sub(r'(?m)^(?:Chapter|CHAPTER)\s+(\d+)[\.:]?\s+(.+)$', r'# \1 \2', text)
        
        # Convert section headers
        text = re.sub(r'(?m)^(?:\d+\.\d+\.?\s+|\d+\.\s+)([A-Z][A-Za-z\s\-:]+)$', r'## \1', text)
        
        # Convert subsection headers
        text = re.sub(r'(?m)^(?:[A-Z]\.\d+\.?\s+|[A-Z]\.\s+)([A-Za-z][A-Za-z\s\-:]+)$', r'### \1', text)
        
        # Convert recommendation headers
        text = re.sub(
            r'(?m)^RECOMMENDATION\s+([A-Z0-9\.]+):\s+(.+?)(?:\((?:Recommended|Context-specific|Not recommended).*?\))?$', 
            r'#### RECOMMENDATION \1: \2', 
            text
        )
        
        # Convert remarks sections
        text = re.sub(r'(?m)^Remarks:$', r'##### Remarks:', text)
        
        return text

class MedicalEvidenceExtractor(BaseDocumentTransformer):
    """Extract evidence levels and recommendation types from medical text."""
    
    def __init__(self):
        self.evidence_pattern = re.compile(r'(?:high|moderate|low|very\s+low)(?:-|\s+)(?:quality|certainty)\s+evidence', re.IGNORECASE)
        self.recommendation_type_pattern = re.compile(r'\((Recommended|Context-specific recommendation|Not recommended).*?\)')
    
    def transform_documents(
        self, documents: List[Document], **kwargs
    ) -> List[Document]:
        """Extract evidence levels and enhance document metadata."""
        for doc in documents:
            # Only process if it's a recommendation
            if doc.metadata.get('heading_type') == 'recommendation':
                # Extract evidence level
                evidence_match = self.evidence_pattern.search(doc.page_content)
                if evidence_match:
                    doc.metadata['evidence_level'] = evidence_match.group(0)
                
                # Extract recommendation type
                rec_type_match = self.recommendation_type_pattern.search(doc.page_content)
                if rec_type_match:
                    doc.metadata['recommendation_type'] = rec_type_match.group(1)
                
                # Extract recommendation ID
                rec_id_match = re.search(r'RECOMMENDATION\s+([A-Z0-9\.]+):', doc.page_content)
                if rec_id_match:
                    doc.metadata['recommendation_id'] = rec_id_match.group(1)
        
        return documents

class TableExtractor(BaseDocumentTransformer):
    """Extract tables as separate documents with metadata."""
    
    def transform_documents(
        self, documents: List[Document], **kwargs
    ) -> List[Document]:
        """Identify and mark table content."""
        table_pattern = re.compile(r'(Table\s+\d+[\.:]?\s+.*?)(?:\n\n|\Z)', re.DOTALL)
        
        result_docs = []
        for doc in documents:
            # Find tables in the document
            tables = table_pattern.findall(doc.page_content)
            
            # If tables found, create separate documents for them
            if tables:
                # Create a copy of the original document with tables removed
                modified_content = doc.page_content
                for table in tables:
                    modified_content = modified_content.replace(table, "")
                
                # Add the modified document if it still has significant content
                if len(modified_content.strip()) > 100:
                    modified_doc = Document(
                        page_content=modified_content,
                        metadata=doc.metadata.copy()
                    )
                    result_docs.append(modified_doc)
                
                # Add each table as a separate document
                for table in tables:
                    if len(table.strip()) > 50:  # Skip very small tables
                        table_doc = Document(
                            page_content=table,
                            metadata={
                                **doc.metadata.copy(),
                                "chunk_type": "table",
                                "parent_section_path": doc.metadata.get("section_path", [])
                            }
                        )
                        result_docs.append(table_doc)
            else:
                # No tables, keep the original document
                result_docs.append(doc)
                
        return result_docs

class MedicalDocumentProcessor:
    """Process medical documents into semantically meaningful chunks."""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        embedding_model_name: str = "text-embedding-ada-002"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        
        # Initialize document processing pipeline
        self.header_splitter = MedicalHeaderTextSplitter()
        self.paragraph_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.evidence_extractor = MedicalEvidenceExtractor()
        self.table_extractor = TableExtractor()
        
        # Initialize embedding model
        self.embeddings = OpenAIEmbeddings(model=embedding_model_name)
    
    def _extract_document_metadata(self, text: str) -> Dict[str, str]:
        """Extract document-level metadata."""
        title = ""
        doc_type = ""
        
        # Extract title from WHO document
        title_match = re.search(r'(?:WHO|World Health Organization)\s+(?:recommendations|guidelines)\s+(?:on|for)\s+([A-Za-z\s\-,]+)', text[:3000])
        if title_match:
            title = title_match.group(1).strip()
            doc_type = "WHO Guidelines"
            
        # If no specific match, try to extract a general title
        if not title:
            title_match = re.search(r'^([A-Z][A-Za-z\s\-:,]+(?:Guidelines|Recommendations|Guidance))', text[:1000])
            if title_match:
                title = title_match.group(1).strip()
                doc_type = "Medical Guidelines"
        
        return {
            "title": title,
            "document_type": doc_type
        }
    
    def _build_section_path(self, doc: Document) -> List[str]:
        """Build the hierarchical section path for a document."""
        path = []
        
        # Add chapter if available
        if 'chapter' in doc.metadata:
            path.append(doc.metadata['chapter'])
            
        # Add section if available
        if 'section' in doc.metadata:
            path.append(doc.metadata['section'])
            
        # Add subsection if available
        if 'subsection' in doc.metadata:
            path.append(doc.metadata['subsection'])
        
        return path
    
    def process_text(self, text: str, source_name: str = "") -> List[Document]:
        """Process text into hierarchical chunks."""
        # Extract document metadata
        doc_metadata = self._extract_document_metadata(text)
        
        # Add source information to metadata
        if source_name:
            doc_metadata["source"] = source_name
        
        # Convert headers to markdown format for the splitter
        md_text = self.header_splitter._add_md_header(text)
        
        # Split on headers
        docs = self.header_splitter.split_text(md_text)
        
        # Extract evidence levels and recommendation metadata
        docs = self.evidence_extractor.transform_documents(docs)
        
        # Extract tables
        docs = self.table_extractor.transform_documents(docs)
        
        # Build section paths for each document
        for doc in docs:
            section_path = self._build_section_path(doc)
            doc.metadata['section_path'] = section_path
            
            # Add document metadata
            doc.metadata['document_title'] = doc_metadata.get('title', '')
            doc.metadata['document_type'] = doc_metadata.get('document_type', '')
            
            # Determine chunk type if not already set
            if 'chunk_type' not in doc.metadata:
                if doc.metadata.get('heading_type') == 'recommendation':
                    doc.metadata['chunk_type'] = 'recommendation'
                elif doc.metadata.get('heading_type') == 'remarks':
                    doc.metadata['chunk_type'] = 'remarks'
                else:
                    doc.metadata['chunk_type'] = 'text'
        
        # Further split large chunks while preserving metadata
        final_docs = []
        for doc in docs:
            # Don't split recommendation or remarks sections
            if doc.metadata.get('chunk_type') in ['recommendation', 'remarks', 'table']:
                final_docs.append(doc)
            else:
                # Split text sections into smaller chunks
                if len(doc.page_content) > self.chunk_size:
                    smaller_chunks = self.paragraph_splitter.split_text(doc.page_content)
                    for i, chunk in enumerate(smaller_chunks):
                        chunk_doc = Document(
                            page_content=chunk,
                            metadata={
                                **doc.metadata,
                                'chunk_index': i,
                                'total_chunks': len(smaller_chunks)
                            }
                        )
                        final_docs.append(chunk_doc)
                else:
                    final_docs.append(doc)
        
        return final_docs
    
    def load_documents(self, input_path: str) -> List[Document]:
        """Load documents from file or directory."""
        documents = []
        
        if os.path.isdir(input_path):
            # Process all files in directory
            for filename in os.listdir(input_path):
                file_path = os.path.join(input_path, filename)
                if os.path.isfile(file_path):
                    documents.extend(self._load_single_document(file_path))
        else:
            # Process single file
            documents = self._load_single_document(input_path)
        
        return documents
    
    def _load_single_document(self, file_path: str) -> List[Document]:
        """Load and process a single document."""
        print(f"Processing {file_path}...")
        
        # Load the document
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text = "\n\n".join([page.page_content for page in pages])
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Process the text
        source_name = os.path.basename(file_path)
        return self.process_text(text, source_name)
    
    def create_vector_store(self, documents: List[Document], persist_directory: str = None) -> Chroma:
        """Create a vector store from processed documents."""
        # Create Chroma DB with metadata
        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}  # Optimize for medical text similarity
        )
        
        if persist_directory:
            db.persist()
            print(f"Vector database persisted to {persist_directory}")
        
        return db
    
    def load_vector_store(self, persist_directory: str) -> Chroma:
        """Load an existing vector store."""
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

