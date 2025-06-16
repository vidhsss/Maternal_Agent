"""
heirarchical_markdown_chunking.py
--------------------------------
Provides hierarchical markdown chunking for medical documents, including custom header splitting and evidence extraction.
This module is used for splitting documents into hierarchical sections and extracting structured metadata.

Classes:
    - MedicalHeaderTextSplitter: Custom text splitter for medical documents that respects section headers.
    - MedicalEvidenceExtractor: Extracts evidence levels and recommendation types from medical text.
    - TableExtractor: Extracts tables as separate documents with metadata.
    - RecursiveHierarchicalSplitter: Splits documents hierarchically, preserving parent-child relationships.
"""

import re
import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# LangChain imports
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain.docstore.document import Document
from langchain.schema import BaseDocumentTransformer

class MedicalHeaderTextSplitter(MarkdownHeaderTextSplitter):
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

class RecursiveHierarchicalSplitter:
    """Split documents hierarchically, preserving parent-child relationships."""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        include_metadata: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_metadata = include_metadata
        
        # Initialize splitters
        self.header_splitter = MedicalHeaderTextSplitter()
        self.paragraph_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.evidence_extractor = MedicalEvidenceExtractor()
        self.table_extractor = TableExtractor()
    
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
    
    def _extract_document_metadata(self, text: str) -> Dict[str, str]:
        """Extract document-level metadata."""
        title = ""
        doc_type = ""
        
        # Extract title from WHO document
        title_match = re.search(r'(?:WHO|World Health Organization)\s+(?:recommendations|guidelines)\s+(?:on|for)\s+([A-Za-z\s\-,]+)', text[:3000])
        if title_match:
            title = title_match.group(1).strip()
            doc_type = "WHO Guidelines"
        
        return {
            "title": title,
            "document_type": doc_type
        }
    
    def process_text(self, text: str) -> List[Document]:
        """Process text into hierarchical chunks."""
        # Extract document metadata
        doc_metadata = self._extract_document_metadata(text)
        
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
        
        # Build relationships between chunks
        chunk_dict = {i: doc for i, doc in enumerate(final_docs)}
        for i, doc in chunk_dict.items():
            # Find parent-child relationships
            if doc.metadata.get('heading_type') == 'remarks':
                # Find the recommendation this belongs to
                for j, other_doc in chunk_dict.items():
                    if (other_doc.metadata.get('heading_type') == 'recommendation' and
                        other_doc.metadata.get('recommendation_id') == doc.metadata.get('recommendation_id')):
                        doc.metadata['parent_id'] = j
                        break
        
        return final_docs
    
    def process_file(self, input_file: str, output_file: str = None) -> List[Document]:
        """Process a file and return LangChain documents."""
        # Load the document
        if input_file.lower().endswith('.pdf'):
            loader = PyPDFLoader(input_file)
            pages = loader.load()
            text = "\n\n".join([page.page_content for page in pages])
        else:
            loader = TextLoader(input_file)
            documents = loader.load()
            text = documents[0].page_content
        
        # Process the text
        docs = self.process_text(text)
        
        # Save to output file if requested
        if output_file:
            # Convert to serializable format
            serializable_docs = []
            for i, doc in enumerate(docs):
                serializable_docs.append({
                    "chunk_id": i,
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })
                
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(docs)} chunks to {output_file}")
        
        return docs



    
    # parser = argparse.ArgumentParser(description="Process medical documents with LangChain")
    # parser.add_argument("input_file", help="Path to input file (PDF or text)")
    # parser.add_argument("--output_file", help="Path to output JSON file", default=None)
    # parser.add_argument("--chunk_size", type=int, default=1000, help="Maximum chunk size in characters")
    # parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between chunks in characters")
    
    # args = parser.parse_args()
    
    # Default output file if not specified
    # if not args.output_file:
    #     base_name = os.path.splitext(args.input_file)[0]
    #     args.output_file = f"{base_name}_langchain_chunks.json"
    
    # splitter = RecursiveHierarchicalSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200
    # )

    # docs = splitter.process_file("9789240020306-eng.pdf", "chunks.json")
    # print(f"Generated {len(docs)} chunks")

# if __name__ == "__main__":
#     main()