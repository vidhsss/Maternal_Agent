"""
MaternalHealthDocumentProcessor.py
---------------------------------
Provides a specialized processor for maternal health documents, extending a base medical document processor and using a semantic chunker for maternal health.

Classes:
    - MaternalHealthDocumentProcessor: Enhanced processor specifically for maternal health documents.
"""

import re
from typing import List, Dict, Any, Optional
from langchain.schema import Document
# Import the base processor and semantic chunker from local modules
from .chunking_vector_store import MedicalDocumentProcessor
from .semantic_chunker import SemanticMaternalHealthChunker

class MaternalHealthDocumentProcessor(MedicalDocumentProcessor):
    """Enhanced processor specifically for maternal health documents."""
    
    def __init__(
        self,
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        embedding_model_name: str = "text-embedding-ada-002",
        db_directory: str = "maternal_health_db"
    ):
        super().__init__(chunk_size, chunk_overlap, embedding_model_name, db_directory)
        self.semantic_chunker = SemanticMaternalHealthChunker()
        # Pregnancy stage detection
        self.stage_patterns = {
            'weeks': re.compile(r'(?:week|weeks)\s+(\d+(?:-\d+)?)', re.IGNORECASE),
            'months': re.compile(r'(?:month|months)\s+(\d+(?:-\d+)?)', re.IGNORECASE),
            'trimesters': re.compile(r'(first|second|third)\s+trimester', re.IGNORECASE)
        }
        
        # Key maternal health topics
        self.topic_keywords = {
            'fetal_development': ['development', 'growth', 'fetal', 'foetal', 'baby', 'fetus'],
            'maternal_changes': ['mother', 'maternal', 'body', 'changes', 'symptoms'],
            'nutrition': ['diet', 'food', 'nutrition', 'eat', 'meal', 'vitamin'],
            'warning_signs': ['danger', 'warning', 'risk', 'emergency', 'complication'],
            'normal_variations': ['normal', 'common', 'usual', 'typical', 'variant', 'variation']
        }
        
    def _extract_pregnancy_stage(self, text: str) -> Dict[str, Any]:
        """Extract pregnancy stage information from text."""
        stages = {
            'weeks': [],
            'months': [],
            'trimesters': []
        }
        
        # Extract weeks
        for match in self.stage_patterns['weeks'].finditer(text.lower()):
            week_range = match.group(1)
            if '-' in week_range:
                start, end = map(int, week_range.split('-'))
                stages['weeks'].extend(list(range(start, end + 1)))
            else:
                try:
                    stages['weeks'].append(int(week_range))
                except ValueError:
                    pass
        
        # Extract months
        for match in self.stage_patterns['months'].finditer(text.lower()):
            month_range = match.group(1)
            if '-' in month_range:
                start, end = map(int, month_range.split('-'))
                stages['months'].extend(list(range(start, end + 1)))
            else:
                try:
                    stages['months'].append(int(month_range))
                except ValueError:
                    pass
        
        # Extract trimesters
        for match in self.stage_patterns['trimesters'].finditer(text.lower()):
            trimester = match.group(1).lower()
            if trimester not in stages['trimesters']:
                stages['trimesters'].append(trimester)
        
        return stages
    
    def _identify_topics(self, text: str) -> List[str]:
        """Identify maternal health topics in the text."""
        text_lower = text.lower()
        identified_topics = []
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                identified_topics.append(topic)
        
        return identified_topics
    
    def process_text(self, text: str, source_name: str = "") -> List[Document]:
        """Enhanced processing for maternal health documents."""
        # Process using the parent method first
        # docs = super().process_text(text, source_name)
        
        # # Enhance with maternal health specific metadata
        # for doc in docs:
        #     # Extract pregnancy stage information
        #     stages = self._extract_pregnancy_stage(doc.page_content)
        #     doc.metadata['pregnancy_stages'] = stages
            
        #     # Identify relevant topics
        #     topics = self._identify_topics(doc.page_content)
        #     doc.metadata['maternal_topics'] = topics
            
        #     # Flag content with medical advice or warnings
        #     if any(warning_term in doc.page_content.lower() for warning_term in 
        #            ['warning', 'caution', 'danger', 'emergency', 'seek', 'consult']):
        #         doc.metadata['contains_warnings'] = True
            
        #     # Flag normal variations content
        #     if any(normal_term in doc.page_content.lower() for normal_term in 
        #            ['normal', 'common', 'typically', 'usually', 'generally']):
        #         doc.metadata['discusses_normal_variations'] = True
        doc_metadata = self._extract_document_metadata(text)
        
        # Add source information to metadata
        if source_name:
            doc_metadata["source"] = source_name
        
        # Convert headers to markdown format for the splitter
        # md_text = self.header_splitter._add_md_header(text)
        
        # Split on headers
        # docs = self.header_splitter.split_text(md_text)
        docs = self.semantic_chunker.create_semantic_chunks(text, doc_metadata)
        
        # Extract evidence levels and recommendation metadata
        docs = self.evidence_extractor.transform_documents(docs)
        
        # Extract tables
        docs = self.table_extractor.transform_documents(docs)
        
        
        return docs
        
    def create_specialized_chunks(self, docs: List[Document]) -> List[Document]:
        """Create specialized maternal health chunks for improved retrieval."""
        specialized_chunks = []
        
        # Group documents by pregnancy stage
        stage_groups = {}
        
        for doc in docs:
            # Process week-specific chunks
            for week in doc.metadata.get('pregnancy_stages', {}).get('weeks', []):
                key = f"week_{week}"
                if key not in stage_groups:
                    stage_groups[key] = []
                stage_groups[key].append(doc)
            
            # Process month-specific chunks
            for month in doc.metadata.get('pregnancy_stages', {}).get('months', []):
                key = f"month_{month}"
                if key not in stage_groups:
                    stage_groups[key] = []
                stage_groups[key].append(doc)
            
            # Process trimester-specific chunks
            for trimester in doc.metadata.get('pregnancy_stages', {}).get('trimesters', []):
                key = f"trimester_{trimester}"
                if key not in stage_groups:
                    stage_groups[key] = []
                stage_groups[key].append(doc)
        
        # Create specialized chunks for each stage group
        for stage_key, stage_docs in stage_groups.items():
            # Group by topics within stage
            topic_contents = {}
            
            for doc in stage_docs:
                for topic in doc.metadata.get('maternal_topics', []):
                    if topic not in topic_contents:
                        topic_contents[topic] = []
                    topic_contents[topic].append(doc.page_content)
            
            # Create specialized chunks for each topic in this stage
            for topic, contents in topic_contents.items():
                if len(contents) >= 2:  # Only create specialized chunks if we have multiple sources
                    specialized_chunk = Document(
                        page_content="\n\n".join(contents[:3]),  # Limit to avoid too-large chunks
                        metadata={
                            'chunk_type': 'specialized',
                            'stage_key': stage_key,
                            'topic': topic,
                            'source_count': len(contents),
                            'is_synthesized': True
                        }
                    )
                    specialized_chunks.append(specialized_chunk)
        
        return specialized_chunks
    
    def create_vector_store(self, documents: List[Document], collection_name: Optional[str] = None):
        """Create enhanced vector store with specialized chunks."""
        # Create specialized chunks
        specialized_chunks = self.create_specialized_chunks(documents)
        
        # Add specialized chunks to the document set
        all_docs = documents + specialized_chunks
        
        # Create the vector store with all documents
        return super().create_vector_store(all_docs, collection_name)


processor = MaternalHealthDocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process maternal health documents and build a vector store.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input text file')
    parser.add_argument('--output_db', type=str, required=True, help='Directory to save the vector store')
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()
    docs = processor.process_text(text, source_name=args.input_file)
    processor.create_vector_store(docs, collection_name=args.output_db)
    print(f"Processed and stored vector DB at {args.output_db}")