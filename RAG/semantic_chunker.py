"""
semantic_chunker.py
------------------
Provides advanced semantic chunking for maternal health documents using SpaCy and regex patterns.
This chunker is used for splitting documents into semantically meaningful sections, preserving important blocks (lists, tables, recommendations, etc.).

Classes:
    - SemanticMaternalHealthChunker: Creates semantically meaningful chunks for maternal health documents.
"""

from typing import List, Dict, Any, Optional, Tuple
import re

class SemanticMaternalHealthChunker:
    """Creates semantically meaningful chunks for maternal health documents."""
    
    def __init__(self, spacy_model="en_core_web_md"):
        import spacy
        # Load SpaCy for text processing
        self.nlp = spacy.load(spacy_model)
        
        # Semantic boundary patterns
        self.section_patterns = [
            r'^(?:[0-9]+\.){1,3}\s+[A-Z]',           # Numbered sections (1.2.3 Title)
            r'^[A-Z][A-Za-z\s]+:',                    # Title with colon (Pregnancy Symptoms:)
            r'^[A-Z][A-Za-z\s]+ during [A-Za-z\s]+',  # "X during Y" patterns
            r'^(?:Table|Figure|Box)\s+[0-9]+:',       # Tables and figures
            r'^WARNING:',                              # Warning sections
            r'^NOTE:'                                  # Note sections
        ]
        self.section_regex = re.compile('|'.join(f'({pattern})' for pattern in self.section_patterns), 
                                       re.MULTILINE)
        
        # Define semantic units we want to preserve
        self.preservation_patterns = [
            # Lists
            r'(?:^[•\-\*]\s+.+\n){2,}',  
            # Tables with multiple columns
            r'(?:\|.+\|.+\|.+\n){2,}',
            # Structured recommendations
            r'Recommendation:[\s\S]+?(?:Evidence quality|Strength of recommendation)',
            # Numbered steps/procedures
            r'(?:^[0-9]+\.\s+.+\n){2,}',
            # Warning blocks
            r'(?:Warning|Caution|Alert):[\s\S]+?(?:\n\n|\Z)'
        ]
        self.preservation_regex = re.compile('|'.join(f'({pattern})' for pattern in self.preservation_patterns), 
                                            re.MULTILINE)
        
        # Topic keywords for maternal health
        self.topic_keywords = {
            'fetal_development': ['development', 'growth', 'fetal', 'foetal', 'baby', 'fetus'],
            'maternal_changes': ['mother', 'maternal', 'body', 'changes', 'symptoms'],
            'nutrition': ['diet', 'food', 'nutrition', 'eat', 'meal', 'vitamin'],
            'warning_signs': ['danger', 'warning', 'risk', 'emergency', 'complication'],
            'normal_variations': ['normal', 'common', 'usual', 'typical', 'variant', 'variation'],
            'medical_tests': ['test', 'scan', 'ultrasound', 'screening', 'monitor'],
            'delivery': ['birth', 'labor', 'labour', 'delivery', 'contractions'],
            'postpartum': ['after birth', 'post-delivery', 'postpartum', 'post-natal', 'breastfeeding']
        }
    
    def split_text_into_sections(self, text: str) -> List[str]:
        """Split text into semantically meaningful sections."""
        # Find all semantic boundaries
        section_boundaries = [match.start() for match in self.section_regex.finditer(text)]
        
        if not section_boundaries:
            # If no sections detected, return the whole text
            return [text]
        
        # Add the start and end of text to boundaries
        section_boundaries = [0] + section_boundaries + [len(text)]
        
        # Create sections based on boundaries
        sections = []
        for i in range(len(section_boundaries) - 1):
            section_text = text[section_boundaries[i]:section_boundaries[i+1]].strip()
            if section_text:
                sections.append(section_text)
        
        return sections
    
    def find_preservation_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """Find blocks of text that should be preserved."""
        preserved_blocks = []
        
        for match in self.preservation_regex.finditer(text):
            preserved_blocks.append((match.start(), match.end(), match.group(0)))
        
        return preserved_blocks
    
    def segment_by_semantic_boundaries(self, section: str, max_chunk_size: int = 1000) -> List[str]:
        """Split a section into semantic chunks respecting preserved blocks."""
        # Find blocks that should be preserved
        preserved_blocks = self.find_preservation_blocks(section)
        
        # If section is short enough, return it as-is
        if len(section) <= max_chunk_size and not preserved_blocks:
            return [section]
        
        # Process text with SpaCy for sentence boundaries
        doc = self.nlp(section)
        sentences = list(doc.sents)
        
        # Create chunks respecting sentence boundaries and preserved blocks
        chunks = []
        current_chunk = []
        current_length = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            # Check if this sentence is within a preserved block
            in_preserved_block = False
            for start, end, block_text in preserved_blocks:
                if start <= sentence.start_char < end:
                    # This sentence is part of a preserved block
                    # Add the entire block as a chunk
                    chunks.append(block_text)
                    # Skip all sentences in this block
                    while i < len(sentences) and sentences[i].start_char < end:
                        i += 1
                    in_preserved_block = True
                    break
            
            if in_preserved_block:
                # Continue to next sentence
                continue
            
            # If not in a preserved block, process normally
            sentence_text = sentence.text.strip()
            
            # If adding this sentence would exceed max size, create a new chunk
            if current_length + len(sentence_text) > max_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence_text)
            current_length += len(sentence_text)
            i += 1
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def identify_topics(self, text: str) -> List[str]:
        """Identify maternal health topics in the text."""
        text_lower = text.lower()
        identified_topics = []
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                identified_topics.append(topic)
        
        return identified_topics
    
    def extract_pregnancy_stage(self, text: str) -> Dict[str, List]:
        """Extract pregnancy stages mentioned in the text."""
        # Week pattern (e.g., "week 12", "12th week", "weeks 12-14")
        week_pattern = re.compile(r'(?:week|weeks)\s+(\d+(?:-\d+)?)|(\d+)(?:st|nd|rd|th)?\s+week', re.IGNORECASE)
        
        # Month pattern (e.g., "month 3", "3rd month", "months 4-6")
        month_pattern = re.compile(r'(?:month|months)\s+(\d+(?:-\d+)?)|(\d+)(?:st|nd|rd|th)?\s+month', re.IGNORECASE)
        
        # Trimester pattern
        trimester_pattern = re.compile(r'(first|second|third)\s+trimester', re.IGNORECASE)
        
        stages = {
            "weeks": [],
            "months": [],
            "trimesters": []
        }
        
        # Extract weeks
        for match in week_pattern.finditer(text):
            week_str = match.group(1) or match.group(2)
            if '-' in week_str:
                start, end = map(int, week_str.split('-'))
                stages["weeks"].extend(list(range(start, end + 1)))
            else:
                stages["weeks"].append(int(week_str))
        
        # Extract months
        for match in month_pattern.finditer(text):
            month_str = match.group(1) or match.group(2)
            if '-' in month_str:
                start, end = map(int, month_str.split('-'))
                stages["months"].extend(list(range(start, end + 1)))
            else:
                stages["months"].append(int(month_str))
        
        # Extract trimesters
        for match in trimester_pattern.finditer(text):
            trimester = match.group(1).lower()
            stages["trimesters"].append(trimester)
        
        # Remove duplicates
        for key in stages:
            stages[key] = list(set(stages[key]))
        
        return stages
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text using SpaCy."""
        doc = self.nlp(text)
        
        entities = {
            "conditions": [],
            "medications": [],
            "procedures": [],
            "measurements": []
        }
        
        # Simple matching for common maternal health entities
        condition_keywords = ["preeclampsia", "eclampsia", "gestational diabetes", 
                              "anemia", "anaemia", "hypertension", "nausea", "vomiting"]
        
        medication_keywords = ["iron", "folate", "folic acid", "vitamin", "supplement"]
        
        procedure_keywords = ["ultrasound", "scan", "test", "screening", "amniocentesis", 
                             "delivery", "c-section", "cesarean", "caesarean"]
        
        measurement_keywords = ["weight", "blood pressure", "fundal height", "bp", "heart rate"]
        
        # Extract entities using simple keyword matching
        text_lower = text.lower()
        
        for condition in condition_keywords:
            if condition in text_lower:
                entities["conditions"].append(condition)
        
        for medication in medication_keywords:
            if medication in text_lower:
                entities["medications"].append(medication)
        
        for procedure in procedure_keywords:
            if procedure in text_lower:
                entities["procedures"].append(procedure)
        
        for measurement in measurement_keywords:
            if measurement in text_lower:
                entities["measurements"].append(measurement)
        
        # Extract named entities from SpaCy
        for ent in doc.ents:
            if ent.label_ == "CONDITION" and ent.text.lower() not in entities["conditions"]:
                entities["conditions"].append(ent.text.lower())
            elif ent.label_ == "MEDICATION" and ent.text.lower() not in entities["medications"]:
                entities["medications"].append(ent.text.lower())
            elif ent.label_ == "PROCEDURE" and ent.text.lower() not in entities["procedures"]:
                entities["procedures"].append(ent.text.lower())
        
        return entities
    
    def create_semantic_chunks(self, text: str, metadata: Dict = None) -> List[Document]:
        """Create semantically meaningful chunks from text."""
        if metadata is None:
            metadata = {}
        
        # Split text into sections
        sections = self.split_text_into_sections(text)
        
        # Process each section into chunks
        documents = []
        
        for section in sections:
            # Extract section title
            section_title = ""
            first_line = section.split('\n')[0].strip()
            if re.match(r'^[0-9\.]+\s+|^[A-Z][A-Za-z\s]+:', first_line):
                section_title = first_line
            
            # Create semantic chunks from section
            chunks = self.segment_by_semantic_boundaries(section)
            
            for i, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue
                
                # Extract semantic metadata
                topics = self.identify_topics(chunk)
                pregnancy_stages = self.extract_pregnancy_stage(chunk)
                medical_entities = self.extract_medical_entities(chunk)
                
                # Determine chunk type
                chunk_type = "general_text"
                if any(marker in chunk.lower() for marker in ["warning", "caution", "danger"]):
                    chunk_type = "warning"
                elif any(marker in chunk.lower() for marker in ["recommendation", "advised", "should"]):
                    chunk_type = "recommendation"
                elif re.search(r'(?:\|.+\|.+\|.+\n){2,}', chunk) or re.search(r'Table\s+[0-9]+:', chunk):
                    chunk_type = "table"
                elif re.search(r'(?:^[•\-\*]\s+.+\n){2,}', chunk):
                    chunk_type = "list"
                
                # Create document with metadata
                chunk_metadata = {
                    **metadata,
                    "section_title": section_title,
                    "chunk_index": i,
                    "total_chunks_in_section": len(chunks),
                    "chunk_type": chunk_type,
                    "topics": topics,
                    "pregnancy_stages": pregnancy_stages,
                    "medical_entities": medical_entities,
                    "contains_warning": "warning" in chunk_type or any(warning in chunk.lower() 
                                                                     for warning in ["warning", "caution", "danger"]),
                    "contains_recommendation": "recommendation" in chunk_type or any(rec in chunk.lower() 
                                                                                   for rec in ["recommend", "advised", "should"])
                }
                
                # Create document
                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                
                documents.append(doc)
        
        return documents