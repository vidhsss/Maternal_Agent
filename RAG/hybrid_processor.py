"""
hybrid_processor.py
------------------
Provides a hybrid semantic-hierarchical chunking pipeline for maternal health documents.
This module implements a comprehensive pipeline for processing and retrieving maternal health documents, optimized for patient queries.

Classes:
    - MaternalHealthPipeline: Comprehensive pipeline for processing and retrieving maternal health documents.
"""

import os
import re
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from functools import lru_cache
import logging
from tqdm import tqdm

# LangChain imports
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import BaseDocumentTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("maternal_health_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MaternalHealthPipeline:
    """
    Comprehensive pipeline for processing and retrieving maternal health documents.
    This implements a hybrid semantic-hierarchical chunking approach optimized for
    maternal health content and patient queries.
    """
    
    def __init__(
        self,
        chunk_size: int = 2500,
        chunk_overlap: int = 400,
        embedding_model_name: str = "google/muril-base-cased",  # MuRIL for multilingual support
        db_directory: str = "maternal_health_db",
        spacy_model: str = "en_core_web_md",
        use_spacy: bool = True
    ):
        """
        Initialize the maternal health document pipeline.
        
        Args:
            chunk_size: Target size for document chunks
            chunk_overlap: Amount of overlap between chunks
            embedding_model_name: HuggingFace model for embeddings
            db_directory: Directory to store vector database
            spacy_model: SpaCy model for NLP processing
            use_spacy: Whether to use SpaCy for enhanced NLP
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.db_directory = db_directory
        self.db = None
        
        # Create db directory if it doesn't exist
        os.makedirs(db_directory, exist_ok=True)
        
        # Initialize embedding model - use MuRIL for better multilingual support
        logger.info(f"Initializing embeddings model: {embedding_model_name}")
        self.embeddings = E5Embeddings(device="cuda")
        # Initialize SpaCy if requested
        self.use_spacy = use_spacy
        if use_spacy:
            try:
                import spacy
                try:
                    self.nlp = spacy.load(spacy_model)
                    logger.info(f"Loaded SpaCy model: {spacy_model}")
                except OSError:
                    logger.info(f"Downloading SpaCy model: {spacy_model}")
                    spacy.cli.download(spacy_model)
                    self.nlp = spacy.load(spacy_model)
            except ImportError:
                logger.warning("SpaCy not available. Some features will be limited.")
                self.use_spacy = False
                self.nlp = None
        else:
            self.nlp = None

        # Check for existing vector store
        index_path = os.path.join(db_directory, "index.faiss")
        if os.path.exists(index_path):
            logger.info(f"Loading existing vector store from {index_path}")
            self.db = FAISS.load_local(
                folder_path=db_directory,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.db = None
        
        # Compile regex patterns
        self._compile_regex_patterns()
        
        # Initialize domain-specific knowledge
        self._initialize_domain_knowledge()
    
    def _compile_regex_patterns(self):
        """Compile all regex patterns used across the pipeline for performance."""
        # Document structure patterns
        self.structure_patterns = {
            # Formal headers in guidelines
            'formal_header': re.compile(r'^(?:[0-9]+\.){1,3}\s+[A-Z]|^(?:CHAPTER|Section|SECTION)\s+[0-9IVX]+', re.MULTILINE),
            
            # Natural language headers
            'nl_header': re.compile(r'^[A-Z][A-Za-z\s]+:$|^[A-Z][A-Za-z\s]+ during [A-Za-z\s]+$', re.MULTILINE),
            
            # Special content markers
            'special_marker': re.compile(r'^(?:NOTE|WARNING|CAUTION|RECOMMENDATION|EVIDENCE|BOX|Table|Figure):?', re.MULTILINE),
            
            # Document boundaries
            'doc_boundary': re.compile(r'(?:\n\s*){3,}|\f|(?:=+\s*){3,}|(?:-+\s*){3,}')
        }
        
        # Content preservation patterns
        self.preservation_patterns = {
            # Lists
            'list': re.compile(r'(?:^[•\-\*√]\s+.+(?:\n|$)){2,}', re.MULTILINE),
            
            # Tables (various formats)
            'table': re.compile(r'(?:\|.+\|.+\|.+\n){2,}|(?:^[\w\s]+\t+[\w\s]+\t+[\w\s]+(?:\n|$)){2,}', re.MULTILINE),
            
            # Warning blocks
            'warning': re.compile(r'(?:Warning|Caution|Alert|WARNING|CAUTION|ALERT):[\s\S]+?(?:\n\n|\Z)', re.MULTILINE),
            
            # Recommendation blocks
            'recommendation': re.compile(r'(?:Recommendation|RECOMMENDATION):[\s\S]+?(?:\n\n|\Z)', re.MULTILINE),
            
            # Numbered steps/procedures
            'steps': re.compile(r'(?:^[0-9]+\.\s+.+(?:\n|$)){2,}', re.MULTILINE)
        }
        
        # Pregnancy stage patterns
        self.stage_patterns = {
            'week': re.compile(r'(?:week|weeks)\s+(\d+(?:-\d+)?)|(\d+)(?:st|nd|rd|th)?\s+week', re.IGNORECASE),
            'month': re.compile(r'(?:month|months)\s+(\d+(?:-\d+)?)|(\d+)(?:st|nd|rd|th)?\s+month', re.IGNORECASE),
            'trimester': re.compile(r'(first|second|third)\s+trimester', re.IGNORECASE),
            'stage': re.compile(r'(preconception|prenatal|antenatal|perinatal|intrapartum|postpartum|postnatal)', re.IGNORECASE)
        }
        
        # Clinical patterns
        self.clinical_patterns = {
            'measurement': re.compile(r'(\d+(?:\.\d+)?)\s*(cm|mm|kg|g|lb|oz|bpm|mmHg|%|mg|mcg|IU)', re.IGNORECASE),
            'dosage': re.compile(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|IU|units)\s+(\d+(?:-\d+)?)\s*times?(\/day|\/week|daily|weekly)', re.IGNORECASE)
        }
    
    def _initialize_domain_knowledge(self):
        """Initialize comprehensive domain knowledge for maternal health."""
        # Document categories
        self.doc_categories = {
            'clinical': ['guideline', 'protocol', 'recommendation', 'clinical', 'medical', 'treatment', 'diagnosis', 'procedure'],
            'educational': ['education', 'information', 'guide', 'leaflet', 'brochure', 'advice'],
            'nutritional': ['nutrition', 'diet', 'food', 'eating', 'meal', 'vitamin', 'supplement'],
            'behavioral': ['behavior', 'behaviour', 'habit', 'lifestyle', 'exercise', 'activity', 'practice'],
            'emotional': ['emotional', 'mental', 'psychological', 'feeling', 'support', 'stress', 'anxiety', 'depression']
        }
        
        # Pregnancy stages (for improved stage detection)
        self.pregnancy_stages = {
            'preconception': ['planning', 'before pregnancy', 'preconception', 'pre-conception', 'fertility'],
            'first_trimester': ['first trimester', 'early pregnancy', 'weeks 1-12', '1st trimester'],
            'second_trimester': ['second trimester', 'mid pregnancy', 'weeks 13-26', '2nd trimester'],
            'third_trimester': ['third trimester', 'late pregnancy', 'weeks 27-40', '3rd trimester'],
            'labor_delivery': ['labor', 'labour', 'delivery', 'birth', 'childbirth', 'intrapartum'],
            'postpartum': ['postpartum', 'postnatal', 'after birth', 'puerperium', 'fourth trimester', '4th trimester']
        }
        
        # Common maternal health topics
        self.maternal_topics = {
            'fetal_development': ['development', 'growth', 'fetal', 'foetal', 'baby', 'fetus', 'embryo'],
            'maternal_changes': ['mother', 'maternal', 'body', 'changes', 'symptoms', 'physical'],
            'nutrition': ['diet', 'food', 'nutrition', 'eat', 'meal', 'vitamin', 'nutrient', 'folate', 'iron'],
            'warning_signs': ['danger', 'warning', 'risk', 'emergency', 'complication', 'problem'],
            'normal_variations': ['normal', 'common', 'usual', 'typical', 'variant', 'variation', 'expected'],
            'medical_tests': ['test', 'scan', 'ultrasound', 'screening', 'monitor', 'measurement'],
            'delivery': ['birth', 'labor', 'labour', 'delivery', 'contractions', 'childbirth'],
            'postpartum': ['after birth', 'post-delivery', 'postpartum', 'post-natal', 'breastfeeding'],
            'mental_health': ['mental', 'depression', 'anxiety', 'mood', 'stress', 'emotional'],
            'prenatal_care': ['prenatal', 'antenatal', 'checkup', 'visit', 'appointment', 'healthcare']
        }
        
        # Common patient concerns (for enhanced retrieval)
        self.patient_concerns = {
            'pain': ['pain', 'ache', 'discomfort', 'hurt', 'sore', 'cramp'],
            'bleeding': ['bleed', 'blood', 'spotting', 'hemorrhage', 'discharge'],
            'movement': ['movement', 'kick', 'active', 'moving', 'stirring', 'flutter'],
            'nausea': ['nausea', 'vomit', 'sick', 'morning sickness', 'hyperemesis'],
            'sleep': ['sleep', 'insomnia', 'tired', 'fatigue', 'exhausted', 'rest'],
            'emotions': ['emotion', 'mood', 'cry', 'anxiety', 'worry', 'fear', 'depression'],
            'nutrition': ['eat', 'food', 'diet', 'craving', 'appetite', 'weight', 'nutrition']
        }
        
        # Medical entities
        self.medical_entities = {
            'conditions': [
                'preeclampsia', 'eclampsia', 'gestational diabetes', 'anemia', 'anaemia',
                'hypertension', 'nausea', 'vomiting', 'morning sickness', 'hyperemesis',
                'placenta previa', 'preterm labor', 'preterm birth', 'ectopic pregnancy',
                'miscarriage', 'bleeding', 'spotting', 'edema', 'swelling', 'back pain',
                'postpartum depression', 'postpartum hemorrhage', 'mastitis', 'HELLP syndrome',
                'gestational hypertension', 'cholestasis', 'UTI', 'yeast infection'
            ],
            'medications': [
                'iron', 'folate', 'folic acid', 'vitamin', 'supplement', 'prenatal vitamin',
                'calcium', 'magnesium', 'zinc', 'omega-3', 'aspirin', 'antacid', 'insulin',
                'labetalol', 'methyldopa', 'nifedipine', 'paracetamol', 'acetaminophen',
                'ibuprofen', 'metformin', 'heparin', 'progesterone', 'oxytocin', 'epidural'
            ],
            'procedures': [
                'ultrasound', 'scan', 'test', 'screening', 'amniocentesis', 'delivery',
                'c-section', 'cesarean', 'caesarean', 'induction', 'epidural', 'monitoring',
                'glucose test', 'blood test', 'urine test', 'group b strep', 'genetic screening',
                'CVS', 'chorionic villus sampling', 'NIPT', 'NST', 'non-stress test',
                'biophysical profile', 'cervical check', 'membrane sweep'
            ],
            'measurements': [
                'weight', 'blood pressure', 'fundal height', 'bp', 'heart rate', 'fetal heart rate',
                'bpm', 'glucose', 'temperature', 'hcg', 'hemoglobin', 'haemoglobin', 'blood sugar',
                'contractions', 'dilation', 'effacement', 'station', 'bilirubin', 'amniotic fluid index'
            ]
        }
    
    @lru_cache(maxsize=32)
    def _extract_document_metadata(self, text_sample: str, filename: str = None) -> Dict[str, Any]:
        """
        Extract document-level metadata from text sample.
        
        Args:
            text_sample: Representative sample of document text
            filename: Original document filename
            
        Returns:
            Dict containing document metadata
        """
        text_lower = text_sample.lower()
        metadata = {
            "filename": filename,
            "document_categories": []
        }
        
        # Extract document categories
        for category, keywords in self.doc_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                metadata["document_categories"].append(category)
        
        # Determine document type by content and filename patterns
        doc_type = "Maternal Health Document"  # Default
        
        if filename:
            filename_lower = filename.lower()
            if any(term in filename_lower for term in ['guideline', 'guide', 'protocol']):
                if 'anc' in filename_lower or 'prenatal' in filename_lower or 'antenatal' in filename_lower:
                    doc_type = "Antenatal Care Guideline"
                elif 'pnc' in filename_lower or 'postnatal' in filename_lower or 'postpartum' in filename_lower:
                    doc_type = "Postnatal Care Guideline"
                elif 'nutr' in filename_lower or 'diet' in filename_lower or 'food' in filename_lower:
                    doc_type = "Maternal Nutrition Guide"
                elif 'ppd' in filename_lower or 'depress' in filename_lower or 'mental' in filename_lower:
                    doc_type = "Maternal Mental Health Guide"
                else:
                    doc_type = "Clinical Guideline"
        
        # Look for title in content
        title_pattern = re.compile(r'^[A-Z][\w\s:]+(?:Guidelines?|Protocol|Care|Health|Management)', re.MULTILINE)
        title_match = title_pattern.search(text_sample[:2000])
        if title_match:
            metadata["title"] = title_match.group(0).strip()
        else:
            # Try to construct title from filename if available
            if filename:
                base_name = os.path.splitext(os.path.basename(filename))[0]
                # Convert snake_case or kebab-case to Title Case
                base_name = re.sub(r'[_-]', ' ', base_name).title()
                metadata["title"] = base_name
            else:
                metadata["title"] = doc_type
        
        metadata["document_type"] = doc_type
        
        # Extract primary pregnancy stages
        primary_stages = []
        for stage, keywords in self.pregnancy_stages.items():
            if any(keyword in text_lower for keyword in keywords):
                primary_stages.append(stage)
        
        metadata["primary_stages"] = primary_stages
        
        return metadata
    
    def extract_pregnancy_stages(self, text: str) -> Dict[str, List]:
        """
        Extract all pregnancy stages mentioned in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with pregnancy stage information
        """
        stages = {
            "weeks": [],
            "months": [],
            "trimesters": [],
            "named_stages": []
        }
        
        # Extract weeks
        for match in self.stage_patterns['week'].finditer(text):
            week_str = match.group(1) or match.group(2)
            if week_str:
                if '-' in week_str:
                    try:
                        start, end = map(int, week_str.split('-'))
                        stages["weeks"].extend(list(range(start, end + 1)))
                    except ValueError:
                        pass
                else:
                    try:
                        stages["weeks"].append(int(week_str))
                    except ValueError:
                        pass
        
        # Extract months
        for match in self.stage_patterns['month'].finditer(text):
            month_str = match.group(1) or match.group(2)
            if month_str:
                if '-' in month_str:
                    try:
                        start, end = map(int, month_str.split('-'))
                        stages["months"].extend(list(range(start, end + 1)))
                    except ValueError:
                        pass
                else:
                    try:
                        stages["months"].append(int(month_str))
                    except ValueError:
                        pass
        
        # Extract trimesters
        for match in self.stage_patterns['trimester'].finditer(text):
            trimester = match.group(1).lower()
            stages["trimesters"].append(trimester)
        
        # Extract named stages
        for match in self.stage_patterns['stage'].finditer(text):
            named_stage = match.group(1).lower()
            stages["named_stages"].append(named_stage)
        
        # Remove duplicates
        for key in stages:
            stages[key] = list(set(stages[key]))
            
        # Map to standard stage categorization
        standard_stages = set()
        
        # Map from weeks to standard stages
        for week in stages["weeks"]:
            if 1 <= week <= 12:
                standard_stages.add("first_trimester")
            elif 13 <= week <= 26:
                standard_stages.add("second_trimester")
            elif 27 <= week <= 42:
                standard_stages.add("third_trimester")
        
        # Map from months to standard stages
        for month in stages["months"]:
            if 1 <= month <= 3:
                standard_stages.add("first_trimester")
            elif 4 <= month <= 6:
                standard_stages.add("second_trimester")
            elif 7 <= month <= 10:
                standard_stages.add("third_trimester")
        
        # Map from named trimesters
        for trimester in stages["trimesters"]:
            if trimester == "first":
                standard_stages.add("first_trimester")
            elif trimester == "second":
                standard_stages.add("second_trimester")
            elif trimester == "third":
                standard_stages.add("third_trimester")
        
        # Map from named stages
        for stage in stages["named_stages"]:
            if stage in ["preconception", "pre-conception"]:
                standard_stages.add("preconception")
            elif stage in ["prenatal", "antenatal"]:
                # These are general terms - don't add a specific stage
                pass
            elif stage == "perinatal":
                standard_stages.add("labor_delivery")
            elif stage == "intrapartum":
                standard_stages.add("labor_delivery")
            elif stage in ["postpartum", "postnatal"]:
                standard_stages.add("postpartum")
        
        stages["standard_stages"] = list(standard_stages)
        
        return stages
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text using pattern matching and NLP.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with medical entities by category
        """
        text_lower = text.lower()
        
        entities = {
            "conditions": [],
            "medications": [],
            "procedures": [],
            "measurements": []
        }
        
        # Extract entities using pattern matching
        for entity_type, entity_list in self.medical_entities.items():
            for entity in entity_list:
                if entity in text_lower:
                    entities[entity_type].append(entity)
        
        # Extract measurements using regex
        for match in self.clinical_patterns['measurement'].finditer(text):
            measurement = match.group(0)
            if measurement not in entities["measurements"]:
                entities["measurements"].append(measurement)
        
        # Extract dosages using regex
        for match in self.clinical_patterns['dosage'].finditer(text):
            dosage = match.group(0)
            if dosage not in entities["medications"]:
                entities["medications"].append(dosage)
        
        # Use SpaCy for enhanced entity extraction if available
        if self.use_spacy and self.nlp is not None:
            try:
                # Process with SpaCy - limit text length to avoid memory issues
                # Truncate to ~10,000 chars for efficiency
                truncated_text = text[:10000] if len(text) > 10000 else text
                doc = self.nlp(truncated_text)
                
                # Extract medical entities from SpaCy
                for ent in doc.ents:
                    ent_text = ent.text.lower()
                    
                    # Map SpaCy entities to our categories
                    if ent.label_ in ["CONDITION", "DISEASE", "PROBLEM"]:
                        if ent_text not in entities["conditions"]:
                            entities["conditions"].append(ent_text)
                    elif ent.label_ in ["TREATMENT", "PROCEDURE"]:
                        if ent_text not in entities["procedures"]:
                            entities["procedures"].append(ent_text)
                    elif ent.label_ in ["MEDICATION", "DRUG"]:
                        if ent_text not in entities["medications"]:
                            entities["medications"].append(ent_text)
                    elif ent.label_ in ["QUANTITY"] and any(unit in ent_text for unit in ["mg", "kg", "cm", "mmHg", "bpm"]):
                        if ent_text not in entities["measurements"]:
                            entities["measurements"].append(ent_text)
            except Exception as e:
                logger.warning(f"Error during SpaCy processing: {e}")
        
        return entities
    
    def identify_topics_and_concerns(self, text: str) -> Dict[str, List[str]]:
        """
        Identify maternal health topics and patient concerns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with identified topics and concerns
        """
        text_lower = text.lower()
        results = {
            "topics": [],
            "concerns": [],
            "content_types": []
        }
        
        # Identify topics
        for topic, keywords in self.maternal_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                results["topics"].append(topic)
        
        # Identify patient concerns
        for concern, keywords in self.patient_concerns.items():
            if any(keyword in text_lower for keyword in keywords):
                results["concerns"].append(concern)
        
        # Identify content types
        content_indicators = {
            "warning": ["warning", "caution", "danger", "alert", "emergency", "attention"],
            "recommendation": ["recommend", "should", "advised", "advise", "important", "must"],
            "information": ["information", "facts", "overview", "background", "introduction"],
            "instruction": ["step", "steps", "procedure", "instructions", "follow", "method"],
            "faq": ["question", "answer", "frequently", "asked", "faq", "q&a", "q:", "a:"]
        }
        
        for content_type, indicators in content_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                results["content_types"].append(content_type)
        
        # Check for lists and tables using regex
        if self.preservation_patterns['list'].search(text):
            results["content_types"].append("list")
        
        if self.preservation_patterns['table'].search(text):
            results["content_types"].append("table")
        
        return results
    
    def split_text_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into semantically meaningful sections with headers.
        
        Args:
            text: Full document text
            
        Returns:
            List of (header, section_text) tuples
        """
        # If text is small enough, don't bother splitting
        if len(text) < 5000:
            return [("", text)]
        
        # Find all semantic boundaries using header patterns
        header_positions = []
        
        # Collect positions of all types of headers
        for pattern_name, pattern in self.structure_patterns.items():
            if pattern_name in ['formal_header', 'nl_header', 'special_marker']:
                for match in pattern.finditer(text):
                    header_positions.append((match.start(), match.group(0)))
        
        # Sort header positions by their occurrence in text
        header_positions.sort(key=lambda x: x[0])
        
        # Filter out headers that are too close together (likely not real section headers)
        if header_positions:
            filtered_positions = [(0, "")]  # Start with the beginning of text
            for pos, header in header_positions:
                # Only keep headers that are at least 500 chars from the previous header
                if pos - filtered_positions[-1][0] >= 500:
                    filtered_positions.append((pos, header))
            
            # If we only have the start position, there are no useful headers
            if len(filtered_positions) == 1:
                return [("", text)]
            
            # Create sections based on header positions
            sections = []
            for i in range(len(filtered_positions) - 1):
                start_pos, header = filtered_positions[i]
                end_pos = filtered_positions[i+1][0]
                
                # Skip the first "header" which is just the start position
                if i == 0 and start_pos == 0 and header == "":
                    continue
                
                section_text = text[start_pos:end_pos].strip()
                if section_text:
                    sections.append((header.strip(), section_text))
            
            # Don't forget the last section
            last_pos, last_header = filtered_positions[-1]
            last_section = text[last_pos:].strip()
            if last_section:
                sections.append((last_header.strip(), last_section))
            
            return sections if sections else [("", text)]
        else:
            # If no headers detected, return the whole text
            return [("", text)]
    
    def find_preservation_blocks(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Find blocks of text that should be preserved during chunking.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of (start_pos, end_pos, block_text, block_type) tuples
        """
        preserved_blocks = []
        
        # Check each preservation pattern
        for block_type, pattern in self.preservation_patterns.items():
            for match in pattern.finditer(text):
                preserved_blocks.append((match.start(), match.end(), match.group(0), block_type))
        
        # Sort by start position
        preserved_blocks.sort(key=lambda x: x[0])
        
        return preserved_blocks
    
    def create_semantic_chunks(self, section_text: str, section_header: str, doc_metadata: Dict[str, Any]) -> List[Document]:
        """
        Create semantically meaningful chunks from a section of text.
        
        Args:
            section_text: Text of the section to chunk
            section_header: Header of the section
            doc_metadata: Document-level metadata
            
        Returns:
            List of Document objects with chunks and metadata
        """
        # If section is small enough, keep it as one chunk
        if len(section_text) <= self.chunk_size:
            return [self._create_chunk_document(section_text, section_header, doc_metadata)]
        
        # Find blocks that should be preserved
        preserved_blocks = self.find_preservation_blocks(section_text)
        
        chunks = []
        
        # If we have SpaCy available, use sentence boundaries for better chunking
        if self.use_spacy and self.nlp is not None:
            # Handle preserved blocks first
            if preserved_blocks:
                # Create a map of positions to skip (preserved blocks)
                skip_ranges = [(start, end) for start, end, _, _ in preserved_blocks]
                
                # Process text outside preserved blocks with sentence boundaries
                current_pos = 0
                current_chunk = []
                current_length = 0
                
                try:
                    # Process text in chunks to avoid memory issues with very large texts
                    segment_length = 50000  # Process 50k chars at a time
                    for i in range(0, len(section_text), segment_length):
                        segment = section_text[i:i+segment_length]
                        
                        # Adjust positions to account for the segment offset
                        segment_offset = i
                        
                        # Process with SpaCy for sentence boundaries
                        doc = self.nlp(segment)
                        sentences = list(doc.sents)
                        
                        for sentence in sentences:
                            # Adjust sentence positions to absolute document position
                            abs_start = sentence.start_char + segment_offset
                            abs_end = sentence.end_char + segment_offset
                            
                            # Check if this sentence overlaps with any preserved block
                            in_preserved_block = False
                            for start, end in skip_ranges:
                                if (start <= abs_start < end) or (start < abs_end <= end) or (abs_start <= start and abs_end >= end):
                                    in_preserved_block = True
                                    break
                            
                            if in_preserved_block:
                                continue
                                
                            sentence_text = sentence.text.strip()
                            
                            # If adding this sentence exceeds chunk size, create a new chunk
                            if current_length + len(sentence_text) > self.chunk_size and current_chunk:
                                chunk_text = ' '.join(current_chunk)
                                chunks.append(self._create_chunk_document(chunk_text, section_header, doc_metadata))
                                current_chunk = []
                                current_length = 0
                            
                            # Add sentence to current chunk
                            current_chunk.append(sentence_text)
                            current_length += len(sentence_text) + 1  # +1 for space
                        
                except Exception as e:
                    logger.warning(f"Error during SpaCy processing: {e}")
                    # Fall back to simpler chunking method
                    return self._simple_chunk_section(section_text, section_header, doc_metadata, preserved_blocks)
                
                # Add final chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk_document(chunk_text, section_header, doc_metadata))
                
                # Now add preserved blocks as separate chunks
                for start, end, block_text, block_type in preserved_blocks:
                    block_metadata = doc_metadata.copy()
                    block_metadata["content_type"] = block_type
                    chunks.append(self._create_chunk_document(block_text, section_header, block_metadata))
            else:
                # No preserved blocks, process entire text with sentence boundaries
                try:
                    # Process text in chunks to avoid memory issues
                    current_chunk = []
                    current_length = 0
                    
                    # Process text in segments
                    segment_length = 50000  # Process 50k chars at a time
                    for i in range(0, len(section_text), segment_length):
                        segment = section_text[i:i+segment_length]
                        
                        # Process with SpaCy for sentence boundaries
                        doc = self.nlp(segment)
                        sentences = list(doc.sents)
                        
                        for sentence in sentences:
                            sentence_text = sentence.text.strip()
                            
                            # If adding this sentence exceeds chunk size, create a new chunk
                            if current_length + len(sentence_text) > self.chunk_size and current_chunk:
                                chunk_text = ' '.join(current_chunk)
                                chunks.append(self._create_chunk_document(chunk_text, section_header, doc_metadata))
                                current_chunk = []
                                current_length = 0
                            
                            # Add sentence to current chunk
                            current_chunk.append(sentence_text)
                            current_length += len(sentence_text) + 1  # +1 for space
                    
                    # Add final chunk
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(self._create_chunk_document(chunk_text, section_header, doc_metadata))
                        
                except Exception as e:
                    logger.warning(f"Error during SpaCy processing: {e}")
                    # Fall back to simpler chunking method
                    return self._simple_chunk_section(section_text, section_header, doc_metadata, preserved_blocks)
        else:
            # SpaCy not available, use simpler chunking
            chunks = self._simple_chunk_section(section_text, section_header, doc_metadata, preserved_blocks)
        
        return chunks
    
    def _simple_chunk_section(self, text: str, header: str, metadata: Dict[str, Any], 
                            preserved_blocks: List[Tuple[int, int, str, str]]) -> List[Document]:
        """
        Chunk text using paragraph boundaries when SpaCy is not available.
        
        Args:
            text: Text to chunk
            header: Section header
            metadata: Document metadata
            preserved_blocks: List of preserved blocks
            
        Returns:
            List of Document objects
        """
        chunks = []
        
        # Handle preserved blocks first
        if preserved_blocks:
            # Sort preserved blocks by position
            preserved_blocks.sort(key=lambda x: x[0])
            
            # Keep track of the current position
            current_pos = 0
            
            # Process text segments between preserved blocks
            for start, end, block_text, block_type in preserved_blocks:
                # Handle text before this preserved block
                if start > current_pos:
                    preceding_text = text[current_pos:start].strip()
                    if preceding_text:
                        # Chunk this text by paragraphs
                        preceding_chunks = self._chunk_by_paragraphs(preceding_text, header, metadata)
                        chunks.extend(preceding_chunks)
                
                # Add the preserved block as its own chunk
                block_metadata = metadata.copy()
                block_metadata["content_type"] = block_type
                chunks.append(self._create_chunk_document(block_text, header, block_metadata))
                
                # Update current position
                current_pos = end
            
            # Handle text after the last preserved block
            if current_pos < len(text):
                remaining_text = text[current_pos:].strip()
                if remaining_text:
                    # Chunk this text by paragraphs
                    remaining_chunks = self._chunk_by_paragraphs(remaining_text, header, metadata)
                    chunks.extend(remaining_chunks)
        else:
            # No preserved blocks, chunk entire text by paragraphs
            chunks = self._chunk_by_paragraphs(text, header, metadata)
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, header: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split text into chunks based on paragraph boundaries.
        
        Args:
            text: Text to chunk
            header: Section header
            metadata: Document metadata
            
        Returns:
            List of Document objects
        """
        chunks = []
        
        # Split on paragraph boundaries (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed chunk size and we already have content,
            # create a new chunk
            if current_length + len(paragraph) > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk_document(chunk_text, header, metadata))
                current_chunk = []
                current_length = 0
            
            # If a single paragraph is larger than chunk size, we need to split it
            if len(paragraph) > self.chunk_size:
                # If we have accumulated content, create a chunk first
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk_document(chunk_text, header, metadata))
                    current_chunk = []
                    current_length = 0
                
                # Split the large paragraph by sentences (using basic regex)
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                
                sentence_chunk = []
                sentence_length = 0
                
                for sentence in sentences:
                    # If this sentence alone exceeds chunk size, we have to include it anyway
                    # Users will just have a larger-than-ideal chunk in this case
                    if sentence_length + len(sentence) > self.chunk_size and sentence_chunk:
                        sentence_text = ' '.join(sentence_chunk)
                        chunks.append(self._create_chunk_document(sentence_text, header, metadata))
                        sentence_chunk = []
                        sentence_length = 0
                    
                    sentence_chunk.append(sentence)
                    sentence_length += len(sentence) + 1  # +1 for space
                
                # Add final sentence chunk if any
                if sentence_chunk:
                    sentence_text = ' '.join(sentence_chunk)
                    chunks.append(self._create_chunk_document(sentence_text, header, metadata))
            else:
                # Normal sized paragraph
                current_chunk.append(paragraph)
                current_length += len(paragraph) + 2  # +2 for newlines
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk_document(chunk_text, header, metadata))
        
        return chunks
    
    def _create_chunk_document(self, text: str, header: str, base_metadata: Dict[str, Any]) -> Document:
        """
        Create a document object with metadata from chunk text.
        
        Args:
            text: Chunk text
            header: Section header
            base_metadata: Base document metadata
            
        Returns:
            Document object with metadata
        """
        # Create a copy of base metadata
        metadata = base_metadata.copy()
        
        # Add section header
        if header:
            metadata["section_header"] = header
        
        # Extract pregnancy stages
        stages = self.extract_pregnancy_stages(text)
        metadata["pregnancy_stages"] = stages
        
        # Extract medical entities
        entities = self.extract_medical_entities(text)
        metadata["medical_entities"] = entities
        
        # Identify topics and concerns
        topics_and_concerns = self.identify_topics_and_concerns(text)
        metadata["topics"] = topics_and_concerns["topics"]
        metadata["patient_concerns"] = topics_and_concerns["concerns"]
        
        # Set content type if not already set
        if "content_type" not in metadata:
            content_types = topics_and_concerns["content_types"]
            if content_types:
                metadata["content_type"] = content_types[0]  # Primary content type
                metadata["all_content_types"] = content_types
            else:
                metadata["content_type"] = "general_text"
        
        # Create unique ID for the chunk
        metadata["chunk_id"] = str(uuid.uuid4())
        
        return Document(page_content=text, metadata=metadata)
    
    def create_synthetic_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Create synthetic chunks for improved retrieval.
        
        Args:
            documents: Original document chunks
            
        Returns:
            List of synthetic Document objects
        """
        synthetic_chunks = []
        logger.info("Creating synthetic chunks for improved retrieval...")
        
        # Group documents by stage
        stage_groups = {}
        
        for doc in documents:
            stages = doc.metadata.get('pregnancy_stages', {}).get('standard_stages', [])
            if not stages:
                continue
                
            for stage in stages:
                if stage not in stage_groups:
                    stage_groups[stage] = []
                stage_groups[stage].append(doc)
        
        # Create stage+topic chunks
        for stage, stage_docs in stage_groups.items():
            # Skip if too few documents
            if len(stage_docs) < 3:
                continue
                
            # Group by topics within stage
            topic_groups = {}
            
            for doc in stage_docs:
                for topic in doc.metadata.get('topics', []):
                    if topic not in topic_groups:
                        topic_groups[topic] = []
                    topic_groups[topic].append(doc)
            
            # Create synthetic chunks for each topic with sufficient content
            for topic, topic_docs in topic_groups.items():
                if len(topic_docs) < 2:
                    continue
                    
                # Select best content (at most 5 chunks or 5000 chars)
                selected_docs = []
                total_length = 0
                
                # Prioritize docs with specific content types
                for content_type in ["recommendation", "information", "table", "list"]:
                    for doc in topic_docs:
                        if doc.metadata.get('content_type') == content_type:
                            if len(selected_docs) < 5 and total_length < 5000:
                                selected_docs.append(doc)
                                total_length += len(doc.page_content)
                
                # Add other docs if needed
                for doc in topic_docs:
                    if doc not in selected_docs:
                        if len(selected_docs) < 5 and total_length < 5000:
                            selected_docs.append(doc)
                            total_length += len(doc.page_content)
                
                if not selected_docs:
                    continue
                
                # Create synthetic content
                synthetic_content = f"# {stage.replace('_', ' ').title()} - {topic.replace('_', ' ').title()}\n\n"
                
                for i, doc in enumerate(selected_docs):
                    # Add section header if available
                    if "section_header" in doc.metadata:
                        synthetic_content += f"## {doc.metadata['section_header']}\n\n"
                    else:
                        synthetic_content += f"## Information {i+1}\n\n"
                    
                    synthetic_content += doc.page_content + "\n\n"
                
                # Create synthetic document
                synthetic_doc = Document(
                    page_content=synthetic_content,
                    metadata={
                        "document_type": "Synthetic Document",
                        "title": f"{stage.replace('_', ' ').title()} - {topic.replace('_', ' ').title()}",
                        "is_synthetic": True,
                        "source_stage": stage,
                        "source_topic": topic,
                        "content_type": "synthetic",
                        "pregnancy_stages": {
                            "standard_stages": [stage]
                        },
                        "topics": [topic],
                        "source_count": len(selected_docs),
                        "chunk_id": f"synthetic_{stage}_{topic}_{uuid.uuid4().hex[:8]}"
                    }
                )
                
                synthetic_chunks.append(synthetic_doc)
        
        # Create FAQ chunks for common questions
        synthetic_chunks.extend(self._create_faq_chunks(documents))
        
        logger.info(f"Created {len(synthetic_chunks)} synthetic chunks")
        return synthetic_chunks
    
    def _create_faq_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Create FAQ-oriented chunks for common patient questions.
        
        Args:
            documents: Original documents
            
        Returns:
            List of FAQ Document objects
        """
        faq_chunks = []
        
        # Common questions by stage and concern
        faq_templates = {
            "first_trimester": [
                ("What symptoms are normal in the first trimester?", ["symptoms", "normal_variations", "maternal_changes"]),
                ("What foods should I avoid in early pregnancy?", ["nutrition", "warning_signs"]),
                ("Is spotting normal in early pregnancy?", ["bleeding", "normal_variations"])
            ],
            "second_trimester": [
                ("When will I feel the baby move?", ["movement", "fetal_development"]),
                ("What tests are done in the second trimester?", ["medical_tests", "prenatal_care"]),
                ("Is it normal to have pain in my abdomen as my belly grows?", ["pain", "normal_variations"])
            ],
            "third_trimester": [
                ("What are signs of labor?", ["delivery", "warning_signs"]),
                ("How can I relieve back pain in late pregnancy?", ["pain", "maternal_changes"]),
                ("How do I know if my water broke?", ["delivery"])
            ],
            "postpartum": [
                ("How much bleeding is normal after birth?", ["bleeding", "normal_variations", "postpartum"]),
                ("How can I tell if I have postpartum depression?", ["mental_health", "postpartum"]),
                ("When will my milk come in?", ["postpartum", "breastfeeding"])
            ]
        }
        
        # Create FAQs for each stage
        for stage, questions in faq_templates.items():
            for question, topics in questions:
                # Find relevant documents
                relevant_docs = []
                
                for doc in documents:
                    # Check if the document matches stage and topics
                    doc_stages = doc.metadata.get('pregnancy_stages', {}).get('standard_stages', [])
                    doc_topics = doc.metadata.get('topics', [])
                    doc_concerns = doc.metadata.get('patient_concerns', [])
                    
                    # Match if document has the right stage and at least one matching topic
                    if stage in doc_stages and any(topic in doc_topics for topic in topics):
                        relevant_docs.append(doc)
                
                # Skip if insufficient relevant documents
                if len(relevant_docs) < 2:
                    continue
                
                # Select best content (maximum 4 docs or 4000 chars)
                selected_docs = []
                total_length = 0
                
                # First add recommendations and key information
                for doc in sorted(relevant_docs, 
                                 key=lambda d: 1 if d.metadata.get('content_type') == 'recommendation' else 2):
                    if len(selected_docs) < 4 and total_length < 4000:
                        selected_docs.append(doc)
                        total_length += len(doc.page_content)
                
                # Create FAQ content
                faq_content = f"# FAQ: {question}\n\n"
                faq_content += "## Answer\n\n"
                
                # Add content from selected documents
                for doc in selected_docs:
                    faq_content += doc.page_content + "\n\n"
                
                # Create FAQ document
                faq_doc = Document(
                    page_content=faq_content,
                    metadata={
                        "document_type": "FAQ",
                        "title": question,
                        "is_faq": True,
                        "question": question,
                        "source_stage": stage,
                        "source_topics": topics,
                        "content_type": "faq",
                        "pregnancy_stages": {
                            "standard_stages": [stage]
                        },
                        "topics": topics,
                        "chunk_id": f"faq_{stage}_{uuid.uuid4().hex[:8]}"
                    }
                )
                
                faq_chunks.append(faq_doc)
        
        return faq_chunks
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Process a single document into semantically meaningful chunks.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of Document objects
        """
        logger.info(f"Processing {file_path}...")
        
        # Load document text
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                text = "\n\n".join([page.page_content for page in pages])
            elif file_path.lower().endswith('.html'):
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    html = f.read()
                soup = BeautifulSoup(html, "html.parser")
                # get_text keeps readability; separator inserts newlines for headings/paras
                text = soup.get_text(separator="\n")
            else:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
        
        # Extract document-level metadata
        filename = os.path.basename(file_path)
        doc_metadata = self._extract_document_metadata(text[:10000], filename)
        
        # Split into sections
        logger.info(f"Splitting {filename} into sections...")
        sections = self.split_text_into_sections(text)
        logger.info(f"Found {len(sections)} sections")
        
        # Process each section into chunks
        all_chunks = []
        
        for header, section_text in tqdm(sections, desc="Processing sections"):
            chunks = self.create_semantic_chunks(section_text, header, doc_metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {filename}")
        return all_chunks
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            List of all Document objects
        """
        all_documents = []
        
        # Get all files in directory
        file_paths = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                file_paths.append(file_path)
        
        # Process each file
        logger.info(f"Processing {len(file_paths)} files in {directory_path}")
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                chunks = self.process_document(file_path)
                all_documents.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        return all_documents
    
    def create_vector_store(self, documents: List[Document], include_synthetic: bool = True) -> FAISS:
        """
        Create vector store from processed documents.
        
        Args:
            documents: List of Document objects
            include_synthetic: Whether to include synthetic chunks
            
        Returns:
            FAISS vector store
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        # Create synthetic chunks if requested
        if include_synthetic:
            synthetic_chunks = self.create_synthetic_chunks(documents)
            all_docs = documents + synthetic_chunks
        else:
            all_docs = documents
        
        logger.info(f"Creating vector store with {len(all_docs)} documents...")
        
        # Initialize FAISS index from documents
        self.db = FAISS.from_documents(
            documents=all_docs,
            embedding=self.embeddings
        )
        
        # Save to disk
        logger.info(f"Saving vector store to {self.db_directory}")
        self.db.save_local(self.db_directory)
        
        return self.db
    
    def create_stage_optimized_retrievers(self) -> Dict[str, Any]:
        """Create stage-optimized retrievers for each pregnancy stage."""
        if not self.db:
            raise ValueError("No vector store available. Please create or load a vector store first.")
        
        # Define standard pregnancy stages
        stages = [
            "preconception",
            "first_trimester",
            "second_trimester", 
            "third_trimester",
            "labor_delivery",
            "postpartum"
        ]
        
        retrievers = {}
        
        # Create a retriever for each stage with metadata filtering
        for stage in stages:
            # Define filter function that works with metadata dictionaries
            def stage_filter(metadata):
                # Get pregnancy stages from metadata
                pregnancy_stages = metadata.get("pregnancy_stages", {})
                if isinstance(pregnancy_stages, dict):
                    standard_stages = pregnancy_stages.get("standard_stages", [])
                else:
                    standard_stages = []
                    
                # Check if content is stage-agnostic
                is_general_content = (
                    not standard_stages and 
                    metadata.get("content_type") in ["recommendation", "information"]
                )
                    
                # Include if stage matches or it's general content
                return stage in standard_stages or is_general_content
            
            # Create retriever with this filter
            retrievers[stage] = self.db.as_retriever(
                search_kwargs={"k": 7, "filter": stage_filter}
            )
        
        # Create a general retriever for queries without clear stage
        retrievers["general"] = self.db.as_retriever(
            search_kwargs={"k": 5}
        )
    
        return retrievers
    
    def detect_query_context(self, query: str) -> Dict[str, Any]:
        """
        Detect pregnancy stage and concerns from user query.
        
        Args:
            query: User query text
            
        Returns:
            Dict with detected context
        """
        context = {
            "stage": None,
            "concerns": [],
            "detected_topics": []
        }
        
        # Detect pregnancy stage
        # First look for explicit stage mentions
        query_lower = query.lower()
        
        for stage, keywords in self.pregnancy_stages.items():
            if any(keyword in query_lower for keyword in keywords):
                context["stage"] = stage
                break
        
        # If no explicit stage, check for week/month/trimester mentions
        if not context["stage"]:
            stages = self.extract_pregnancy_stages(query)
            if stages["standard_stages"]:
                context["stage"] = stages["standard_stages"][0]
        
        # Detect concerns
        for concern, keywords in self.patient_concerns.items():
            if any(keyword in query_lower for keyword in keywords):
                context["concerns"].append(concern)
        
        # Detect topics
        for topic, keywords in self.maternal_topics.items():
            if any(keyword in query_lower for keyword in keywords):
                context["detected_topics"].append(topic)
        
        return context
    
    def retrieve_for_query(self, query: str) -> List[Document]:
        """Smart retrieval based on query context."""
        if not self.db:
            raise ValueError("No vector store available. Please create or load a vector store first.")
        
        # Detect query context
        context = self.detect_query_context(query)
        logger.info(f"Query context: {context}")
        
        # Create stage-specific retrievers if not already created
        if not hasattr(self, 'stage_retrievers'):
            self.stage_retrievers = self.create_stage_optimized_retrievers()
        
        # Select appropriate retriever based on context
        if context["stage"]:
            retriever = self.stage_retrievers.get(context["stage"])
            if not retriever:
                # Fall back to general retriever
                retriever = self.stage_retrievers["general"]
        else:
            # Use general retriever for stage-agnostic queries
            retriever = self.stage_retrievers["general"]
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        
        # If we have specific concerns, try to ensure we have documents addressing them
        if context["concerns"] and len(docs) >= 3:
            # Check if concerns are addressed in retrieved docs
            concerns_addressed = False
            for doc in docs[:3]:  # Check top 3 docs
                doc_concerns = doc.metadata.get("patient_concerns", [])
                if any(concern in doc_concerns for concern in context["concerns"]):
                    concerns_addressed = True
                    break
        
        # If concerns not addressed in top results, add specific concern filters
            if not concerns_addressed:
                # Updated filter function to work with metadata dictionaries
                def concern_filter(metadata):
                    # Get patient concerns from metadata
                    patient_concerns = metadata.get("patient_concerns", [])
                    # Check if any of our concerns are in the document's concerns
                    return any(concern in patient_concerns for concern in context["concerns"])
                
                concern_retriever = self.db.as_retriever(
                    search_kwargs={"k": 2, "filter": concern_filter}
                )
                
                # Get concern-specific docs and add to results
                concern_docs = concern_retriever.get_relevant_documents(query)
                
                # Add unique concern docs (avoid duplicates)
                existing_ids = {doc.metadata.get("chunk_id", "") for doc in docs}
                for doc in concern_docs:
                    if doc.metadata.get("chunk_id", "") not in existing_ids:
                        docs.append(doc)
        
        return docs
    
    def create_qa_chain(self, model_name: str = "gpt-4", temperature: float = 0.0) -> RetrievalQA:
        """
        Create a question answering chain.
        
        Args:
            model_name: LLM model name
            temperature: LLM temperature
            
        Returns:
            RetrievalQA chain
        """
        if not self.db:
            raise ValueError("No vector store available. Please create or load a vector store first.")
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        
        # Create custom prompt for maternal health
        qa_prompt = PromptTemplate(
            template="""You are a maternal health assistant providing evidence-based information to pregnant women and new mothers.
            Use the following retrieved information to answer the question.
            
            If the information doesn't contain the answer, just say "I don't have enough information to answer that question accurately" - don't make up information.
            If providing medical advice, clearly state that this is general information and the person should consult their healthcare provider.
            Use a reassuring, supportive tone and clear, non-technical language.
            
            Retrieved information:
            {context}
            
            Question: {question}
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        # Create question answering chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
        
        return qa_chain
    
    def answer_question(self, question: str, model_name: str = "gpt-4") -> Dict[str, Any]:
        """
        Answer a maternal health question using the optimized retrieval pipeline.
        
        Args:
            question: User question
            model_name: LLM model name
            
        Returns:
            Dict with answer and source documents
        """
        # Retrieve relevant documents with context-aware retrieval
        docs = self.retrieve_for_query(question)
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.1  # Low temperature for factual responses
        )
        
        # Create maternal health-specific prompt
        # prompt = PromptTemplate(
        #     template="""You are a supportive maternal health assistant providing evidence-based information to pregnant women and new mothers.
            
        #     Use the following retrieved information to answer the question.
            
        #     Guidelines for your response:
        #     1. Be accurate - only use information from the provided context
        #     2. If the information doesn't fully answer the question, acknowledge the limitations
        #     3. Be reassuring but honest - don't minimize legitimate concerns
        #     4. Use clear, non-technical language
        #     5. Remind users to consult healthcare providers for personal medical advice
        #     6. If discussing warning signs, clearly distinguish between normal variations and concerning symptoms
            
        #     Retrieved information:
        #     {context}
            
        #     Question: {question}
            
        #     Answer:""",
        #     input_variables=["context", "question"]
        # )
        prompt = PromptTemplate(
            template="""You are a supportive maternal health assistant providing evidence-based information to pregnant women and new mothers.
Use the following retrieved information to answer the question.
Response Guidelines:
Answer ONLY using the provided context. Do not speculate or generate information not found in the retrieved data.


If the information doesn't fully answer the question, state the limitations. Example: "I don't have enough information to answer that accurately."


Be reassuring but honest. Do not minimize legitimate concerns. If the topic is serious, clearly explain when to seek medical help.


Use clear, non-technical language. Avoid jargon and explain complex terms simply.


Always remind users: "This is general information; please consult your healthcare provider for personal medical advice."


Warning signs: When discussing symptoms or warning signs, clearly distinguish between normal variations and symptoms that require medical attention. List these separately if appropriate.


If you detect any signs of a possible emergency (such as heavy bleeding, severe pain, loss of consciousness, seizures, absent fetal movement, severe swelling, difficulty breathing, fever with rash, premature rupture of membranes, or vision changes):


Do NOT give any advice.


Respond ONLY with:


"IMPORTANT: This appears to be an emergency situation that requires immediate medical attention.
This chatbot cannot provide emergency medical advice. Please:
1. Contact your healthcare provider immediately OR
2. Go to the nearest emergency department OR
3. Call your local emergency number for immediate assistance.

Do not wait or delay seeking professional care."

Do not provide:


Any specific drug brand or prescription recommendations.


Any information related to fetal gender detection or selection.


Any answers that are not about pregnancy, postpartum, or maternal/newborn wellness (if off-topic, say: "I'm sorry, I can only help with pregnancy and maternal-health questions.")


Any instructions or advice for suicidal/self-harm situations. Instead, respond only with the relevant crisis hotline (India: 9152987821, US: 988, UK: 0800 689 5652).



Retrieved information:
 {context}
Question:
 {question}
Answer:

""",
            input_variables=["context", "question"]
        )
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(),  # Default retriever not used since we pass docs directly
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        # Call QA chain with pre-retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        response = llm.predict(
            prompt.format(context=context, question=question)
        )
        
        return {
            "answer": response,
            "source_documents": docs
        }

# def main():
#     """Main function to run the maternal health pipeline."""
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Process maternal health documents")
#     parser.add_argument("--input", required=True, help="Input file or directory")
#     parser.add_argument("--output", default="maternal_health_db", help="Output directory for vector store")
#     parser.add_argument("--embedding", default="google/muril-base-cased", help="Embedding model name")
#     parser.add_argument("--no-synthetic", action="store_true", help="Skip synthetic chunk creation")
#     parser.add_argument("--chunk-size", type=int, default=2500, help="Chunk size")
#     parser.add_argument("--no-spacy", action="store_true", help="Disable SpaCy (use simpler chunking)")
    
#     args = parser.parse_args()
    
#     # Initialize pipeline
    # pipeline = MaternalHealthPipeline(
    #     chunk_size=args.chunk_size,
    #     embedding_model_name=args.embedding,
    #     db_directory=args.output,
    #     use_spacy=not args.no_spacy
    # )
    
#     # Process documents
#     if os.path.isdir(args.input):
#         documents = pipeline.process_directory(args.input)
#     else:
#         documents = pipeline.process_document(args.input)
    
#     # Create vector store
#     if documents:
#         pipeline.create_vector_store(documents, include_synthetic=not args.no_synthetic)
#         logger.info(f"Successfully processed {len(documents)} chunks and created vector store in {args.output}")
#     else:
#         logger.error("No documents were processed.")

# if __name__ == "__main__":
#     main()

pipeline = MaternalHealthPipeline(
        chunk_size=3000,
        embedding_model_name="google/muril-base-cased",
        db_directory="/data/user_data/vidhij2/medical_db/rag_2"
    )

documents = pipeline.process_directory("/data/user_data/vidhij2/medical/html")
    
document_2 = pipeline.process_directory("/data/user_data/vidhij2/medical/pdfs")
documents.extend(document_2)
def answer_maternal_health_query(pipeline, query):
    """
    Answer a maternal health query using the pipeline.
    
    Args:
        pipeline: Initialized MaternalHealthPipeline
        query: Question from a pregnant woman
        
    Returns:
        Answer with source information
    """
    print(f"\nProcessing query: {query}")
    
    # Detect context from query
    context = pipeline.detect_query_context(query)
    stage = context["stage"] or "general"
    concerns = ", ".join(context["concerns"]) if context["concerns"] else "none detected"
    print(f"Detected stage: {stage}, concerns: {concerns}")
    
    # Retrieve relevant documents
    print("Retrieving relevant information...")
    docs = pipeline.retrieve_for_query(query)
    
    # Print source information
    print(f"Found {len(docs)} relevant chunks:")
    for i, doc in enumerate(docs[:3]):  # Show top 3 sources
        source = doc.metadata.get("title", "Unknown document")
        content_type = doc.metadata.get("content_type", "general")
        print(f"  Source {i+1}: {source} ({content_type})")
    
    # Generate answer
    print("Generating answer...")
    response = pipeline.answer_question(query, model_name="gpt-4")
    
    return response

def demo_queries():
    """Run demo queries to test the pipeline."""
    return [
        "I'm 8 weeks pregnant and having morning sickness. Is this normal and what can I do?",
        "What foods should I avoid during my pregnancy?",
        "I'm in my third trimester and having trouble sleeping. Any advice?",
        "I'm feeling my baby move less today. Should I be concerned?", 
        "What are the warning signs I should look out for after giving birth?",
        "I'm 6 months pregnant and having headaches. What could be causing this?"
    ]

def interactive_mode(pipeline):
    """Run in interactive mode to answer user questions."""
    print("\n==== Maternal Health Assistant ====")
    print("Type your pregnancy or postpartum questions, or 'quit' to exit.")
    
    while True:
        query = input("\nQuestion: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        response = answer_maternal_health_query(pipeline, query)
        print("\nAnswer:")
        print(response["answer"])


