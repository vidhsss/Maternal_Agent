import re
from langchain.schema import Document
import PyPDF2
from typing import List

class MedicalPDFProcessor:
    """
    Processes medical PDFs with specialized techniques for handling medical content.
    """
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def split_into_sections(self, text: str) -> List[str]:
        """Split medical document into logical sections based on common headers."""
        common_sections = [
            "History", "Physical Examination", "Assessment", "Plan", "Diagnosis",
            "Chief Complaint", "Past Medical History", "Medications", "Allergies",
            "Family History", "Social History", "Review of Systems", "Labs",
            "Imaging", "Discussion", "Conclusion", "Recommendations"
        ]
        pattern = r'(?i)(?:^|\n)(' + '|'.join(re.escape(s) for s in common_sections) + r')(?::|:)?\s*(?:\n|\s)'
        matches = list(re.finditer(pattern, text))
        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i < len(matches) - 1 else len(text)
            header = match.group(1)
            content = text[start:end].strip()
            sections.append(f"{header}:\n{content}")
        if not sections:
            sections = [text]
        return sections

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a medical PDF and return LangChain Document objects."""
        text = self.extract_text_from_pdf(pdf_path)
        sections = self.split_into_sections(text)
        documents = []
        filename = pdf_path.split('/')[-1]
        for i, section in enumerate(sections):
            metadata = {
                "source": filename,
                "page": i,
                "section": section.split(":", 1)[0] if ":" in section else "General"
            }
            documents.append(Document(page_content=section, metadata=metadata))
        return documents 