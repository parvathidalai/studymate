import fitz  # PyMuPDF
import os
from typing import List, Tuple

class PDFProcessor:
    def __init__(self):
        self.chunk_size = 500
        self.overlap = 100
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def create_chunks(self, text: str, filename: str) -> List[dict]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_text = " ".join(words[i:i + self.chunk_size])
            chunks.append({
                "text": chunk_text,
                "source": filename,
                "chunk_id": len(chunks)
            })
        
        return chunks
    
    def process_multiple_pdfs(self, uploaded_files) -> List[dict]:
        """Process multiple PDF files and return combined chunks"""
        all_chunks = []
        
        for file in uploaded_files:
            try:
                text = self.extract_text_from_pdf(file)
                chunks = self.create_chunks(text, file.name)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")
                continue
        
        return all_chunks