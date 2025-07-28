"""
Level 1: Basic Text Extraction from PDF
Initial extraction using multiple PDF parsing libraries for comprehensive coverage.
"""
import fitz  # PyMuPDF
import pdfplumber
import PyPDF2
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Level1Extractor:
    def __init__(self):
        self.extractors = {
            'pymupdf': self._extract_with_pymupdf,
            'pdfplumber': self._extract_with_pdfplumber,
            'pypdf2': self._extract_with_pypdf2
        }
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using multiple PDF libraries"""
        results = {}
        
        for extractor_name, extractor_func in self.extractors.items():
            try:
                results[extractor_name] = extractor_func(pdf_path)
                logger.info(f"Successfully extracted with {extractor_name}")
            except Exception as e:
                logger.warning(f"Failed to extract with {extractor_name}: {e}")
                results[extractor_name] = []
        
        # Merge and deduplicate results
        merged_text = self._merge_extractions(results)
        
        return {
            'raw_extractions': results,
            'merged_text': merged_text,
            'extraction_metadata': self._get_extraction_metadata(results)
        }
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[Dict]:
        """Extract using PyMuPDF with error suppression"""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    blocks = page.get_text("dict")["blocks"]
                    
                    page_data = {
                        'page_number': page_num + 1,
                        'blocks': []
                    }
                    
                    for block in blocks:
                        if 'lines' in block:
                            for line in block['lines']:
                                for span in line['spans']:
                                    if span.get('text', '').strip():
                                        page_data['blocks'].append({
                                            'text': span['text'],
                                            'bbox': span.get('bbox', [0,0,0,0]),
                                            'font': span.get('font', 'default'),
                                            'size': span.get('size', 12),
                                            'flags': span.get('flags', 0)
                                        })
                    
                    pages.append(page_data)
                except Exception as e:
                    # Skip problematic pages but continue processing
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            return pages
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """Extract using pdfplumber with table awareness"""
        pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_data = {
                    'page_number': page_num + 1,
                    'text': page.extract_text(),
                    'chars': page.chars,
                    'lines': page.lines if hasattr(page, 'lines') else [],
                    'tables': page.extract_tables()
                }
                pages.append(page_data)
        
        return pages
    
    def _extract_with_pypdf2(self, pdf_path: str) -> List[Dict]:
        """Extract using PyPDF2 as fallback"""
        pages = []
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                pages.append({
                    'page_number': page_num + 1,
                    'text': text
                })
        
        return pages
    
    def _merge_extractions(self, results: Dict) -> List[str]:
        """Merge extractions from different libraries"""
        # Prioritize PyMuPDF results, fallback to others
        if results.get('pymupdf'):
            return self._extract_text_from_pymupdf(results['pymupdf'])
        elif results.get('pdfplumber'):
            return [page['text'] for page in results['pdfplumber'] if page['text']]
        elif results.get('pypdf2'):
            return [page['text'] for page in results['pypdf2'] if page['text']]
        else:
            return []
    
    def _extract_text_from_pymupdf(self, pages: List[Dict]) -> List[str]:
        """Extract clean text from PyMuPDF blocks"""
        page_texts = []
        
        for page in pages:
            text_blocks = []
            for block in page['blocks']:
                if block['text'].strip():
                    text_blocks.append(block['text'])
            page_texts.append('\n'.join(text_blocks))
        
        return page_texts
    
    def _get_extraction_metadata(self, results: Dict) -> Dict:
        """Generate metadata about extraction quality"""
        metadata = {
            'successful_extractors': [name for name, data in results.items() if data],
            'total_pages': 0,
            'extraction_quality_score': 0.0
        }
        
        # Get page count from best available source
        for extractor in ['pymupdf', 'pdfplumber', 'pypdf2']:
            if results.get(extractor):
                metadata['total_pages'] = len(results[extractor])
                break
        
        # Calculate quality score based on successful extractions
        quality_weights = {'pymupdf': 0.5, 'pdfplumber': 0.3, 'pypdf2': 0.2}
        for extractor in metadata['successful_extractors']:
            metadata['extraction_quality_score'] += quality_weights.get(extractor, 0)
        
        return metadata
