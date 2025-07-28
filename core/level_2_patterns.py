"""
Level 2: Pattern-Based Heading Detection
Uses regex patterns and text formatting rules to identify potential headings.
"""
import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class Level2PatternDetector:
    def __init__(self):
        self.heading_patterns = {
            'numbered_sections': [
                r'^\s*(\d+\.)+\s+(.+)$',  # 1.1 Section Title
                r'^\s*(\d+)\s+(.+)$',     # 1 Section Title
                r'^\s*([IVX]+\.?)\s+(.+)$',  # I. Roman numerals
                r'^\s*([A-Z]\.?)\s+(.+)$'    # A. Letter sections
            ],
            'formatted_headings': [
                r'^[A-Z][A-Z\s]+$',       # ALL CAPS
                r'^[A-Z][a-z\s]+:$',      # Title Case with colon
                r'^\*\*(.+)\*\*$',        # **Bold markdown**
                r'^_(.+)_$',              # _Italic markdown_
            ],
            'outline_patterns': [
                r'^\s*•\s+(.+)$',         # Bullet points
                r'^\s*-\s+(.+)$',         # Dash points
                r'^\s*\*\s+(.+)$',        # Asterisk points
            ],
            'document_structure': [
                r'^(abstract|introduction|methodology|results|conclusion|references)$',
                r'^(chapter|section|subsection)\s+\d+',
                r'^(table of contents|appendix|bibliography)',
                r'^(summary|overview|background|discussion)'
            ]
        }
        
        self.formatting_indicators = {
            'font_size_ratio': 1.2,  # Heading should be larger
            'font_weight_keywords': ['bold', 'heavy', 'black'],
            'isolation_factor': 2,    # Lines before/after
        }
    
    def detect_headings(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Detect headings using pattern matching"""
        pages = extraction_result.get('merged_text', [])
        raw_extractions = extraction_result.get('raw_extractions', {})
        
        detected_headings = []
        
        for page_num, page_text in enumerate(pages):
            if not page_text:
                continue
                
            lines = page_text.split('\n')
            
            # Pattern-based detection
            pattern_headings = self._detect_pattern_headings(lines, page_num)
            
            # Format-based detection (if we have formatting info)
            format_headings = self._detect_format_headings(
                raw_extractions, page_num, lines
            )
            
            # Combine and deduplicate
            page_headings = self._merge_heading_candidates(
                pattern_headings, format_headings, lines
            )
            
            detected_headings.extend(page_headings)
        
        return {
            'detected_headings': detected_headings,
            'detection_metadata': self._generate_detection_metadata(detected_headings),
            'pattern_statistics': self._get_pattern_statistics(detected_headings)
        }
    
    def _detect_pattern_headings(self, lines: List[str], page_num: int) -> List[Dict]:
        """Detect headings based on text patterns"""
        headings = []
        
        for line_num, line in enumerate(lines):
            if not line.strip():
                continue
            
            for pattern_type, patterns in self.heading_patterns.items():
                for pattern in patterns:
                    match = re.match(pattern, line.strip(), re.IGNORECASE)
                    if match:
                        confidence = self._calculate_pattern_confidence(
                            pattern_type, line, lines, line_num
                        )
                        
                        heading = {
                            'text': line.strip(),
                            'page': page_num + 1,
                            'line': line_num + 1,
                            'pattern_type': pattern_type,
                            'pattern': pattern,
                            'confidence': confidence,
                            'detection_method': 'pattern',
                            'groups': match.groups() if match.groups() else []
                        }
                        headings.append(heading)
                        break
        
        return headings
    
    def _detect_format_headings(self, raw_extractions: Dict, page_num: int, lines: List[str]) -> List[Dict]:
        """Detect headings based on formatting (from PyMuPDF data)"""
        headings = []
        
        if 'pymupdf' not in raw_extractions:
            return headings
        
        pymupdf_pages = raw_extractions['pymupdf']
        if page_num >= len(pymupdf_pages):
            return headings
        
        page_data = pymupdf_pages[page_num]
        blocks = page_data.get('blocks', [])
        
        # Calculate average font size for comparison
        font_sizes = [block['size'] for block in blocks if block.get('size')]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        
        for block in blocks:
            text = block.get('text', '').strip()
            if not text or len(text) < 3:
                continue
            
            font_size = block.get('size', avg_font_size)
            font_flags = block.get('flags', 0)
            
            # Check formatting indicators
            is_bold = font_flags & 2**4  # Bold flag
            is_larger = font_size > avg_font_size * self.formatting_indicators['font_size_ratio']
            is_isolated = self._check_text_isolation(text, lines)
            
            if (is_bold or is_larger) and len(text.split()) <= 10:  # Reasonable heading length
                confidence = self._calculate_format_confidence(
                    font_size, avg_font_size, is_bold, is_isolated, text
                )
                
                heading = {
                    'text': text,
                    'page': page_num + 1,
                    'font_size': font_size,
                    'avg_font_size': avg_font_size,
                    'is_bold': is_bold,
                    'is_isolated': is_isolated,
                    'confidence': confidence,
                    'detection_method': 'format',
                    'bbox': block.get('bbox', [])
                }
                headings.append(heading)
        
        return headings
    
    def _calculate_pattern_confidence(self, pattern_type: str, line: str, 
                                    lines: List[str], line_num: int) -> float:
        """Calculate confidence score for pattern-based detection"""
        base_confidence = {
            'numbered_sections': 0.9,
            'formatted_headings': 0.7,
            'outline_patterns': 0.6,
            'document_structure': 0.8
        }.get(pattern_type, 0.5)
        
        # Adjust based on context
        modifiers = 0.0
        
        # Check if line is isolated (empty lines before/after)
        if self._check_line_isolation(lines, line_num):
            modifiers += 0.1
        
        # Check length (headings shouldn't be too long)
        word_count = len(line.split())
        if 2 <= word_count <= 8:
            modifiers += 0.1
        elif word_count > 15:
            modifiers -= 0.2
        
        # Check for ending punctuation (headings usually don't end with periods)
        if not line.rstrip().endswith('.'):
            modifiers += 0.05
        
        return min(1.0, base_confidence + modifiers)
    
    def _calculate_format_confidence(self, font_size: float, avg_font_size: float,
                                   is_bold: bool, is_isolated: bool, text: str) -> float:
        """Calculate confidence for format-based detection"""
        confidence = 0.5
        
        # Font size factor
        size_ratio = font_size / avg_font_size
        if size_ratio > 1.5:
            confidence += 0.3
        elif size_ratio > 1.2:
            confidence += 0.2
        
        # Bold text
        if is_bold:
            confidence += 0.2
        
        # Text isolation
        if is_isolated:
            confidence += 0.15
        
        # Text characteristics
        word_count = len(text.split())
        if 2 <= word_count <= 10:
            confidence += 0.1
        
        # Title case check
        if text.istitle():
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _check_line_isolation(self, lines: List[str], line_num: int) -> bool:
        """Check if line is isolated by empty lines"""
        before_empty = (line_num == 0 or 
                       line_num > 0 and not lines[line_num - 1].strip())
        after_empty = (line_num == len(lines) - 1 or 
                      line_num < len(lines) - 1 and not lines[line_num + 1].strip())
        return before_empty or after_empty
    
    def _check_text_isolation(self, text: str, lines: List[str]) -> bool:
        """Check if text appears isolated in the line list"""
        for i, line in enumerate(lines):
            if text in line:
                return self._check_line_isolation(lines, i)
        return False
    
    def _merge_heading_candidates(self, pattern_headings: List[Dict], 
                                format_headings: List[Dict], lines: List[str]) -> List[Dict]:
        """Merge and deduplicate heading candidates"""
        all_headings = pattern_headings + format_headings
        
        # Remove duplicates based on text similarity
        merged = []
        for heading in all_headings:
            is_duplicate = False
            for existing in merged:
                if self._texts_similar(heading['text'], existing['text']):
                    # Keep the one with higher confidence
                    if heading['confidence'] > existing['confidence']:
                        merged.remove(existing)
                        merged.append(heading)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(heading)
        
        return sorted(merged, key=lambda x: (x['page'], x.get('line', 0)))
    
    def _texts_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar"""
        # Simple similarity check - can be enhanced with more sophisticated methods
        text1_clean = re.sub(r'[^\w\s]', '', text1.lower())
        text2_clean = re.sub(r'[^\w\s]', '', text2.lower())
        
        if text1_clean == text2_clean:
            return True
        
        # Check if one is substring of another
        if text1_clean in text2_clean or text2_clean in text1_clean:
            return True
        
        return False
    
    def _generate_detection_metadata(self, headings: List[Dict]) -> Dict:
        """Generate metadata about detection results"""
        return {
            'total_headings': len(headings),
            'headings_by_method': {
                'pattern': len([h for h in headings if h['detection_method'] == 'pattern']),
                'format': len([h for h in headings if h['detection_method'] == 'format'])
            },
            'average_confidence': sum(h['confidence'] for h in headings) / len(headings) if headings else 0,
            'confidence_distribution': self._get_confidence_distribution(headings)
        }
    
    def _get_confidence_distribution(self, headings: List[Dict]) -> Dict:
        """Get distribution of confidence scores"""
        high = len([h for h in headings if h['confidence'] > 0.8])
        medium = len([h for h in headings if 0.5 < h['confidence'] <= 0.8])
        low = len([h for h in headings if h['confidence'] <= 0.5])
        
        return {'high': high, 'medium': medium, 'low': low}
    
    def _get_pattern_statistics(self, headings: List[Dict]) -> Dict:
        """Get statistics about pattern usage"""
        pattern_counts = {}
        for heading in headings:
            if heading['detection_method'] == 'pattern':
                pattern_type = heading['pattern_type']
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return pattern_counts
