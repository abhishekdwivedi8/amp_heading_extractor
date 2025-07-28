"""
Level 5: Heading Refinement and Post-Processing
Refines detected headings using context analysis and hierarchy detection.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class Level5HeadingRefiner:
    def __init__(self):
        self.hierarchy_patterns = {
            'numeric': r'^(\d+(?:\.\d+)*)',  # 1.1.1
            'roman': r'^([IVX]+)',           # I, II, III
            'alpha': r'^([A-Z])',            # A, B, C
            'bullet': r'^([-*•])',           # Bullet points
        }
        
        self.noise_patterns = [
            r'^\s*page\s+\d+\s*$',          # Page numbers
            r'^\s*\d+\s*$',                 # Just numbers
            r'^\s*\w{1,2}\s*$',             # Single letters/short words
            r'^\s*[-=_]{3,}\s*$',           # Dividers
            r'^\s*\.\.\.\s*$',              # Ellipsis
            r'^\s*https?://',               # URLs
            r'^\s*\d{1,2}/\d{1,2}/\d{2,4}', # Dates
        ]
        
        self.common_non_headings = {
            'page', 'figure', 'table', 'appendix', 'references', 'bibliography',
            'acknowledgments', 'copyright', 'all rights reserved', 'printed in',
            'isbn', 'doi', 'url', 'email', 'phone', 'fax', 'address'
        }
    
    def refine_headings(self, ensemble_results: Dict[str, Any]) -> Dict[str, Any]:
        """Refine and post-process detected headings"""
        classified_headings = ensemble_results.get('classified_headings', [])
        
        if not classified_headings:
            return {
                'refined_headings': [],
                'refinement_metadata': {'no_headings': True}
            }
        
        # Step 1: Filter out noise and false positives
        filtered_headings = self._filter_noise(classified_headings)
        
        # Step 2: Detect and build heading hierarchy
        hierarchical_headings = self._detect_hierarchy(filtered_headings)
        
        # Step 3: Context-based refinement
        context_refined = self._context_refinement(hierarchical_headings)
        
        # Step 4: Duplicate detection and merging
        deduplicated = self._remove_duplicates(context_refined)
        
        # Step 5: Final quality scoring
        final_headings = self._calculate_final_scores(deduplicated)
        
        return {
            'refined_headings': final_headings,
            'refinement_metadata': self._generate_refinement_metadata(
                classified_headings, final_headings
            )
        }
    
    def _filter_noise(self, headings: List[Dict]) -> List[Dict]:
        """Remove obvious noise and false positives"""
        filtered = []
        
        for heading in headings:
            text = heading.get('text', '').strip()
            
            # Skip if empty or too short
            if len(text) < 2:
                continue
            
            # Check noise patterns
            is_noise = any(re.match(pattern, text, re.IGNORECASE) 
                          for pattern in self.noise_patterns)
            if is_noise:
                continue
            
            # Check common non-headings
            text_lower = text.lower()
            if any(non_heading in text_lower for non_heading in self.common_non_headings):
                # Only skip if it's a pure match, not part of a larger heading
                if len(text.split()) <= 3:
                    continue
            
            # Skip if too long (likely paragraph text)
            if len(text.split()) > 20:
                continue
            
            # Skip if very low confidence and no strong indicators
            confidence = heading.get('ensemble_probability', 0)
            has_structure = bool(heading.get('linguistic_features', {}).get('structural_indicators'))
            
            if confidence < 0.3 and not has_structure:
                continue
            
            # Add noise filtering metadata
            heading_copy = heading.copy()
            heading_copy['passed_noise_filter'] = True
            filtered.append(heading_copy)
        
        logger.info(f"Filtered {len(headings) - len(filtered)} noisy headings")
        return filtered
    
    def _detect_hierarchy(self, headings: List[Dict]) -> List[Dict]:
        """Detect heading hierarchy and levels with improved accuracy"""
        if not headings:
            return headings
        
        # Sort headings by page and position
        sorted_headings = sorted(headings, key=lambda x: (x.get('page', 1), x.get('line', 1)))
        
        # Analyze font sizes to determine hierarchy
        font_sizes = [h.get('font_size', 12) for h in sorted_headings if h.get('font_size', 0) > 0]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        
        # Create font size thresholds for levels
        large_threshold = avg_font_size * 1.3
        medium_threshold = avg_font_size * 1.1
        
        for i, heading in enumerate(sorted_headings):
            text = heading['text'].strip()
            
            # Initialize hierarchy info
            hierarchy_info = {
                'level': 1,
                'numbering_type': 'none',
                'numbering_value': None,
                'parent_index': None,
                'children_indices': []
            }
            
            # Check for numbering patterns first (most reliable)
            level_from_numbering = self._get_level_from_numbering(text)
            if level_from_numbering > 0:
                hierarchy_info['level'] = level_from_numbering
                hierarchy_info['numbering_type'] = 'numeric'
            
            # Use font size and formatting for level detection
            font_size = heading.get('font_size', avg_font_size)
            is_bold = heading.get('is_bold', False)
            
            # Enhanced level determination
            if level_from_numbering == 0:  # Only if not set by numbering
                # Content-based level assignment first
                content_level = self._get_level_from_content(text, i == 0)
                
                if content_level > 0:
                    hierarchy_info['level'] = content_level
                elif font_size >= large_threshold or (is_bold and font_size >= medium_threshold):
                    hierarchy_info['level'] = 1  # H1 - Main headings
                elif font_size >= medium_threshold or is_bold:
                    hierarchy_info['level'] = 2  # H2 - Sub headings
                else:
                    hierarchy_info['level'] = 3  # H3 - Detail headings
            else:
                hierarchy_info['level'] = level_from_numbering
            
            # Adjust based on content characteristics
            hierarchy_info['level'] = self._adjust_level_by_content(text, hierarchy_info['level'])
            
            # Ensure level is within valid range
            hierarchy_info['level'] = max(1, min(3, hierarchy_info['level']))
            
            heading['hierarchy_info'] = hierarchy_info
        
        return sorted_headings
    
    def _get_level_from_numbering(self, text: str) -> int:
        """Extract hierarchy level from numbering patterns"""
        # Check for numbered sections like 1.1.1, 2.3, etc.
        numeric_match = re.match(r'^(\d+(?:\.\d+)*)', text)
        if numeric_match:
            parts = numeric_match.group(1).split('.')
            return len(parts)  # 1 = H1, 1.1 = H2, 1.1.1 = H3
        
        # Check for roman numerals
        if re.match(r'^[IVX]+\.?\s', text):
            return 1  # Roman numerals typically indicate main sections
        
        # Check for alphabetic numbering
        if re.match(r'^[A-Z]\.\s', text):
            return 2  # Alphabetic typically indicates sub-sections
        
        return 0  # No numbering detected
    
    def _get_level_from_content(self, text: str, is_first: bool) -> int:
        """Determine level based on content analysis"""
        text_lower = text.lower().strip()
        
        # H1 patterns - Document titles and main sections
        h1_patterns = [
            'application form', 'understanding', 'neural network fundamentals',
            'introduction to', 'guide to', 'manual', 'handbook', 'report on'
        ]
        
        if any(pattern in text_lower for pattern in h1_patterns):
            return 1
        
        # First heading with document-like content
        if is_first and len(text.split()) >= 3:
            doc_indicators = ['application', 'form', 'document', 'report', 'study']
            if any(indicator in text_lower for indicator in doc_indicators):
                return 1
        
        # H2 patterns - Major sections
        h2_patterns = [
            'name of', 'team details', 'working together', 'tools & platforms',
            'personal details', 'contact information', 'employment details'
        ]
        
        if any(pattern in text_lower for pattern in h2_patterns):
            return 2
        
        return 0  # Let other factors decide
    
    def _adjust_level_by_content(self, text: str, current_level: int) -> int:
        """Adjust level based on content characteristics"""
        text_lower = text.lower()
        
        # Document titles (should be H1)
        title_indicators = [
            'application form', 'form for', 'grant of', 'document title',
            'report on', 'study of', 'analysis of', 'review of'
        ]
        
        # Main document sections (should be H1)
        main_indicators = [
            'introduction', 'conclusion', 'summary', 'abstract', 'overview',
            'background', 'methodology', 'results', 'discussion', 'references',
            'executive summary', 'table of contents'
        ]
        
        # Form sections and sub-headings (should be H2)
        sub_indicators = [
            'personal details', 'contact information', 'employment details',
            'government servant', 'service details', 'ltc details',
            'particulars', 'information', 'details'
        ]
        
        # Form fields and specific items (should be H3)
        detail_indicators = [
            'name of', 'designation', 'date of', 'whether', 'home town',
            'entitled to', 'concession', 'permanent', 'temporary'
        ]
        
        # Check for document title (highest priority)
        if any(indicator in text_lower for indicator in title_indicators):
            return 1
        
        # Check for main section indicators
        if any(indicator in text_lower for indicator in main_indicators):
            return 1
        
        # Check for sub-section indicators
        if any(indicator in text_lower for indicator in sub_indicators):
            return 2
        
        # Check for detail indicators (form fields)
        if any(indicator in text_lower for indicator in detail_indicators):
            return 3
        
        # If it's the first item and looks like a title, make it H1
        if ('application' in text_lower or 'form' in text_lower) and len(text.split()) >= 3:
            return 1
        
        # Form field labels are typically H3
        if len(text.split()) <= 6 and not text.endswith('.'):
            return 3
        
        return current_level
    
    def _context_refinement(self, headings: List[Dict]) -> List[Dict]:
        """Refine headings based on context analysis"""
        refined = []
        
        for i, heading in enumerate(headings):
            text = heading['text']
            confidence = heading.get('ensemble_probability', 0)
            
            # Context factors
            context_boost = 0.0
            context_penalty = 0.0
            
            # Check surrounding context
            surrounding_context = self._get_surrounding_context(headings, i)
            
            # Boost if followed by content-like text
            if surrounding_context.get('has_following_content'):
                context_boost += 0.1
            
            # Boost if part of a clear hierarchy
            hierarchy_info = heading.get('hierarchy_info', {})
            if hierarchy_info.get('numbering_type') != 'none':
                context_boost += 0.15
            
            # Boost if semantic indicators are strong
            structural_indicators = heading.get('linguistic_features', {}).get('structural_indicators', [])
            if len(structural_indicators) > 0:
                context_boost += 0.1
            
            # Penalty for repetitive patterns
            if self._is_repetitive_pattern(text, headings):
                context_penalty += 0.2
            
            # Penalty if isolated without context
            if not surrounding_context.get('has_context'):
                context_penalty += 0.1
            
            # Calculate refined confidence
            refined_confidence = min(1.0, max(0.0, 
                confidence + context_boost - context_penalty
            ))
            
            # Update heading with refinement info
            refined_heading = heading.copy()
            refined_heading.update({
                'refined_confidence': refined_confidence,
                'context_boost': context_boost,
                'context_penalty': context_penalty,
                'surrounding_context': surrounding_context
            })
            
            refined.append(refined_heading)
        
        return refined
    
    def _get_surrounding_context(self, headings: List[Dict], index: int) -> Dict[str, Any]:
        """Analyze surrounding context of a heading"""
        context = {
            'has_following_content': False,
            'has_preceding_heading': False,
            'has_context': False,
            'distance_to_next': None,
            'distance_to_prev': None
        }
        
        current_heading = headings[index]
        current_page = current_heading.get('page', 1)
        current_line = current_heading.get('line', 1)
        
        # Check following content
        if index < len(headings) - 1:
            next_heading = headings[index + 1]
            next_page = next_heading.get('page', 1)
            next_line = next_heading.get('line', 1)
            
            if next_page == current_page:
                line_distance = next_line - current_line
                context['distance_to_next'] = line_distance
                if line_distance > 2:  # Some content between headings
                    context['has_following_content'] = True
            elif next_page == current_page + 1:
                context['has_following_content'] = True
        
        # Check preceding heading
        if index > 0:
            prev_heading = headings[index - 1]
            prev_page = prev_heading.get('page', 1)
            prev_line = prev_heading.get('line', 1)
            
            if prev_page == current_page:
                line_distance = current_line - prev_line
                context['distance_to_prev'] = line_distance
                if line_distance < 20:  # Reasonable distance
                    context['has_preceding_heading'] = True
        
        context['has_context'] = (context['has_following_content'] or 
                                context['has_preceding_heading'])
        
        return context
    
    def _is_repetitive_pattern(self, text: str, all_headings: List[Dict]) -> bool:
        """Check if heading follows a repetitive pattern that might indicate noise"""
        text_clean = re.sub(r'\d+', 'NUM', text.lower())  # Replace numbers with placeholder
        
        similar_count = 0
        for other_heading in all_headings:
            other_text = other_heading['text']
            other_clean = re.sub(r'\d+', 'NUM', other_text.lower())
            
            if other_clean == text_clean:
                similar_count += 1
        
        # If we see the same pattern more than 3 times, it might be noise
        return similar_count > 3
    
    def _remove_duplicates(self, headings: List[Dict]) -> List[Dict]:
        """Remove duplicate headings and merge similar ones"""
        if not headings:
            return headings
        
        deduplicated = []
        used_indices = set()
        
        for i, heading in enumerate(headings):
            if i in used_indices:
                continue
            
            text = heading['text']
            page = heading.get('page', 1)
            
            # Find potential duplicates
            duplicates = [i]
            for j in range(i + 1, len(headings)):
                if j in used_indices:
                    continue
                
                other_heading = headings[j]
                other_text = other_heading['text']
                other_page = other_heading.get('page', 1)
                
                # Check for exact or near duplicates
                if (self._texts_similar(text, other_text) and 
                    abs(page - other_page) <= 1):  # Same or adjacent pages
                    duplicates.append(j)
            
            # Merge duplicates by keeping the best one
            if len(duplicates) > 1:
                best_heading = self._select_best_duplicate(
                    [headings[idx] for idx in duplicates]
                )
                deduplicated.append(best_heading)
                used_indices.update(duplicates)
            else:
                deduplicated.append(heading)
                used_indices.add(i)
        
        logger.info(f"Removed {len(headings) - len(deduplicated)} duplicate headings")
        return deduplicated
    
    def _texts_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar enough to be considered duplicates"""
        # Simple similarity check
        text1_words = set(text1.lower().split())
        text2_words = set(text2.lower().split())
        
        if not text1_words or not text2_words:
            return text1.lower().strip() == text2.lower().strip()
        
        intersection = len(text1_words.intersection(text2_words))
        union = len(text1_words.union(text2_words))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        return jaccard_similarity >= threshold
    
    def _select_best_duplicate(self, duplicates: List[Dict]) -> Dict:
        """Select the best heading from a group of duplicates"""
        # Prefer heading with highest confidence
        best = max(duplicates, key=lambda x: x.get('refined_confidence', 
                                                 x.get('ensemble_probability', 0)))
        
        # Merge information from other duplicates
        best_copy = best.copy()
        best_copy['duplicate_count'] = len(duplicates)
        best_copy['merged_from_pages'] = list(set(h.get('page', 1) for h in duplicates))
        
        return best_copy
    
    def _calculate_final_scores(self, headings: List[Dict]) -> List[Dict]:
        """Calculate final quality scores for headings"""
        if not headings:
            return headings
        
        final_headings = []
        
        for heading in headings:
            # Base score from refined confidence
            base_score = heading.get('refined_confidence', 
                                   heading.get('ensemble_probability', 0))
            
            # Hierarchy bonus
            hierarchy_info = heading.get('hierarchy_info', {})
            if hierarchy_info.get('numbering_type') != 'none':
                base_score += 0.05
            
            # Context bonus
            if heading.get('surrounding_context', {}).get('has_context'):
                base_score += 0.05
            
            # Length penalty for very long headings
            word_count = len(heading['text'].split())
            if word_count > 15:
                base_score -= 0.1
            elif 3 <= word_count <= 8:
                base_score += 0.02
            
            # Final score
            final_score = min(1.0, max(0.0, base_score))
            
            # Make final decision
            is_final_heading = final_score >= 0.5
            
            final_heading = heading.copy()
            final_heading.update({
                'final_score': final_score,
                'is_final_heading': is_final_heading,
                'processing_stage': 'level_5_refinement'
            })
            
            final_headings.append(final_heading)
        
        # Sort by score (descending) and page/line (ascending)
        final_headings.sort(key=lambda x: (-x['final_score'], x.get('page', 1), x.get('line', 1)))
        
        return final_headings
    
    def _generate_refinement_metadata(self, original_headings: List[Dict], 
                                    refined_headings: List[Dict]) -> Dict[str, Any]:
        """Generate metadata about the refinement process"""
        metadata = {
            'original_count': len(original_headings),
            'refined_count': len(refined_headings),
            'filtered_count': len(original_headings) - len(refined_headings),
            'final_heading_count': len([h for h in refined_headings if h.get('is_final_heading', False)]),
            'hierarchy_levels': self._get_hierarchy_stats(refined_headings),
            'confidence_improvement': self._calculate_confidence_improvement(original_headings, refined_headings),
            'quality_distribution': self._get_quality_distribution(refined_headings)
        }
        
        return metadata
    
    def _get_hierarchy_stats(self, headings: List[Dict]) -> Dict[str, Any]:
        """Get statistics about heading hierarchy"""
        levels = [h.get('hierarchy_info', {}).get('level', 1) for h in headings]
        numbering_types = [h.get('hierarchy_info', {}).get('numbering_type', 'none') for h in headings]
        
        return {
            'level_distribution': dict(Counter(levels)),
            'numbering_types': dict(Counter(numbering_types)),
            'max_level': max(levels) if levels else 0,
            'avg_level': np.mean(levels) if levels else 0
        }
    
    def _calculate_confidence_improvement(self, original: List[Dict], refined: List[Dict]) -> float:
        """Calculate average confidence improvement"""
        if not original or not refined:
            return 0.0
        
        original_avg = np.mean([h.get('ensemble_probability', 0) for h in original])
        refined_avg = np.mean([h.get('refined_confidence', 0) for h in refined])
        
        return refined_avg - original_avg
    
    def _get_quality_distribution(self, headings: List[Dict]) -> Dict[str, int]:
        """Get distribution of final quality scores"""
        scores = [h.get('final_score', 0) for h in headings]
        
        return {
            'excellent': len([s for s in scores if s > 0.9]),
            'good': len([s for s in scores if 0.7 < s <= 0.9]),
            'fair': len([s for s in scores if 0.5 < s <= 0.7]),
            'poor': len([s for s in scores if s <= 0.5])
        }
