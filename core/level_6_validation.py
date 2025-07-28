"""
Level 6: Heading Validation and Quality Assurance
Validates detected headings using multiple validation strategies and quality metrics.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
import statistics
import logging

logger = logging.getLogger(__name__)

class Level6HeadingValidator:
    def __init__(self, validation_config: Optional[Dict] = None):
        """Initialize heading validator with configuration"""
        self.config = validation_config or self._get_default_config()
        
        # Document structure patterns
        self.document_patterns = {
            'academic': [
                'abstract', 'introduction', 'methodology', 'methods', 'results',
                'discussion', 'conclusion', 'references', 'bibliography'
            ],
            'technical': [
                'overview', 'architecture', 'implementation', 'design',
                'testing', 'deployment', 'maintenance', 'api'
            ],
            'business': [
                'executive summary', 'background', 'analysis', 'recommendations',
                'financial', 'market', 'strategy', 'risk'
            ]
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_confidence': 0.4,
            'min_hierarchy_consistency': 0.6,
            'max_heading_length': 150,
            'min_heading_length': 2,
            'max_headings_per_page': 10
        }
    
    def validate_headings(self, refinement_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and quality-check refined headings"""
        refined_headings = refinement_results.get('refined_headings', [])
        
        if not refined_headings:
            return {
                'validated_headings': [],
                'validation_metadata': {'no_headings': True},
                'quality_scores': {}
            }
        
        # Step 1: Individual heading validation
        individually_validated = self._validate_individual_headings(refined_headings)
        
        # Step 2: Document-level validation
        document_validated = self._validate_document_structure(individually_validated)
        
        # Step 3: Hierarchy validation
        hierarchy_validated = self._validate_hierarchy_consistency(document_validated)
        
        # Step 4: Cross-validation checks
        cross_validated = self._cross_validate_headings(hierarchy_validated)
        
        # Step 5: Final quality assessment
        final_validated, quality_scores = self._assess_final_quality(cross_validated)
        
        return {
            'validated_headings': final_validated,
            'validation_metadata': self._generate_validation_metadata(
                refined_headings, final_validated
            ),
            'quality_scores': quality_scores
        }
    
    def _validate_individual_headings(self, headings: List[Dict]) -> List[Dict]:
        """Validate each heading individually"""
        validated = []
        
        for heading in headings:
            validation_results = {
                'length_valid': True,
                'format_valid': True,
                'content_valid': True,
                'confidence_valid': True,
                'validation_score': 1.0,
                'validation_issues': []
            }
            
            text = heading.get('text', '').strip()
            confidence = heading.get('final_score', 0)
            
            # Length validation
            if len(text) < self.quality_thresholds['min_heading_length']:
                validation_results['length_valid'] = False
                validation_results['validation_issues'].append('too_short')
                validation_results['validation_score'] -= 0.3
            
            if len(text) > self.quality_thresholds['max_heading_length']:
                validation_results['length_valid'] = False
                validation_results['validation_issues'].append('too_long')
                validation_results['validation_score'] -= 0.2
            
            # Format validation
            format_issues = self._check_format_issues(text)
            if format_issues:
                validation_results['format_valid'] = False
                validation_results['validation_issues'].extend(format_issues)
                validation_results['validation_score'] -= len(format_issues) * 0.1
            
            # Content validation
            content_issues = self._check_content_issues(text)
            if content_issues:
                validation_results['content_valid'] = False
                validation_results['validation_issues'].extend(content_issues)
                validation_results['validation_score'] -= len(content_issues) * 0.15
            
            # Confidence validation
            if confidence < self.quality_thresholds['min_confidence']:
                validation_results['confidence_valid'] = False
                validation_results['validation_issues'].append('low_confidence')
                validation_results['validation_score'] -= 0.2
            
            # Ensure score doesn't go below 0
            validation_results['validation_score'] = max(0.0, validation_results['validation_score'])
            
            # Add validation results to heading
            validated_heading = heading.copy()
            validated_heading['individual_validation'] = validation_results
            validated.append(validated_heading)
        
        return validated
    
    def _check_format_issues(self, text: str) -> List[str]:
        """Check for format-related issues"""
        issues = []
        
        # Check for excessive punctuation
        punct_count = len(re.findall(r'[!@#$%^&*()_+=\[\]{};:"|<>?,./]', text))
        if punct_count > len(text) * 0.3:
            issues.append('excessive_punctuation')
        
        # Check for all caps (unless it's an acronym)
        if text.isupper() and len(text.split()) > 1:
            issues.append('all_caps')
        
        # Check for excessive whitespace
        if '  ' in text or text.startswith(' ') or text.endswith(' '):
            issues.append('whitespace_issues')
        
        # Check for repeated characters
        if re.search(r'(.)\1{3,}', text):
            issues.append('repeated_characters')
        
        # Check for suspicious patterns
        if re.search(r'\d{3,}', text) and not re.search(r'^(chapter|section)\s+\d+', text, re.I):
            issues.append('suspicious_numbers')
        
        return issues
    
    def _check_content_issues(self, text: str) -> List[str]:
        """Check for content-related issues"""
        issues = []
        
        # Check for non-heading content patterns
        non_heading_patterns = [
            r'page \d+',
            r'figure \d+',
            r'table \d+',
            r'\d+ of \d+',
            r'continued on',
            r'see page',
            r'as shown in',
            r'according to',
            r'for example',
            r'in conclusion',
            r'this chapter',
            r'the following'
        ]
        
        text_lower = text.lower()
        for pattern in non_heading_patterns:
            if re.search(pattern, text_lower):
                issues.append('non_heading_content')
                break
        
        # Check for incomplete sentences
        if text.endswith(',') or text.endswith(';'):
            issues.append('incomplete_sentence')
        
        # Check for question format (usually not headings)
        if text.strip().endswith('?') and len(text.split()) > 3:
            issues.append('question_format')
        
        # Check for narrative language
        narrative_indicators = ['i ', 'we ', 'you ', 'they ', 'he ', 'she ']
        if any(text_lower.startswith(indicator) for indicator in narrative_indicators):
            issues.append('narrative_language')
        
        return issues
    
    def _validate_document_structure(self, headings: List[Dict]) -> List[Dict]:
        """Validate headings against expected document structure"""
        if not headings:
            return headings
        
        # Extract heading texts for analysis
        heading_texts = [h['text'].lower() for h in headings]
        
        # Determine document type
        document_type = self._identify_document_type(heading_texts)
        
        # Check structural consistency
        structure_score = self._calculate_structure_score(heading_texts, document_type)
        
        # Update headings with document validation info
        for heading in headings:
            heading['document_validation'] = {
                'document_type': document_type,
                'structure_score': structure_score,
                'fits_document_pattern': self._heading_fits_pattern(
                    heading['text'], document_type
                )
            }
        
        return headings
    
    def _identify_document_type(self, heading_texts: List[str]) -> str:
        """Identify the type of document based on heading patterns"""
        type_scores = {}
        
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            for heading in heading_texts:
                for pattern in patterns:
                    if pattern in heading:
                        score += 1
            type_scores[doc_type] = score
        
        # Return type with highest score, or 'general' if no clear match
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        else:
            return 'general'
    
    def _calculate_structure_score(self, heading_texts: List[str], document_type: str) -> float:
        """Calculate how well headings fit expected document structure"""
        if document_type == 'general':
            return 0.5  # Neutral score for unknown document types
        
        expected_patterns = self.document_patterns[document_type]
        found_patterns = []
        
        for heading in heading_texts:
            for pattern in expected_patterns:
                if pattern in heading:
                    found_patterns.append(pattern)
                    break
        
        # Score based on coverage and order
        coverage_score = len(set(found_patterns)) / len(expected_patterns)
        
        # Bonus for proper order (simplified check)
        order_score = 0.0
        if len(found_patterns) > 1:
            expected_order = [p for p in expected_patterns if p in found_patterns]
            if found_patterns == expected_order:
                order_score = 0.3
        
        return min(1.0, coverage_score + order_score)
    
    def _heading_fits_pattern(self, heading_text: str, document_type: str) -> bool:
        """Check if individual heading fits document pattern"""
        if document_type == 'general':
            return True
        
        heading_lower = heading_text.lower()
        patterns = self.document_patterns[document_type]
        
        return any(pattern in heading_lower for pattern in patterns)
    
    def _validate_hierarchy_consistency(self, headings: List[Dict]) -> List[Dict]:
        """Validate hierarchy consistency across headings"""
        if len(headings) < 2:
            return headings
        
        # Analyze hierarchy patterns
        hierarchy_analysis = self._analyze_hierarchy_patterns(headings)
        
        # Check for consistency issues
        consistency_score = self._calculate_hierarchy_consistency(headings, hierarchy_analysis)
        
        # Update headings with hierarchy validation
        for heading in headings:
            heading['hierarchy_validation'] = {
                'consistency_score': consistency_score,
                'hierarchy_analysis': hierarchy_analysis,
                'is_hierarchy_consistent': consistency_score >= self.quality_thresholds['min_hierarchy_consistency']
            }
        
        return headings
    
    def _analyze_hierarchy_patterns(self, headings: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in heading hierarchy"""
        levels = []
        numbering_types = []
        font_sizes = []
        
        for heading in headings:
            hierarchy_info = heading.get('hierarchy_info', {})
            levels.append(hierarchy_info.get('level', 1))
            numbering_types.append(hierarchy_info.get('numbering_type', 'none'))
            
            font_size = heading.get('font_size', 0)
            if font_size > 0:
                font_sizes.append(font_size)
        
        analysis = {
            'level_distribution': dict(zip(*np.unique(levels, return_counts=True))),
            'numbering_consistency': len(set(numbering_types)) <= 2,  # Allow up to 2 different types
            'level_progression': self._check_level_progression(levels),
            'font_size_consistency': self._check_font_size_consistency(font_sizes) if font_sizes else True
        }
        
        return analysis
    
    def _check_level_progression(self, levels: List[int]) -> bool:
        """Check if heading levels progress logically"""
        if len(levels) < 2:
            return True
        
        # Check for reasonable progression (no jumps > 2 levels)
        for i in range(1, len(levels)):
            level_jump = abs(levels[i] - levels[i-1])
            if level_jump > 2:
                return False
        
        return True
    
    def _check_font_size_consistency(self, font_sizes: List[float]) -> bool:
        """Check if font sizes are consistent with hierarchy"""
        if len(font_sizes) < 2:
            return True
        
        # Check if there's reasonable variation in font sizes
        size_std = statistics.stdev(font_sizes)
        size_mean = statistics.mean(font_sizes)
        
        # Coefficient of variation should be reasonable
        cv = size_std / size_mean if size_mean > 0 else 0
        return 0.05 <= cv <= 0.3  # 5-30% variation is reasonable
    
    def _calculate_hierarchy_consistency(self, headings: List[Dict], 
                                       hierarchy_analysis: Dict) -> float:
        """Calculate overall hierarchy consistency score"""
        score = 1.0
        
        # Penalize for inconsistent numbering
        if not hierarchy_analysis['numbering_consistency']:
            score -= 0.2
        
        # Penalize for poor level progression
        if not hierarchy_analysis['level_progression']:
            score -= 0.3
        
        # Penalize for inconsistent font sizes
        if not hierarchy_analysis['font_size_consistency']:
            score -= 0.2
        
        # Bonus for good level distribution
        level_dist = hierarchy_analysis['level_distribution']
        if 1 in level_dist and level_dist[1] >= len(headings) * 0.2:  # At least 20% level 1 headings
            score += 0.1
        
        return max(0.0, score)
    
    def _cross_validate_headings(self, headings: List[Dict]) -> List[Dict]:
        """Perform cross-validation checks between headings"""
        if len(headings) < 2:
            return headings
        
        # Check for overlapping/duplicate content
        overlap_scores = self._calculate_content_overlap(headings)
        
        # Check spacing and distribution
        distribution_scores = self._analyze_heading_distribution(headings)
        
        # Update headings with cross-validation results
        for i, heading in enumerate(headings):
            heading['cross_validation'] = {
                'overlap_score': overlap_scores[i],
                'distribution_score': distribution_scores[i],
                'relative_quality': self._calculate_relative_quality(heading, headings)
            }
        
        return headings
    
    def _calculate_content_overlap(self, headings: List[Dict]) -> List[float]:
        """Calculate content overlap scores for each heading"""
        overlap_scores = []
        
        for i, heading in enumerate(headings):
            text = heading['text'].lower()
            words = set(text.split())
            
            max_overlap = 0.0
            for j, other_heading in enumerate(headings):
                if i == j:
                    continue
                
                other_text = other_heading['text'].lower()
                other_words = set(other_text.split())
                
                if words and other_words:
                    overlap = len(words.intersection(other_words)) / len(words.union(other_words))
                    max_overlap = max(max_overlap, overlap)
            
            # Higher overlap is worse (lower score)
            overlap_scores.append(1.0 - max_overlap)
        
        return overlap_scores
    
    def _analyze_heading_distribution(self, headings: List[Dict]) -> List[float]:
        """Analyze distribution of headings across document"""
        distribution_scores = []
        pages = [h.get('page', 1) for h in headings]
        
        # Calculate page distribution statistics
        page_counts = defaultdict(int)
        for page in pages:
            page_counts[page] += 1
        
        max_per_page = max(page_counts.values()) if page_counts else 0
        
        for heading in headings:
            page = heading.get('page', 1)
            headings_on_page = page_counts[page]
            
            # Score based on density (too many headings per page is suspicious)
            if headings_on_page <= self.quality_thresholds['max_headings_per_page']:
                distribution_score = 1.0
            else:
                distribution_score = max(0.0, 1.0 - (headings_on_page - self.quality_thresholds['max_headings_per_page']) * 0.1)
            
            distribution_scores.append(distribution_score)
        
        return distribution_scores
    
    def _calculate_relative_quality(self, heading: Dict, all_headings: List[Dict]) -> float:
        """Calculate heading quality relative to other headings"""
        current_score = heading.get('final_score', 0)
        all_scores = [h.get('final_score', 0) for h in all_headings]
        
        if not all_scores:
            return 0.5
        
        # Percentile rank
        rank = sum(1 for score in all_scores if score <= current_score)
        relative_quality = rank / len(all_scores)
        
        return relative_quality
    
    def _assess_final_quality(self, headings: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """Assess final quality and make final validation decisions"""
        quality_scores = {
            'overall_quality': 0.0,
            'individual_quality': [],
            'document_quality': 0.0,
            'hierarchy_quality': 0.0,
            'validation_summary': {}
        }
        
        final_headings = []
        
        for heading in headings:
            # Aggregate all validation scores
            individual_val = heading.get('individual_validation', {})
            document_val = heading.get('document_validation', {})
            hierarchy_val = heading.get('hierarchy_validation', {})
            cross_val = heading.get('cross_validation', {})
            
            # Calculate composite quality score
            quality_components = {
                'individual': individual_val.get('validation_score', 0.5),
                'document_fit': document_val.get('structure_score', 0.5),
                'hierarchy': hierarchy_val.get('consistency_score', 0.5),
                'overlap': cross_val.get('overlap_score', 0.5),
                'distribution': cross_val.get('distribution_score', 0.5),
                'relative': cross_val.get('relative_quality', 0.5)
            }
            
            # Weighted composite score
            weights = {
                'individual': 0.3,
                'document_fit': 0.2,
                'hierarchy': 0.2,
                'overlap': 0.1,
                'distribution': 0.1,
                'relative': 0.1
            }
            
            composite_score = sum(
                quality_components[component] * weights[component]
                for component in quality_components
            )
            
            # Final validation decision
            is_validated = (
                composite_score >= 0.6 and
                individual_val.get('validation_score', 0) >= 0.5 and
                len(individual_val.get('validation_issues', [])) <= 2
            )
            
            # Update heading with final validation results
            final_heading = heading.copy()
            final_heading.update({
                'composite_quality_score': composite_score,
                'quality_components': quality_components,
                'is_validated': is_validated,
                'processing_stage': 'level_6_validation'
            })
            
            final_headings.append(final_heading)
            quality_scores['individual_quality'].append(composite_score)
        
        # Calculate overall quality metrics
        if quality_scores['individual_quality']:
            quality_scores['overall_quality'] = np.mean(quality_scores['individual_quality'])
            quality_scores['document_quality'] = np.mean([
                h.get('document_validation', {}).get('structure_score', 0.5)
                for h in headings
            ])
            quality_scores['hierarchy_quality'] = np.mean([
                h.get('hierarchy_validation', {}).get('consistency_score', 0.5)
                for h in headings
            ])
        
        # Validation summary
        validated_count = len([h for h in final_headings if h.get('is_validated', False)])
        quality_scores['validation_summary'] = {
            'total_headings': len(final_headings),
            'validated_headings': validated_count,
            'validation_rate': validated_count / len(final_headings) if final_headings else 0,
            'quality_distribution': self._get_quality_distribution(final_headings)
        }
        
        return final_headings, quality_scores
    
    def _get_quality_distribution(self, headings: List[Dict]) -> Dict[str, int]:
        """Get distribution of quality scores"""
        scores = [h.get('composite_quality_score', 0) for h in headings]
        
        return {
            'excellent': len([s for s in scores if s > 0.8]),
            'good': len([s for s in scores if 0.6 < s <= 0.8]),
            'fair': len([s for s in scores if 0.4 < s <= 0.6]),
            'poor': len([s for s in scores if s <= 0.4])
        }
    
    def _generate_validation_metadata(self, original_headings: List[Dict], 
                                    validated_headings: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive validation metadata"""
        metadata = {
            'validation_pipeline': {
                'individual_validation': True,
                'document_validation': True,
                'hierarchy_validation': True,
                'cross_validation': True,
                'quality_assessment': True
            },
            'processing_summary': {
                'input_headings': len(original_headings),
                'output_headings': len(validated_headings),
                'validated_headings': len([h for h in validated_headings if h.get('is_validated', False)])
            },
            'quality_thresholds': self.quality_thresholds,
            'validation_issues': self._summarize_validation_issues(validated_headings)
        }
        
        return metadata
    
    def _summarize_validation_issues(self, headings: List[Dict]) -> Dict[str, int]:
        """Summarize all validation issues found"""
        issue_counts = defaultdict(int)
        
        for heading in headings:
            issues = heading.get('individual_validation', {}).get('validation_issues', [])
            for issue in issues:
                issue_counts[issue] += 1
        
        return dict(issue_counts)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration"""
        return {
            'strict_mode': False,
            'document_type_detection': True,
            'hierarchy_validation': True,
            'cross_validation': True,
            'quality_threshold': 0.6
        }
