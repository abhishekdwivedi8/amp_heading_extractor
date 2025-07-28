"""
PDF Layout Parser for Heading Detection
Analyzes PDF layout structure to identify heading candidates based on visual formatting.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class PDFLayoutParser:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PDF layout parser"""
        self.config = config or self._get_default_config()
        
        # Font analysis thresholds
        self.font_thresholds = {
            'large_font_ratio': 1.2,    # Font size ratio to consider "large"
            'small_font_ratio': 0.9,    # Font size ratio to consider "small"
            'bold_flag': 16,            # Flag value for bold text in PyMuPDF
            'italic_flag': 2,           # Flag value for italic text
        }
        
        # Layout analysis parameters
        self.layout_params = {
            'margin_threshold': 50,     # Pixels from edge to consider margin
            'line_spacing_factor': 1.5, # Factor for determining line spacing
            'column_width_min': 100,    # Minimum column width
            'block_gap_threshold': 20,  # Minimum gap between blocks
        }
        
        # Cached analysis results
        self.page_layouts = {}
        self.font_statistics = {}
    
    def parse_pdf_layout(self, pdf_extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse PDF layout from extraction result"""
        raw_extractions = pdf_extraction_result.get('raw_extractions', {})
        
        if 'pymupdf' not in raw_extractions:
            logger.warning("PyMuPDF extraction not available for layout analysis")
            return self._create_empty_layout_result()
        
        pymupdf_data = raw_extractions['pymupdf']
        
        # Analyze each page
        page_analyses = []
        for page_data in pymupdf_data:
            page_analysis = self._analyze_page_layout(page_data)
            page_analyses.append(page_analysis)
        
        # Combine results and generate document-level insights
        layout_result = self._combine_page_analyses(page_analyses)
        
        return layout_result
    
    def _analyze_page_layout(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze layout of a single page"""
        page_number = page_data.get('page_number', 1)
        blocks = page_data.get('blocks', [])
        
        if not blocks:
            return self._create_empty_page_analysis(page_number)
        
        # Extract text blocks with formatting information
        text_elements = self._extract_text_elements(blocks)
        
        # Analyze font characteristics
        font_analysis = self._analyze_fonts(text_elements)
        
        # Analyze spatial layout
        spatial_analysis = self._analyze_spatial_layout(text_elements)
        
        # Detect potential headings based on layout
        heading_candidates = self._detect_layout_headings(text_elements, font_analysis, spatial_analysis)
        
        # Analyze text hierarchy
        hierarchy_analysis = self._analyze_text_hierarchy(text_elements, font_analysis)
        
        page_analysis = {
            'page_number': page_number,
            'text_elements': text_elements,
            'font_analysis': font_analysis,
            'spatial_analysis': spatial_analysis,
            'heading_candidates': heading_candidates,
            'hierarchy_analysis': hierarchy_analysis,
            'layout_quality_score': self._calculate_layout_quality(text_elements, spatial_analysis)
        }
        
        return page_analysis
    
    def _extract_text_elements(self, blocks: List[Dict]) -> List[Dict[str, Any]]:
        """Extract text elements with formatting and position information"""
        text_elements = []
        
        for block in blocks:
            text = block.get('text', '').strip()
            if not text:
                continue
            
            element = {
                'text': text,
                'bbox': block.get('bbox', [0, 0, 0, 0]),  # [x0, y0, x1, y1]
                'font': block.get('font', ''),
                'font_size': block.get('size', 12),
                'flags': block.get('flags', 0),
                'is_bold': self._is_bold(block.get('flags', 0)),
                'is_italic': self._is_italic(block.get('flags', 0)),
                'word_count': len(text.split()),
                'char_count': len(text)
            }
            
            # Calculate additional properties
            element.update(self._calculate_element_properties(element))
            
            text_elements.append(element)
        
        return text_elements
    
    def _calculate_element_properties(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional properties for text element"""
        bbox = element['bbox']
        text = element['text']
        
        properties = {
            'width': bbox[2] - bbox[0],
            'height': bbox[3] - bbox[1],
            'center_x': (bbox[0] + bbox[2]) / 2,
            'center_y': (bbox[1] + bbox[3]) / 2,
            'aspect_ratio': (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1),
            'text_density': len(text.replace(' ', '')) / max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1),
            'is_single_line': element['word_count'] <= self.config['single_line_word_limit'],
            'is_short': element['word_count'] <= self.config['short_text_word_limit'],
            'is_capitalized': text[0].isupper() if text else False,
            'is_all_caps': text.isupper(),
            'is_title_case': text.istitle(),
            'has_numbers': any(c.isdigit() for c in text),
            'ends_with_punctuation': text.rstrip()[-1] in '.!?:;' if text.rstrip() else False
        }
        
        return properties
    
    def _analyze_fonts(self, text_elements: List[Dict]) -> Dict[str, Any]:
        """Analyze font characteristics across text elements"""
        if not text_elements:
            return {}
        
        font_sizes = [elem['font_size'] for elem in text_elements]
        fonts = [elem['font'] for elem in text_elements]
        
        # Basic statistics
        font_stats = {
            'font_sizes': font_sizes,
            'unique_fonts': list(set(fonts)),
            'avg_font_size': statistics.mean(font_sizes),
            'median_font_size': statistics.median(font_sizes),
            'font_size_std': statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0,
            'min_font_size': min(font_sizes),
            'max_font_size': max(font_sizes),
            'font_size_range': max(font_sizes) - min(font_sizes)
        }
        
        # Font size distribution
        font_size_counts = defaultdict(int)
        for size in font_sizes:
            font_size_counts[size] += 1
        
        font_stats['font_size_distribution'] = dict(font_size_counts)
        font_stats['most_common_font_size'] = max(font_size_counts, key=font_size_counts.get)
        
        # Identify size categories
        avg_size = font_stats['avg_font_size']
        font_stats['large_font_threshold'] = avg_size * self.font_thresholds['large_font_ratio']
        font_stats['small_font_threshold'] = avg_size * self.font_thresholds['small_font_ratio']
        
        # Font family analysis
        font_families = defaultdict(int)
        for font in fonts:
            font_families[font] += 1
        
        font_stats['font_distribution'] = dict(font_families)
        font_stats['primary_font'] = max(font_families, key=font_families.get) if font_families else ''
        
        # Formatting analysis
        bold_count = sum(1 for elem in text_elements if elem['is_bold'])
        italic_count = sum(1 for elem in text_elements if elem['is_italic'])
        
        font_stats['bold_ratio'] = bold_count / len(text_elements)
        font_stats['italic_ratio'] = italic_count / len(text_elements)
        
        return font_stats
    
    def _analyze_spatial_layout(self, text_elements: List[Dict]) -> Dict[str, Any]:
        """Analyze spatial layout and positioning"""
        if not text_elements:
            return {}
        
        # Extract coordinates
        x_coords = []
        y_coords = []
        widths = []
        heights = []
        
        for elem in text_elements:
            bbox = elem['bbox']
            x_coords.extend([bbox[0], bbox[2]])
            y_coords.extend([bbox[1], bbox[3]])
            widths.append(elem['width'])
            heights.append(elem['height'])
        
        # Page boundaries
        page_bounds = {
            'left': min(x_coords) if x_coords else 0,
            'right': max(x_coords) if x_coords else 0,
            'top': min(y_coords) if y_coords else 0,
            'bottom': max(y_coords) if y_coords else 0
        }
        
        page_width = page_bounds['right'] - page_bounds['left']
        page_height = page_bounds['bottom'] - page_bounds['top']
        
        # Margin analysis
        margin_analysis = self._analyze_margins(text_elements, page_bounds)
        
        # Column detection
        column_analysis = self._detect_columns(text_elements, page_width)
        
        # Spacing analysis
        spacing_analysis = self._analyze_spacing(text_elements)
        
        # Alignment analysis
        alignment_analysis = self._analyze_alignment(text_elements, page_width)
        
        spatial_analysis = {
            'page_bounds': page_bounds,
            'page_width': page_width,
            'page_height': page_height,
            'margins': margin_analysis,
            'columns': column_analysis,
            'spacing': spacing_analysis,
            'alignment': alignment_analysis,
            'text_density': len(text_elements) / max(page_width * page_height, 1),
            'avg_element_width': statistics.mean(widths) if widths else 0,
            'avg_element_height': statistics.mean(heights) if heights else 0
        }
        
        return spatial_analysis
    
    def _analyze_margins(self, text_elements: List[Dict], page_bounds: Dict) -> Dict[str, Any]:
        """Analyze page margins"""
        left_positions = [elem['bbox'][0] for elem in text_elements]
        right_positions = [elem['bbox'][2] for elem in text_elements]
        top_positions = [elem['bbox'][1] for elem in text_elements]
        bottom_positions = [elem['bbox'][3] for elem in text_elements]
        
        # Calculate margins
        left_margin = min(left_positions) - page_bounds['left'] if left_positions else 0
        right_margin = page_bounds['right'] - max(right_positions) if right_positions else 0
        top_margin = min(top_positions) - page_bounds['top'] if top_positions else 0
        bottom_margin = page_bounds['bottom'] - max(bottom_positions) if bottom_positions else 0
        
        return {
            'left': left_margin,
            'right': right_margin,
            'top': top_margin,
            'bottom': bottom_margin,
            'total_horizontal': left_margin + right_margin,
            'total_vertical': top_margin + bottom_margin
        }
    
    def _detect_columns(self, text_elements: List[Dict], page_width: float) -> Dict[str, Any]:
        """Detect column layout"""
        if not text_elements:
            return {'num_columns': 1, 'column_boundaries': []}
        
        # Collect x-coordinates of text elements
        left_edges = [elem['bbox'][0] for elem in text_elements]
        right_edges = [elem['bbox'][2] for elem in text_elements]
        
        # Find potential column boundaries
        column_boundaries = []
        
        # Simple approach: look for gaps in horizontal space
        sorted_lefts = sorted(set(left_edges))
        
        for i in range(len(sorted_lefts) - 1):
            gap = sorted_lefts[i + 1] - sorted_lefts[i]
            if gap > self.layout_params['column_width_min']:
                column_boundaries.append(sorted_lefts[i + 1])
        
        num_columns = len(column_boundaries) + 1
        
        return {
            'num_columns': num_columns,
            'column_boundaries': column_boundaries,
            'is_single_column': num_columns == 1,
            'is_multi_column': num_columns > 1
        }
    
    def _analyze_spacing(self, text_elements: List[Dict]) -> Dict[str, Any]:
        """Analyze spacing between text elements"""
        if len(text_elements) < 2:
            return {'vertical_gaps': [], 'horizontal_gaps': []}
        
        # Sort elements by position
        sorted_by_y = sorted(text_elements, key=lambda x: x['bbox'][1])
        
        vertical_gaps = []
        horizontal_gaps = []
        
        # Calculate vertical gaps
        for i in range(len(sorted_by_y) - 1):
            current_bottom = sorted_by_y[i]['bbox'][3]
            next_top = sorted_by_y[i + 1]['bbox'][1]
            gap = next_top - current_bottom
            if gap > 0:
                vertical_gaps.append(gap)
        
        # Calculate horizontal gaps (for elements on similar y-levels)
        y_tolerance = 10  # Pixels tolerance for "same line"
        
        for i, elem1 in enumerate(text_elements):
            for j, elem2 in enumerate(text_elements[i + 1:], i + 1):
                # Check if elements are on similar y-level
                y_diff = abs(elem1['center_y'] - elem2['center_y'])
                if y_diff <= y_tolerance:
                    # Calculate horizontal gap
                    if elem1['bbox'][2] < elem2['bbox'][0]:  # elem1 is to the left
                        gap = elem2['bbox'][0] - elem1['bbox'][2]
                        horizontal_gaps.append(gap)
                    elif elem2['bbox'][2] < elem1['bbox'][0]:  # elem2 is to the left
                        gap = elem1['bbox'][0] - elem2['bbox'][2]
                        horizontal_gaps.append(gap)
        
        return {
            'vertical_gaps': vertical_gaps,
            'horizontal_gaps': horizontal_gaps,
            'avg_vertical_gap': statistics.mean(vertical_gaps) if vertical_gaps else 0,
            'avg_horizontal_gap': statistics.mean(horizontal_gaps) if horizontal_gaps else 0,
            'median_vertical_gap': statistics.median(vertical_gaps) if vertical_gaps else 0,
            'large_vertical_gaps': [g for g in vertical_gaps if g > self.layout_params['block_gap_threshold']]
        }
    
    def _analyze_alignment(self, text_elements: List[Dict], page_width: float) -> Dict[str, Any]:
        """Analyze text alignment patterns"""
        if not text_elements:
            return {}
        
        left_edges = [elem['bbox'][0] for elem in text_elements]
        right_edges = [elem['bbox'][2] for elem in text_elements]
        center_x_coords = [elem['center_x'] for elem in text_elements]
        
        # Find common alignment positions
        alignment_tolerance = 5  # Pixels tolerance for alignment
        
        # Left alignment clusters
        left_clusters = self._find_alignment_clusters(left_edges, alignment_tolerance)
        
        # Right alignment clusters
        right_clusters = self._find_alignment_clusters(right_edges, alignment_tolerance)
        
        # Center alignment detection
        page_center = page_width / 2
        center_aligned = [elem for elem in text_elements 
                         if abs(elem['center_x'] - page_center) < alignment_tolerance]
        
        return {
            'left_alignment_clusters': left_clusters,
            'right_alignment_clusters': right_clusters,
            'center_aligned_count': len(center_aligned),
            'dominant_left_alignment': max(left_clusters, key=len) if left_clusters else [],
            'has_consistent_left_alignment': len(left_clusters) > 0 and max(len(cluster) for cluster in left_clusters) > len(text_elements) * 0.3,
            'center_alignment_ratio': len(center_aligned) / len(text_elements)
        }
    
    def _find_alignment_clusters(self, positions: List[float], tolerance: float) -> List[List[float]]:
        """Find clusters of aligned positions"""
        if not positions:
            return []
        
        sorted_positions = sorted(positions)
        clusters = []
        current_cluster = [sorted_positions[0]]
        
        for pos in sorted_positions[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                if len(current_cluster) > 1:  # Only keep clusters with multiple elements
                    clusters.append(current_cluster)
                current_cluster = [pos]
        
        # Don't forget the last cluster
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        return clusters
    
    def _detect_layout_headings(self, text_elements: List[Dict], 
                               font_analysis: Dict, spatial_analysis: Dict) -> List[Dict]:
        """Detect potential headings based on layout characteristics"""
        heading_candidates = []
        
        if not text_elements or not font_analysis:
            return heading_candidates
        
        large_font_threshold = font_analysis.get('large_font_threshold', 0)
        avg_font_size = font_analysis.get('avg_font_size', 12)
        
        for elem in text_elements:
            heading_score = 0.0
            reasons = []
            
            # Font size analysis
            if elem['font_size'] > large_font_threshold:
                heading_score += 0.3
                reasons.append('large_font')
            
            # Bold text
            if elem['is_bold']:
                heading_score += 0.25
                reasons.append('bold')
            
            # Isolation (spacing above and below)
            isolation_score = self._calculate_isolation_score(elem, text_elements, spatial_analysis)
            heading_score += isolation_score * 0.2
            if isolation_score > 0.5:
                reasons.append('isolated')
            
            # Short text (typical of headings)
            if elem['is_short'] and elem['word_count'] >= 2:
                heading_score += 0.15
                reasons.append('appropriate_length')
            
            # Capitalization patterns
            if elem['is_title_case'] or elem['is_all_caps']:
                heading_score += 0.1
                reasons.append('capitalized')
            
            # Position on page (headings often at top)
            if self._is_near_top_of_page(elem, spatial_analysis):
                heading_score += 0.1
                reasons.append('top_position')
            
            # Center alignment (sometimes used for headings)
            center_ratio = spatial_analysis.get('alignment', {}).get('center_alignment_ratio', 0)
            if center_ratio > 0.1 and abs(elem['center_x'] - spatial_analysis['page_width'] / 2) < 20:
                heading_score += 0.05
                reasons.append('center_aligned')
            
            # Different font from body text
            primary_font = font_analysis.get('primary_font', '')
            if primary_font and elem['font'] != primary_font:
                heading_score += 0.05
                reasons.append('different_font')
            
            # Create heading candidate if score is above threshold
            if heading_score >= self.config['heading_detection_threshold']:
                candidate = {
                    'element': elem,
                    'layout_score': heading_score,
                    'detection_reasons': reasons,
                    'font_size_ratio': elem['font_size'] / avg_font_size,
                    'isolation_score': isolation_score
                }
                heading_candidates.append(candidate)
        
        # Sort by score (descending)
        heading_candidates.sort(key=lambda x: x['layout_score'], reverse=True)
        
        return heading_candidates
    
    def _calculate_isolation_score(self, element: Dict, all_elements: List[Dict], 
                                  spatial_analysis: Dict) -> float:
        """Calculate how isolated an element is from surrounding text"""
        elem_y = element['center_y']
        elem_height = element['height']
        
        # Find elements above and below
        elements_above = [e for e in all_elements if e['bbox'][3] < element['bbox'][1]]
        elements_below = [e for e in all_elements if e['bbox'][1] > element['bbox'][3]]
        
        # Calculate distances
        distance_above = float('inf')
        distance_below = float('inf')
        
        if elements_above:
            closest_above = max(elements_above, key=lambda x: x['bbox'][3])
            distance_above = element['bbox'][1] - closest_above['bbox'][3]
        
        if elements_below:
            closest_below = min(elements_below, key=lambda x: x['bbox'][1])
            distance_below = closest_below['bbox'][1] - element['bbox'][3]
        
        # Calculate isolation score based on distances
        avg_vertical_gap = spatial_analysis.get('spacing', {}).get('avg_vertical_gap', 10)
        
        isolation_score = 0.0
        
        # Score based on distance above
        if distance_above > avg_vertical_gap * 1.5:
            isolation_score += 0.5
        elif distance_above > avg_vertical_gap:
            isolation_score += 0.25
        
        # Score based on distance below
        if distance_below > avg_vertical_gap * 1.5:
            isolation_score += 0.5
        elif distance_below > avg_vertical_gap:
            isolation_score += 0.25
        
        return min(1.0, isolation_score)
    
    def _is_near_top_of_page(self, element: Dict, spatial_analysis: Dict) -> bool:
        """Check if element is near the top of the page"""
        page_height = spatial_analysis.get('page_height', 1)
        element_y = element['bbox'][1]
        
        # Consider top 20% of page as "near top"
        top_threshold = page_height * 0.2
        
        return element_y <= top_threshold
    
    def _analyze_text_hierarchy(self, text_elements: List[Dict], 
                               font_analysis: Dict) -> Dict[str, Any]:
        """Analyze text hierarchy based on font sizes and formatting"""
        if not text_elements:
            return {}
        
        # Group elements by font size
        size_groups = defaultdict(list)
        for elem in text_elements:
            size_groups[elem['font_size']].append(elem)
        
        # Sort font sizes (largest first)
        sorted_sizes = sorted(size_groups.keys(), reverse=True)
        
        # Assign hierarchy levels
        hierarchy_levels = {}
        level = 1
        
        for size in sorted_sizes:
            elements_at_size = size_groups[size]
            
            # Further subdivide by formatting (bold vs regular)
            bold_elements = [e for e in elements_at_size if e['is_bold']]
            regular_elements = [e for e in elements_at_size if not e['is_bold']]
            
            # Assign levels
            for elem in bold_elements:
                hierarchy_levels[id(elem)] = level
            
            if bold_elements and regular_elements:
                level += 1
            
            for elem in regular_elements:
                hierarchy_levels[id(elem)] = level
            
            level += 1
        
        # Calculate hierarchy statistics
        level_counts = defaultdict(int)
        for elem_id, level in hierarchy_levels.items():
            level_counts[level] += 1
        
        hierarchy_analysis = {
            'num_hierarchy_levels': len(set(hierarchy_levels.values())),
            'level_distribution': dict(level_counts),
            'hierarchy_levels': hierarchy_levels,
            'has_clear_hierarchy': len(set(hierarchy_levels.values())) >= 2,
            'hierarchy_consistency_score': self._calculate_hierarchy_consistency(hierarchy_levels, text_elements)
        }
        
        return hierarchy_analysis
    
    def _calculate_hierarchy_consistency(self, hierarchy_levels: Dict, 
                                       text_elements: List[Dict]) -> float:
        """Calculate how consistent the text hierarchy is"""
        if len(hierarchy_levels) < 2:
            return 1.0
        
        # Check if similar-looking elements have similar hierarchy levels
        consistency_score = 1.0
        
        # Group elements by their characteristics
        for i, elem1 in enumerate(text_elements):
            for j, elem2 in enumerate(text_elements[i + 1:], i + 1):
                elem1_id = id(elem1)
                elem2_id = id(elem2)
                
                if elem1_id not in hierarchy_levels or elem2_id not in hierarchy_levels:
                    continue
                
                # Check if elements are similar
                are_similar = (
                    elem1['font_size'] == elem2['font_size'] and
                    elem1['is_bold'] == elem2['is_bold'] and
                    elem1['font'] == elem2['font']
                )
                
                if are_similar:
                    # They should have the same hierarchy level
                    if hierarchy_levels[elem1_id] != hierarchy_levels[elem2_id]:
                        consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def _calculate_layout_quality(self, text_elements: List[Dict], 
                                 spatial_analysis: Dict) -> float:
        """Calculate overall layout quality score"""
        if not text_elements:
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Consistency in alignment
        alignment = spatial_analysis.get('alignment', {})
        if alignment.get('has_consistent_left_alignment', False):
            quality_score += 0.2
        
        # Reasonable margins
        margins = spatial_analysis.get('margins', {})
        if margins.get('left', 0) > 20 and margins.get('right', 0) > 20:
            quality_score += 0.1
        
        # Reasonable spacing
        spacing = spatial_analysis.get('spacing', {})
        avg_gap = spacing.get('avg_vertical_gap', 0)
        if 5 < avg_gap < 50:  # Reasonable range
            quality_score += 0.1
        
        # Text density (not too crowded)
        density = spatial_analysis.get('text_density', 0)
        if 0.001 < density < 0.01:  # Reasonable range
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _combine_page_analyses(self, page_analyses: List[Dict]) -> Dict[str, Any]:
        """Combine analyses from multiple pages"""
        if not page_analyses:
            return self._create_empty_layout_result()
        
        # Aggregate statistics
        total_elements = sum(len(analysis.get('text_elements', [])) for analysis in page_analyses)
        total_headings = sum(len(analysis.get('heading_candidates', [])) for analysis in page_analyses)
        
        # Combine font statistics
        all_font_sizes = []
        all_fonts = []
        
        for analysis in page_analyses:
            font_analysis = analysis.get('font_analysis', {})
            all_font_sizes.extend(font_analysis.get('font_sizes', []))
            all_fonts.extend(font_analysis.get('unique_fonts', []))
        
        # Calculate document-level font statistics
        document_font_stats = {}
        if all_font_sizes:
            document_font_stats = {
                'avg_font_size': statistics.mean(all_font_sizes),
                'font_size_range': max(all_font_sizes) - min(all_font_sizes),
                'unique_fonts': list(set(all_fonts)),
                'font_consistency': len(set(all_fonts)) <= 3  # Good if 3 or fewer fonts
            }
        
        # Calculate average layout quality
        quality_scores = [analysis.get('layout_quality_score', 0) for analysis in page_analyses]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        combined_result = {
            'page_analyses': page_analyses,
            'document_statistics': {
                'total_pages': len(page_analyses),
                'total_text_elements': total_elements,
                'total_heading_candidates': total_headings,
                'avg_elements_per_page': total_elements / len(page_analyses) if page_analyses else 0,
                'avg_headings_per_page': total_headings / len(page_analyses) if page_analyses else 0
            },
            'document_font_analysis': document_font_stats,
            'average_layout_quality': avg_quality,
            'layout_consistency': self._calculate_document_layout_consistency(page_analyses)
        }
        
        return combined_result
    
    def _calculate_document_layout_consistency(self, page_analyses: List[Dict]) -> float:
        """Calculate consistency of layout across pages"""
        if len(page_analyses) <= 1:
            return 1.0
        
        consistency_factors = []
        
        # Font consistency
        font_sets = []
        for analysis in page_analyses:
            fonts = analysis.get('font_analysis', {}).get('unique_fonts', [])
            font_sets.append(set(fonts))
        
        if font_sets:
            common_fonts = set.intersection(*font_sets)
            all_fonts = set.union(*font_sets)
            font_consistency = len(common_fonts) / len(all_fonts) if all_fonts else 1.0
            consistency_factors.append(font_consistency)
        
        # Layout quality consistency
        quality_scores = [analysis.get('layout_quality_score', 0) for analysis in page_analyses]
        if quality_scores:
            quality_std = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
            quality_consistency = max(0, 1 - (quality_std / 0.5))  # Normalize by reasonable std
            consistency_factors.append(quality_consistency)
        
        return statistics.mean(consistency_factors) if consistency_factors else 0.5
    
    def _create_empty_layout_result(self) -> Dict[str, Any]:
        """Create empty layout result"""
        return {
            'page_analyses': [],
            'document_statistics': {
                'total_pages': 0,
                'total_text_elements': 0,
                'total_heading_candidates': 0
            },
            'document_font_analysis': {},
            'average_layout_quality': 0.0,
            'layout_consistency': 0.0
        }
    
    def _create_empty_page_analysis(self, page_number: int) -> Dict[str, Any]:
        """Create empty page analysis"""
        return {
            'page_number': page_number,
            'text_elements': [],
            'font_analysis': {},
            'spatial_analysis': {},
            'heading_candidates': [],
            'hierarchy_analysis': {},
            'layout_quality_score': 0.0
        }
    
    def _is_bold(self, flags: int) -> bool:
        """Check if text is bold based on flags"""
        return bool(flags & self.font_thresholds['bold_flag'])
    
    def _is_italic(self, flags: int) -> bool:
        """Check if text is italic based on flags"""
        return bool(flags & self.font_thresholds['italic_flag'])
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'heading_detection_threshold': 0.4,
            'single_line_word_limit': 15,
            'short_text_word_limit': 10,
            'enable_font_analysis': True,
            'enable_spatial_analysis': True,
            'enable_hierarchy_analysis': True,
            'min_font_size': 6,
            'max_font_size': 72
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update parser configuration"""
        self.config.update(new_config)
    
    def get_layout_summary(self, layout_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of layout analysis"""
        doc_stats = layout_result.get('document_statistics', {})
        font_analysis = layout_result.get('document_font_analysis', {})
        
        summary = {
            'pages_processed': doc_stats.get('total_pages', 0),
            'heading_candidates_found': doc_stats.get('total_heading_candidates', 0),
            'layout_quality': layout_result.get('average_layout_quality', 0),
            'layout_consistency': layout_result.get('layout_consistency', 0),
            'font_diversity': len(font_analysis.get('unique_fonts', [])),
            'has_consistent_fonts': font_analysis.get('font_consistency', False),
            'recommendations': self._generate_layout_recommendations(layout_result)
        }
        
        return summary
    
    def _generate_layout_recommendations(self, layout_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on layout analysis"""
        recommendations = []
        
        quality = layout_result.get('average_layout_quality', 0)
        consistency = layout_result.get('layout_consistency', 0)
        font_analysis = layout_result.get('document_font_analysis', {})
        
        if quality < 0.5:
            recommendations.append("Document layout quality is low. Consider improving text formatting and spacing.")
        
        if consistency < 0.5:
            recommendations.append("Layout consistency across pages is low. Check for formatting inconsistencies.")
        
        if len(font_analysis.get('unique_fonts', [])) > 5:
            recommendations.append("Many different fonts detected. Consider using fewer fonts for better consistency.")
        
        heading_count = layout_result.get('document_statistics', {}).get('total_heading_candidates', 0)
        if heading_count == 0:
            recommendations.append("No clear heading candidates found. Document may lack proper heading formatting.")
        
        return recommendations
