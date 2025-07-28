"""
Heading Extraction Evaluation Scorer
Provides comprehensive evaluation metrics for heading detection performance.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from collections import defaultdict
import json
import re

logger = logging.getLogger(__name__)

class HeadingEvaluationScorer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evaluation scorer"""
        self.config = config or self._get_default_config()
        
        # Evaluation metrics
        self.metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'specificity': 0.0
        }
        
        # Detailed analysis
        self.confusion_matrix = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        
        # Error analysis
        self.error_analysis = {
            'missed_headings': [],
            'false_headings': [],
            'error_categories': defaultdict(int)
        }
    
    def evaluate_predictions(self, predicted_headings: List[Dict[str, Any]], 
                           ground_truth_headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate predicted headings against ground truth"""
        logger.info(f"Evaluating {len(predicted_headings)} predictions against {len(ground_truth_headings)} ground truth")
        
        # Normalize headings for comparison
        pred_normalized = self._normalize_headings(predicted_headings)
        gt_normalized = self._normalize_headings(ground_truth_headings)
        
        # Calculate matches
        matches = self._find_matches(pred_normalized, gt_normalized)
        
        # Calculate basic metrics
        tp, fp, fn, tn = self._calculate_confusion_matrix(matches, pred_normalized, gt_normalized)
        
        self.confusion_matrix.update({
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        })
        
        # Calculate derived metrics
        self.metrics = self._calculate_metrics(tp, fp, fn, tn)
        
        # Perform detailed analysis
        detailed_analysis = self._perform_detailed_analysis(
            matches, pred_normalized, gt_normalized, predicted_headings, ground_truth_headings
        )
        
        # Compile evaluation results
        evaluation_results = {
            'metrics': self.metrics,
            'confusion_matrix': self.confusion_matrix,
            'detailed_analysis': detailed_analysis,
            'summary': self._generate_summary()
        }
        
        return evaluation_results
    
    def _normalize_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize headings for consistent comparison"""
        normalized = []
        
        for heading in headings:
            # Extract text and normalize
            text = heading.get('text', '').strip()
            if not text:
                continue
            
            # Text normalization
            if self.config['case_insensitive']:
                text = text.lower()
            
            if self.config['remove_punctuation']:
                text = re.sub(r'[^\w\s]', ' ', text)
            
            if self.config['normalize_whitespace']:
                text = ' '.join(text.split())
            
            normalized_heading = {
                'text': text,
                'page': heading.get('page', 1),
                'confidence': heading.get('confidence', 0.0),
                'original': heading
            }
            
            # Add position information if available
            if 'bbox' in heading:
                normalized_heading['bbox'] = heading['bbox']
            if 'line' in heading:
                normalized_heading['line'] = heading['line']
            
            normalized.append(normalized_heading)
        
        return normalized
    
    def _find_matches(self, predicted: List[Dict], ground_truth: List[Dict]) -> List[Dict[str, Any]]:
        """Find matches between predicted and ground truth headings"""
        matches = []
        used_gt_indices = set()
        
        for pred_idx, pred_heading in enumerate(predicted):
            best_match = None
            best_score = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_heading in enumerate(ground_truth):
                if gt_idx in used_gt_indices:
                    continue
                
                # Calculate similarity score
                similarity = self._calculate_similarity(pred_heading, gt_heading)
                
                if similarity > best_score and similarity >= self.config['match_threshold']:
                    best_match = gt_heading
                    best_score = similarity
                    best_gt_idx = gt_idx
            
            if best_match:
                matches.append({
                    'predicted_index': pred_idx,
                    'ground_truth_index': best_gt_idx,
                    'predicted': pred_heading,
                    'ground_truth': best_match,
                    'similarity_score': best_score,
                    'match_type': 'exact' if best_score >= 0.95 else 'partial'
                })
                used_gt_indices.add(best_gt_idx)
        
        return matches
    
    def _calculate_similarity(self, pred_heading: Dict, gt_heading: Dict) -> float:
        """Calculate similarity between predicted and ground truth heading"""
        # Text similarity (primary factor)
        pred_text = pred_heading['text']
        gt_text = gt_heading['text']
        
        text_similarity = self._text_similarity(pred_text, gt_text)
        
        # Position similarity (if available)
        position_similarity = 1.0
        if 'page' in pred_heading and 'page' in gt_heading:
            if pred_heading['page'] == gt_heading['page']:
                position_similarity = 1.0
            else:
                # Penalize different pages
                page_diff = abs(pred_heading['page'] - gt_heading['page'])
                position_similarity = max(0.0, 1.0 - (page_diff * 0.2))
        
        # Combine similarities
        combined_similarity = (
            text_similarity * self.config['text_weight'] +
            position_similarity * self.config['position_weight']
        )
        
        return combined_similarity
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two strings"""
        if text1 == text2:
            return 1.0
        
        # Jaccard similarity on words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_confusion_matrix(self, matches: List[Dict], 
                                  predicted: List[Dict], 
                                  ground_truth: List[Dict]) -> Tuple[int, int, int, int]:
        """Calculate confusion matrix components"""
        # True Positives: correctly identified headings
        tp = len(matches)
        
        # False Positives: predicted headings that don't match ground truth
        fp = len(predicted) - tp
        
        # False Negatives: ground truth headings not predicted
        matched_gt_indices = {match['ground_truth_index'] for match in matches}
        fn = len(ground_truth) - len(matched_gt_indices)
        
        # True Negatives: correctly identified non-headings
        # This is harder to calculate without negative examples
        # For now, we'll estimate based on document structure
        tn = 0  # Will be estimated later if needed
        
        return tp, fp, fn, tn
    
    def _calculate_metrics(self, tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
        """Calculate evaluation metrics from confusion matrix"""
        metrics = {}
        
        # Precision: TP / (TP + FP)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: TP / (TP + FN)
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        p, r = metrics['precision'], metrics['recall']
        metrics['f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        total = tp + tn + fp + fn
        metrics['accuracy'] = (tp + tn) / total if total > 0 else 0.0
        
        # Specificity: TN / (TN + FP)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return metrics
    
    def _perform_detailed_analysis(self, matches: List[Dict], 
                                 pred_normalized: List[Dict],
                                 gt_normalized: List[Dict],
                                 original_predicted: List[Dict],
                                 original_ground_truth: List[Dict]) -> Dict[str, Any]:
        """Perform detailed error analysis"""
        analysis = {
            'matches': matches,
            'missed_headings': [],
            'false_positives': [],
            'confidence_analysis': {},
            'error_patterns': defaultdict(int),
            'hierarchy_analysis': {}
        }
        
        # Find missed headings (False Negatives)
        matched_gt_indices = {match['ground_truth_index'] for match in matches}
        for i, gt_heading in enumerate(gt_normalized):
            if i not in matched_gt_indices:
                analysis['missed_headings'].append({
                    'ground_truth': gt_heading,
                    'original': original_ground_truth[i] if i < len(original_ground_truth) else None
                })
        
        # Find false positives
        matched_pred_indices = {match['predicted_index'] for match in matches}
        for i, pred_heading in enumerate(pred_normalized):
            if i not in matched_pred_indices:
                analysis['false_positives'].append({
                    'predicted': pred_heading,
                    'original': original_predicted[i] if i < len(original_predicted) else None
                })
        
        # Confidence analysis
        if matches:
            confidences = [match['predicted']['confidence'] for match in matches]
            analysis['confidence_analysis'] = {
                'mean_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'min_confidence': min(confidences),
                'max_confidence': max(confidences)
            }
        
        # Error pattern analysis
        self._analyze_error_patterns(analysis)
        
        # Hierarchy analysis
        analysis['hierarchy_analysis'] = self._analyze_hierarchy_accuracy(
            original_predicted, original_ground_truth
        )
        
        return analysis
    
    def _analyze_error_patterns(self, analysis: Dict[str, Any]):
        """Analyze patterns in errors"""
        # Analyze missed headings
        for missed in analysis['missed_headings']:
            gt_text = missed['ground_truth']['text']
            
            # Categorize missed headings
            if len(gt_text.split()) <= 2:
                analysis['error_patterns']['missed_short_headings'] += 1
            elif len(gt_text.split()) > 10:
                analysis['error_patterns']['missed_long_headings'] += 1
            
            if gt_text.isupper():
                analysis['error_patterns']['missed_all_caps'] += 1
            elif gt_text.istitle():
                analysis['error_patterns']['missed_title_case'] += 1
            
            if any(char.isdigit() for char in gt_text):
                analysis['error_patterns']['missed_numbered_headings'] += 1
        
        # Analyze false positives
        for fp in analysis['false_positives']:
            pred_text = fp['predicted']['text']
            
            # Categorize false positives
            if len(pred_text.split()) <= 2:
                analysis['error_patterns']['fp_short_text'] += 1
            elif len(pred_text.split()) > 10:
                analysis['error_patterns']['fp_long_text'] += 1
            
            if pred_text.isupper():
                analysis['error_patterns']['fp_all_caps'] += 1
            
            # Check for common false positive patterns
            if any(pattern in pred_text.lower() for pattern in ['page', 'figure', 'table']):
                analysis['error_patterns']['fp_document_elements'] += 1
    
    def _analyze_hierarchy_accuracy(self, predicted: List[Dict], 
                                   ground_truth: List[Dict]) -> Dict[str, Any]:
        """Analyze accuracy of heading hierarchy detection"""
        hierarchy_analysis = {
            'level_accuracy': 0.0,
            'level_distribution_predicted': defaultdict(int),
            'level_distribution_ground_truth': defaultdict(int),
            'level_confusion_matrix': defaultdict(lambda: defaultdict(int))
        }
        
        # Extract hierarchy levels
        pred_levels = []
        gt_levels = []
        
        for heading in predicted:
            level = heading.get('level', heading.get('hierarchy_info', {}).get('level', 1))
            pred_levels.append(level)
            hierarchy_analysis['level_distribution_predicted'][level] += 1
        
        for heading in ground_truth:
            level = heading.get('level', 1)
            gt_levels.append(level)
            hierarchy_analysis['level_distribution_ground_truth'][level] += 1
        
        # Calculate level accuracy (if we have matched pairs)
        if pred_levels and gt_levels:
            # This is a simplified analysis - in practice, you'd need proper matching
            min_len = min(len(pred_levels), len(gt_levels))
            correct_levels = sum(1 for i in range(min_len) if pred_levels[i] == gt_levels[i])
            hierarchy_analysis['level_accuracy'] = correct_levels / min_len if min_len > 0 else 0.0
            
            # Level confusion matrix
            for i in range(min_len):
                pred_level = pred_levels[i]
                gt_level = gt_levels[i]
                hierarchy_analysis['level_confusion_matrix'][gt_level][pred_level] += 1
        
        return hierarchy_analysis
    
    def evaluate_confidence_calibration(self, predicted_headings: List[Dict[str, Any]], 
                                      ground_truth_headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate how well confidence scores correlate with accuracy"""
        # This requires matched predictions
        matches = self._find_matches(
            self._normalize_headings(predicted_headings),
            self._normalize_headings(ground_truth_headings)
        )
        
        if not matches:
            return {'error': 'No matches found for calibration analysis'}
        
        # Extract confidence scores and match quality
        confidences = []
        accuracies = []
        
        for match in matches:
            confidence = match['predicted']['confidence']
            accuracy = 1.0 if match['similarity_score'] >= 0.9 else 0.0
            
            confidences.append(confidence)
            accuracies.append(accuracy)
        
        # Calculate calibration metrics
        calibration_analysis = {
            'confidence_accuracy_correlation': np.corrcoef(confidences, accuracies)[0, 1] if len(confidences) > 1 else 0.0,
            'mean_confidence': np.mean(confidences),
            'mean_accuracy': np.mean(accuracies),
            'calibration_bins': self._calculate_calibration_bins(confidences, accuracies)
        }
        
        return calibration_analysis
    
    def _calculate_calibration_bins(self, confidences: List[float], 
                                  accuracies: List[float], n_bins: int = 10) -> List[Dict]:
        """Calculate calibration bins for reliability diagram"""
        bins = []
        bin_size = 1.0 / n_bins
        
        for i in range(n_bins):
            bin_lower = i * bin_size
            bin_upper = (i + 1) * bin_size
            
            # Find predictions in this bin
            bin_indices = [
                j for j, conf in enumerate(confidences)
                if bin_lower <= conf < bin_upper or (i == n_bins - 1 and conf == 1.0)
            ]
            
            if bin_indices:
                bin_confidences = [confidences[j] for j in bin_indices]
                bin_accuracies = [accuracies[j] for j in bin_indices]
                
                bins.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'count': len(bin_indices),
                    'mean_confidence': np.mean(bin_confidences),
                    'mean_accuracy': np.mean(bin_accuracies),
                    'calibration_error': abs(np.mean(bin_confidences) - np.mean(bin_accuracies))
                })
        
        return bins
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report"""
        report = []
        
        # Header
        report.append("HEADING EXTRACTION EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary metrics
        metrics = evaluation_results['metrics']
        report.append("SUMMARY METRICS:")
        report.append(f"  Precision: {metrics['precision']:.3f}")
        report.append(f"  Recall:    {metrics['recall']:.3f}")
        report.append(f"  F1-Score:  {metrics['f1_score']:.3f}")
        report.append(f"  Accuracy:  {metrics['accuracy']:.3f}")
        report.append("")
        
        # Confusion matrix
        cm = evaluation_results['confusion_matrix']
        report.append("CONFUSION MATRIX:")
        report.append(f"  True Positives:  {cm['true_positives']}")
        report.append(f"  False Positives: {cm['false_positives']}")
        report.append(f"  False Negatives: {cm['false_negatives']}")
        report.append(f"  True Negatives:  {cm['true_negatives']}")
        report.append("")
        
        # Error analysis
        analysis = evaluation_results['detailed_analysis']
        report.append("ERROR ANALYSIS:")
        report.append(f"  Missed Headings: {len(analysis['missed_headings'])}")
        report.append(f"  False Positives: {len(analysis['false_positives'])}")
        report.append("")
        
        # Error patterns
        if analysis['error_patterns']:
            report.append("ERROR PATTERNS:")
            for pattern, count in analysis['error_patterns'].items():
                report.append(f"  {pattern}: {count}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        recommendations = self._generate_recommendations(evaluation_results)
        for rec in recommendations:
            report.append(f"  - {rec}")
        
        return "\n".join(report)
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on evaluation results"""
        recommendations = []
        
        metrics = evaluation_results['metrics']
        analysis = evaluation_results['detailed_analysis']
        
        # Precision-based recommendations
        if metrics['precision'] < 0.7:
            recommendations.append("Consider increasing confidence thresholds to reduce false positives")
            
            fp_patterns = analysis['error_patterns']
            if fp_patterns.get('fp_document_elements', 0) > 0:
                recommendations.append("Add filters to exclude page numbers, figures, and table references")
        
        # Recall-based recommendations
        if metrics['recall'] < 0.7:
            recommendations.append("Consider lowering confidence thresholds or improving detection sensitivity")
            
            if analysis['error_patterns'].get('missed_short_headings', 0) > 0:
                recommendations.append("Improve detection of short headings (1-2 words)")
            
            if analysis['error_patterns'].get('missed_numbered_headings', 0) > 0:
                recommendations.append("Enhance numbered heading pattern detection")
        
        # F1-score recommendations
        if metrics['f1_score'] < 0.8:
            recommendations.append("Balance precision and recall by tuning ensemble weights")
        
        # Hierarchy recommendations
        hierarchy = analysis.get('hierarchy_analysis', {})
        if hierarchy.get('level_accuracy', 0) < 0.6:
            recommendations.append("Improve heading hierarchy detection algorithms")
        
        return recommendations
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary"""
        return {
            'overall_performance': self._classify_performance(self.metrics['f1_score']),
            'strongest_metric': max(self.metrics, key=self.metrics.get),
            'weakest_metric': min(self.metrics, key=self.metrics.get),
            'total_predictions': self.confusion_matrix['true_positives'] + self.confusion_matrix['false_positives'],
            'total_ground_truth': self.confusion_matrix['true_positives'] + self.confusion_matrix['false_negatives']
        }
    
    def _classify_performance(self, f1_score: float) -> str:
        """Classify performance based on F1 score"""
        if f1_score >= 0.9:
            return "Excellent"
        elif f1_score >= 0.8:
            return "Good"
        elif f1_score >= 0.7:
            return "Fair"
        elif f1_score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str) -> bool:
        """Save evaluation results to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Evaluation results saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default evaluation configuration"""
        return {
            'match_threshold': 0.7,      # Minimum similarity for match
            'case_insensitive': True,    # Case insensitive comparison
            'remove_punctuation': True,  # Remove punctuation for comparison
            'normalize_whitespace': True, # Normalize whitespace
            'text_weight': 0.8,          # Weight for text similarity
            'position_weight': 0.2,      # Weight for position similarity
            'strict_matching': False,    # Require exact matches
            'hierarchy_evaluation': True # Evaluate hierarchy accuracy
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update evaluation configuration"""
        self.config.update(new_config)
