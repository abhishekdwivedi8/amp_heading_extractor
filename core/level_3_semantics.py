"""
Level 3: Semantic Analysis for Heading Detection
Uses NLP models to understand semantic context and improve heading detection accuracy.
"""
import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple
# Import with fallbacks for missing dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    cosine_similarity = None
    SKLEARN_AVAILABLE = False
import logging

logger = logging.getLogger(__name__)

class Level3SemanticAnalyzer:
    def __init__(self, model_path: str = 'all-MiniLM-L6-v2'):
        """Initialize semantic analyzer with embedding model"""
        self.embedding_model = None
        self.nlp = None
        
        # Use pre-loaded model from StartupOptimizer for instant access
        try:
            from startup_optimizer import StartupOptimizer
            self.embedding_model = StartupOptimizer.get_cached_sentence_transformer()
            if self.embedding_model:
                logger.info(f"Using pre-loaded SentenceTransformer model")
            else:
                logger.warning("SentenceTransformer not available from cache")
        except Exception as e:
            logger.warning(f"Failed to get cached SentenceTransformer: {e}")
            self.embedding_model = None
        
        # Use pre-loaded spaCy model from StartupOptimizer
        try:
            from startup_optimizer import StartupOptimizer
            self.nlp = StartupOptimizer.get_cached_spacy_model()
            if self.nlp:
                logger.info("Using pre-loaded spaCy model")
            else:
                logger.warning("spaCy model not available from cache")
        except Exception as e:
            logger.warning(f"Failed to get cached spaCy: {e}")
            self.nlp = None
        
        # Semantic patterns for headings
        self.heading_keywords = [
            'introduction', 'conclusion', 'methodology', 'results', 'discussion',
            'abstract', 'summary', 'overview', 'background', 'literature review',
            'data analysis', 'experimental setup', 'findings', 'recommendations',
            'chapter', 'section', 'subsection', 'appendix', 'bibliography',
            'table of contents', 'acknowledgments', 'references'
        ]
        
        # Use pre-computed embeddings from StartupOptimizer for instant access
        try:
            from startup_optimizer import StartupOptimizer
            keyword_data = StartupOptimizer.get_cached_embeddings()
            if keyword_data:
                self.heading_keywords = keyword_data['keywords']
                self.heading_embeddings = keyword_data['embeddings']
                logger.info("Using pre-computed keyword embeddings")
            else:
                self.heading_embeddings = None
        except Exception as e:
            logger.warning(f"Failed to get cached embeddings: {e}")
            self.heading_embeddings = None
    
    def analyze_semantics(self, pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic analysis with robust fallback mechanisms"""
        headings = pattern_results.get('detected_headings', [])
        
        if not headings:
            return {
                'enhanced_headings': [],
                'semantic_scores': [],
                'semantic_metadata': {'no_headings': True}
            }
        
        # Use fallback mode if models unavailable
        if not self.embedding_model or os.environ.get('SEMANTIC_FALLBACK_MODE'):
            return self._fallback_semantic_analysis(headings)
        
        enhanced_headings = []
        semantic_scores = []
        
        for heading in headings:
            text = heading['text']
            
            # Calculate semantic features
            semantic_features = self._calculate_semantic_features(text)
            
            # Calculate semantic similarity to known heading patterns
            similarity_score = self._calculate_heading_similarity(text)
            
            # Analyze linguistic features
            linguistic_features = self._analyze_linguistic_features(text)
            
            # Combine scores to get overall semantic confidence
            semantic_confidence = self._calculate_semantic_confidence(
                semantic_features, similarity_score, linguistic_features
            )
            
            # Update heading with semantic information
            enhanced_heading = heading.copy()
            enhanced_heading.update({
                'semantic_confidence': semantic_confidence,
                'similarity_score': similarity_score,
                'semantic_features': semantic_features,
                'linguistic_features': linguistic_features,
                'combined_confidence': self._combine_confidences(
                    heading.get('confidence', 0.5), semantic_confidence
                )
            })
            
            enhanced_headings.append(enhanced_heading)
            semantic_scores.append(semantic_confidence)
        
        return {
            'enhanced_headings': enhanced_headings,
            'semantic_scores': semantic_scores,
            'semantic_metadata': self._generate_semantic_metadata(semantic_scores, enhanced_headings)
        }
    
    def _fallback_semantic_analysis(self, headings: List[Dict]) -> Dict[str, Any]:
        """Fallback semantic analysis using rule-based approach"""
        enhanced_headings = []
        semantic_scores = []
        
        for heading in headings:
            text = heading['text'].lower()
            
            # Rule-based semantic confidence
            semantic_confidence = 0.5  # Default
            
            # Boost for structural keywords
            structural_keywords = [
                'introduction', 'conclusion', 'summary', 'abstract', 'overview',
                'chapter', 'section', 'subsection', 'appendix', 'references',
                'methodology', 'results', 'discussion', 'background'
            ]
            
            if any(keyword in text for keyword in structural_keywords):
                semantic_confidence = 0.8
            
            # Boost for numbered sections
            if re.match(r'^\d+(\.\d+)*\s+', heading['text']):
                semantic_confidence = max(semantic_confidence, 0.7)
            
            # Basic linguistic features without spaCy
            linguistic_features = {
                'word_count': len(text.split()),
                'char_count': len(text),
                'is_title_case': heading['text'].istitle(),
                'starts_with_capital': heading['text'][0].isupper() if heading['text'] else False,
                'topic_indicators': [kw for kw in structural_keywords if kw in text],
                'formality_score': 0.6 if any(kw in text for kw in structural_keywords) else 0.4
            }
            
            enhanced_heading = heading.copy()
            enhanced_heading.update({
                'semantic_confidence': semantic_confidence,
                'similarity_score': semantic_confidence,  # Use same value
                'linguistic_features': linguistic_features,
                'combined_confidence': (heading.get('confidence', 0.5) + semantic_confidence) / 2
            })
            
            enhanced_headings.append(enhanced_heading)
            semantic_scores.append(semantic_confidence)
        
        return {
            'enhanced_headings': enhanced_headings,
            'semantic_scores': semantic_scores,
            'semantic_metadata': {
                'model_available': False,
                'fallback_mode': True,
                'total_analyzed': len(semantic_scores)
            }
        }
    
    def _calculate_semantic_features(self, text: str) -> Dict[str, float]:
        """Calculate various semantic features for the text"""
        features = {}
        
        if not self.nlp:
            return features
        
        doc = self.nlp(text)
        
        # Length features
        features['word_count'] = len(doc)
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(token.text) for token in doc]) if doc else 0
        
        # POS tag features
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        
        total_tokens = len(doc)
        if total_tokens > 0:
            features['noun_ratio'] = pos_counts.get('NOUN', 0) / total_tokens
            features['verb_ratio'] = pos_counts.get('VERB', 0) / total_tokens
            features['adj_ratio'] = pos_counts.get('ADJ', 0) / total_tokens
            features['proper_noun_ratio'] = pos_counts.get('PROPN', 0) / total_tokens
        
        # Named entity features
        features['has_entities'] = len(doc.ents) > 0
        features['entity_ratio'] = len(doc.ents) / total_tokens if total_tokens > 0 else 0
        
        # Dependency features
        features['has_root'] = any(token.dep_ == 'ROOT' for token in doc)
        
        # Text characteristics
        features['is_title_case'] = text.istitle()
        features['is_upper_case'] = text.isupper()
        features['starts_with_capital'] = text[0].isupper() if text else False
        features['ends_with_punctuation'] = text.rstrip()[-1] in '.!?' if text.rstrip() else False
        
        return features
    
    def _calculate_heading_similarity(self, text: str) -> float:
        """Calculate similarity to known heading patterns"""
        if not self.embedding_model or self.heading_embeddings is None or cosine_similarity is None:
            # Fallback: simple keyword matching
            text_lower = text.lower()
            max_similarity = 0.0
            for keyword in self.heading_keywords:
                if keyword in text_lower:
                    # Simple similarity based on keyword presence
                    similarity = len(keyword) / len(text_lower) if text_lower else 0
                    max_similarity = max(max_similarity, similarity)
            return min(1.0, max_similarity * 2)  # Scale up for keyword matches
        
        try:
            text_embedding = self.embedding_model.encode([text])
            similarities = cosine_similarity(text_embedding, self.heading_embeddings)[0]
            return float(np.max(similarities))
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            # Fallback to keyword matching
            text_lower = text.lower()
            for keyword in self.heading_keywords:
                if keyword in text_lower:
                    return 0.7  # High confidence for keyword match
            return 0.0
    
    def _analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic characteristics"""
        features = {
            'complexity_score': 0.0,
            'formality_score': 0.0,
            'topic_indicators': [],
            'structural_indicators': []
        }
        
        if not self.nlp:
            return features
        
        doc = self.nlp(text)
        
        # Complexity based on sentence structure
        if len(doc) > 0:
            avg_dependency_depth = np.mean([
                len(list(token.ancestors)) for token in doc
            ])
            features['complexity_score'] = min(1.0, avg_dependency_depth / 5.0)
        
        # Formality indicators
        formal_indicators = ['methodology', 'analysis', 'evaluation', 'assessment']
        informal_indicators = ['stuff', 'things', 'gets', 'lots']
        
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_indicators if word in text_lower)
        informal_count = sum(1 for word in informal_indicators if word in text_lower)
        
        if formal_count + informal_count > 0:
            features['formality_score'] = formal_count / (formal_count + informal_count)
        
        # Topic indicators
        academic_topics = ['research', 'study', 'analysis', 'method', 'result']
        technical_topics = ['system', 'algorithm', 'implementation', 'design']
        
        for topic in academic_topics + technical_topics:
            if topic in text_lower:
                features['topic_indicators'].append(topic)
        
        # Structural indicators
        structural_words = ['chapter', 'section', 'part', 'appendix', 'conclusion']
        for word in structural_words:
            if word in text_lower:
                features['structural_indicators'].append(word)
        
        return features
    
    def _calculate_semantic_confidence(self, semantic_features: Dict, 
                                     similarity_score: float, 
                                     linguistic_features: Dict) -> float:
        """Calculate overall semantic confidence score"""
        confidence = 0.0
        
        # Base similarity score (40% weight)
        confidence += similarity_score * 0.4
        
        # Length appropriateness (20% weight)
        word_count = semantic_features.get('word_count', 0)
        if 2 <= word_count <= 10:  # Ideal heading length
            confidence += 0.2
        elif word_count <= 15:
            confidence += 0.1
        
        # Linguistic features (25% weight)
        formality = linguistic_features.get('formality_score', 0)
        confidence += formality * 0.1
        
        if linguistic_features.get('structural_indicators'):
            confidence += 0.15
        
        # Text formatting (15% weight)
        if semantic_features.get('is_title_case'):
            confidence += 0.1
        elif semantic_features.get('starts_with_capital'):
            confidence += 0.05
        
        # Penalize if ends with punctuation (except colons)
        if (semantic_features.get('ends_with_punctuation') and 
            not semantic_features.get('text', '').rstrip().endswith(':')):
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _combine_confidences(self, pattern_confidence: float, 
                           semantic_confidence: float) -> float:
        """Combine pattern and semantic confidence scores"""
        # Weighted combination: pattern 60%, semantic 40%
        combined = (pattern_confidence * 0.6) + (semantic_confidence * 0.4)
        
        # Boost if both are high
        if pattern_confidence > 0.8 and semantic_confidence > 0.8:
            combined = min(1.0, combined + 0.1)
        
        # Penalize if there's large disagreement
        if abs(pattern_confidence - semantic_confidence) > 0.5:
            combined *= 0.9
        
        return combined
    
    def _generate_semantic_metadata(self, scores: List[float], 
                                  headings: List[Dict]) -> Dict[str, Any]:
        """Generate metadata about semantic analysis"""
        metadata = {
            'model_available': self.embedding_model is not None,
            'total_analyzed': len(scores),
            'average_semantic_score': np.mean(scores) if scores else 0,
            'semantic_distribution': self._get_score_distribution(scores),
            'top_semantic_features': self._get_top_features(headings)
        }
        
        return metadata
    
    def _get_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Get distribution of semantic scores"""
        if not scores:
            return {'high': 0, 'medium': 0, 'low': 0}
        
        high = len([s for s in scores if s > 0.7])
        medium = len([s for s in scores if 0.4 < s <= 0.7])
        low = len([s for s in scores if s <= 0.4])
        
        return {'high': high, 'medium': medium, 'low': low}
    
    def _get_top_features(self, headings: List[Dict]) -> Dict[str, Any]:
        """Extract top semantic features from headings"""
        features = {
            'common_topics': [],
            'structural_elements': [],
            'avg_formality': 0.0
        }
        
        if not headings:
            return features
        
        # Collect topic indicators
        all_topics = []
        all_structural = []
        formality_scores = []
        
        for heading in headings:
            linguistic = heading.get('linguistic_features', {})
            all_topics.extend(linguistic.get('topic_indicators', []))
            all_structural.extend(linguistic.get('structural_indicators', []))
            formality_scores.append(linguistic.get('formality_score', 0))
        
        # Get most common topics
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        features['common_topics'] = sorted(
            topic_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        # Get structural elements
        structural_counts = {}
        for element in all_structural:
            structural_counts[element] = structural_counts.get(element, 0) + 1
        
        features['structural_elements'] = sorted(
            structural_counts.items(), key=lambda x: x[1], reverse=True
        )
        
        # Average formality
        features['avg_formality'] = np.mean(formality_scores) if formality_scores else 0
        
        return features
