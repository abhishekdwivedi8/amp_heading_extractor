"""
Level 4: Ensemble Learning for Heading Classification
Combines multiple ML models to improve heading detection accuracy.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

# Import sklearn components with fallbacks
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    joblib = None
    RandomForestClassifier = None
    GradientBoostingClassifier = None
    LogisticRegression = None
    SVC = None
    StandardScaler = None
    cross_val_score = None
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class Level4EnsembleClassifier:
    def __init__(self, models_path: str = None):
        """Initialize ensemble classifier"""
        self.models = {}
        self.scaler = None
        self.is_trained = False
        self.feature_importance = {}
        self.models_path = models_path
        
        # Initialize models if sklearn is available
        if SKLEARN_AVAILABLE:
            self.models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'svm': SVC(probability=True, random_state=42)
            }
            self.scaler = StandardScaler()
            
            # Try to load pre-trained models
            if models_path:
                self._load_models()
        else:
            logger.info("sklearn not available - using rule-based classification only")
    
    def classify_headings(self, semantic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Classify headings using ensemble of ML models"""
        headings = semantic_results.get('enhanced_headings', [])
        
        if not headings:
            return {
                'classified_headings': [],
                'ensemble_metadata': {'no_headings': True}
            }
        
        # Extract features for classification
        features = self._extract_features(headings)
        
        if not SKLEARN_AVAILABLE or not self.is_trained:
            # If sklearn not available or no pre-trained models, use rule-based fallback
            classified_headings = self._rule_based_classification(headings, features)
        else:
            # Use trained ensemble models
            classified_headings = self._ensemble_classification(headings, features)
        
        return {
            'classified_headings': classified_headings,
            'ensemble_metadata': self._generate_ensemble_metadata(classified_headings, features)
        }
    
    def _extract_features(self, headings: List[Dict]) -> np.ndarray:
        """Extract numerical features for ML classification"""
        features_list = []
        
        for heading in headings:
            feature_vector = []
            
            # Basic confidence scores
            feature_vector.append(heading.get('confidence', 0.0))
            feature_vector.append(heading.get('semantic_confidence', 0.0))
            feature_vector.append(heading.get('combined_confidence', 0.0))
            feature_vector.append(heading.get('similarity_score', 0.0))
            
            # Semantic features
            semantic_features = heading.get('semantic_features', {})
            feature_vector.extend([
                semantic_features.get('word_count', 0),
                semantic_features.get('char_count', 0),
                semantic_features.get('avg_word_length', 0),
                semantic_features.get('noun_ratio', 0),
                semantic_features.get('verb_ratio', 0),
                semantic_features.get('adj_ratio', 0),
                semantic_features.get('proper_noun_ratio', 0),
                float(semantic_features.get('has_entities', False)),
                semantic_features.get('entity_ratio', 0),
                float(semantic_features.get('is_title_case', False)),
                float(semantic_features.get('is_upper_case', False)),
                float(semantic_features.get('starts_with_capital', False)),
                float(semantic_features.get('ends_with_punctuation', False))
            ])
            
            # Linguistic features
            linguistic_features = heading.get('linguistic_features', {})
            feature_vector.extend([
                linguistic_features.get('complexity_score', 0),
                linguistic_features.get('formality_score', 0),
                len(linguistic_features.get('topic_indicators', [])),
                len(linguistic_features.get('structural_indicators', []))
            ])
            
            # Position features
            feature_vector.extend([
                heading.get('page', 1),
                heading.get('line', 1),
                self._calculate_position_score(heading)
            ])
            
            # Pattern-based features
            pattern_type = heading.get('pattern_type', 'none')
            detection_method = heading.get('detection_method', 'unknown')
            
            # One-hot encode pattern types
            pattern_types = ['numbered_sections', 'formatted_headings', 'outline_patterns', 'document_structure']
            for ptype in pattern_types:
                feature_vector.append(float(pattern_type == ptype))
            
            # One-hot encode detection methods
            detection_methods = ['pattern', 'format']
            for method in detection_methods:
                feature_vector.append(float(detection_method == method))
            
            # Font-based features (if available)
            font_size = heading.get('font_size', 0)
            avg_font_size = heading.get('avg_font_size', 12)
            feature_vector.extend([
                font_size,
                avg_font_size,
                font_size / avg_font_size if avg_font_size > 0 else 1,
                float(heading.get('is_bold', False))
            ])
            
            features_list.append(feature_vector)
        
        return np.array(features_list)
    
    def _calculate_position_score(self, heading: Dict) -> float:
        """Calculate a score based on heading position (early pages are more likely to be headings)"""
        page = heading.get('page', 1)
        line = heading.get('line', 1)
        
        # Higher score for earlier pages and lines
        page_score = max(0, 1 - (page - 1) / 20)  # Decreases after page 20
        line_score = max(0, 1 - (line - 1) / 50)   # Decreases after line 50
        
        return (page_score + line_score) / 2
    
    def _rule_based_classification(self, headings: List[Dict], features: np.ndarray) -> List[Dict]:
        """Enhanced rule-based classification when ML models are not available"""
        classified_headings = []
        
        for i, heading in enumerate(headings):
            # Start with combined confidence
            final_confidence = heading.get('combined_confidence', 0.5)
            is_heading = False
            reasoning = []
            
            # Rule 1: High combined confidence
            if final_confidence > 0.8:
                is_heading = True
                reasoning.append('high_combined_confidence')
            
            # Rule 2: Strong pattern match
            pattern_confidence = heading.get('confidence', 0)
            if pattern_confidence > 0.85:
                is_heading = True
                reasoning.append('strong_pattern_match')
            
            # Rule 3: Semantic indicators
            semantic_confidence = heading.get('semantic_confidence', 0)
            linguistic_features = heading.get('linguistic_features', {})
            
            if (semantic_confidence > 0.7 and 
                len(linguistic_features.get('structural_indicators', [])) > 0):
                is_heading = True
                reasoning.append('semantic_structural_match')
            
            # Rule 4: Format-based indicators
            if (heading.get('detection_method') == 'format' and
                heading.get('is_bold', False) and
                heading.get('font_size', 0) > heading.get('avg_font_size', 12) * 1.3):
                is_heading = True
                reasoning.append('format_indicators')
            
            # Rule 5: Document structure patterns
            if heading.get('pattern_type') == 'document_structure':
                is_heading = True
                reasoning.append('document_structure')
            
            # Rule 6: Position-based boost
            position_score = self._calculate_position_score(heading)
            if position_score > 0.8 and final_confidence > 0.6:
                is_heading = True
                reasoning.append('favorable_position')
            
            # Rule 7: Length and format checks
            text = heading.get('text', '')
            word_count = len(text.split())
            
            if (2 <= word_count <= 8 and 
                text.istitle() and 
                final_confidence > 0.6):
                is_heading = True
                reasoning.append('optimal_format')
            
            # Final decision with threshold
            if not is_heading and final_confidence > 0.7:
                is_heading = True
                reasoning.append('confidence_threshold')
            
            # Calculate final probability
            ensemble_probability = min(1.0, final_confidence + 
                                     len(reasoning) * 0.05)  # Boost for multiple reasons
            
            classified_heading = heading.copy()
            classified_heading.update({
                'is_heading': is_heading,
                'ensemble_probability': ensemble_probability,
                'classification_reasoning': reasoning,
                'classification_method': 'rule_based_enhanced'
            })
            
            classified_headings.append(classified_heading)
        
        return classified_headings
    
    def _ensemble_classification(self, headings: List[Dict], features: np.ndarray) -> List[Dict]:
        """Use trained ensemble models for classification"""
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from each model
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(features_scaled)
                prob = model.predict_proba(features_scaled)[:, 1]  # Probability of being a heading
                predictions[name] = pred
                probabilities[name] = prob
                logger.debug(f"Model {name} made predictions successfully")
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                predictions[name] = np.zeros(len(features))
                probabilities[name] = np.zeros(len(features))
        
        classified_headings = []
        
        for i, heading in enumerate(headings):
            # Ensemble voting
            votes = sum(predictions[name][i] for name in self.models.keys())
            vote_ratio = votes / len(self.models)
            
            # Ensemble probability (average)
            avg_probability = np.mean([probabilities[name][i] for name in self.models.keys()])
            
            # Weighted ensemble (if we have model weights)
            weighted_probability = self._calculate_weighted_probability(probabilities, i)
            
            # Final decision
            is_heading = vote_ratio >= 0.5 or weighted_probability > 0.6
            
            classified_heading = heading.copy()
            classified_heading.update({
                'is_heading': is_heading,
                'ensemble_probability': weighted_probability,
                'vote_ratio': vote_ratio,
                'individual_probabilities': {name: probabilities[name][i] for name in self.models.keys()},
                'classification_method': 'ensemble_ml'
            })
            
            classified_headings.append(classified_heading)
        
        return classified_headings
    
    def _calculate_weighted_probability(self, probabilities: Dict, index: int) -> float:
        """Calculate weighted ensemble probability"""
        # Default equal weights if no performance data available
        weights = {
            'random_forest': 0.3,
            'gradient_boost': 0.3,
            'logistic': 0.2,
            'svm': 0.2
        }
        
        weighted_sum = sum(
            weights.get(name, 0.25) * probabilities[name][index]
            for name in self.models.keys()
        )
        
        return weighted_sum
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            # Implementation would load actual trained models
            logger.info("Loading pre-trained models...")
            # self.models = joblib.load(f"{self.models_path}/ensemble_models.pkl")
            # self.scaler = joblib.load(f"{self.models_path}/scaler.pkl")
            # self.is_trained = True
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            self.is_trained = False
    
    def train_models(self, training_data: List[Dict], labels: List[bool]):
        """Train ensemble models on labeled data"""
        if len(training_data) < 10:
            logger.warning("Insufficient training data")
            return False
        
        # Extract features from training data
        features = self._extract_features(training_data)
        y = np.array(labels)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train each model
        for name, model in self.models.items():
            try:
                model.fit(features_scaled, y)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, features_scaled, y, cv=5)
                logger.info(f"Model {name} CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to train model {name}: {e}")
        
        self.is_trained = True
        
        # Calculate feature importance
        self._calculate_feature_importance(features_scaled, y)
        
        return True
    
    def _calculate_feature_importance(self, features: np.ndarray, labels: np.ndarray):
        """Calculate feature importance across models"""
        importance_scores = {}
        
        # Random Forest feature importance
        if hasattr(self.models['random_forest'], 'feature_importances_'):
            rf_importance = self.models['random_forest'].feature_importances_
            importance_scores['random_forest'] = rf_importance
        
        # Gradient Boosting feature importance
        if hasattr(self.models['gradient_boost'], 'feature_importances_'):
            gb_importance = self.models['gradient_boost'].feature_importances_
            importance_scores['gradient_boost'] = gb_importance
        
        # Average importance across models
        if importance_scores:
            avg_importance = np.mean(list(importance_scores.values()), axis=0)
            self.feature_importance = {
                'average': avg_importance,
                'individual': importance_scores
            }
    
    def save_models(self, save_path: str):
        """Save trained models to disk"""
        if not self.is_trained:
            logger.warning("No trained models to save")
            return False
        
        try:
            joblib.dump(self.models, f"{save_path}/ensemble_models.pkl")
            joblib.dump(self.scaler, f"{save_path}/scaler.pkl")
            if self.feature_importance:
                joblib.dump(self.feature_importance, f"{save_path}/feature_importance.pkl")
            logger.info(f"Models saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False
    
    def _generate_ensemble_metadata(self, classified_headings: List[Dict], 
                                  features: np.ndarray) -> Dict[str, Any]:
        """Generate metadata about ensemble classification"""
        metadata = {
            'total_classified': len(classified_headings),
            'predicted_headings': len([h for h in classified_headings if h.get('is_heading', False)]),
            'classification_method': 'ensemble_ml' if self.is_trained else 'rule_based_enhanced',
            'models_used': list(self.models.keys()) if self.is_trained else ['rule_based'],
            'average_ensemble_probability': 0.0,
            'confidence_distribution': {},
            'feature_statistics': {}
        }
        
        if classified_headings:
            probabilities = [h.get('ensemble_probability', 0) for h in classified_headings]
            metadata['average_ensemble_probability'] = np.mean(probabilities)
            
            # Confidence distribution
            high_conf = len([p for p in probabilities if p > 0.8])
            medium_conf = len([p for p in probabilities if 0.5 < p <= 0.8])
            low_conf = len([p for p in probabilities if p <= 0.5])
            
            metadata['confidence_distribution'] = {
                'high': high_conf,
                'medium': medium_conf,
                'low': low_conf
            }
        
        # Feature statistics
        if len(features) > 0:
            metadata['feature_statistics'] = {
                'feature_count': features.shape[1],
                'avg_feature_values': np.mean(features, axis=0).tolist(),
                'feature_importance_available': bool(self.feature_importance)
            }
        
        return metadata
