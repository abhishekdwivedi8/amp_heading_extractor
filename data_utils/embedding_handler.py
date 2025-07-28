"""
Embedding Handler for Text Vector Operations
Manages text embeddings for semantic analysis in heading detection.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import pickle
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

class EmbeddingHandler:
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize embedding handler"""
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), 'embeddings_cache')
        
        # Model handlers (lazy loading)
        self._model_handlers = {}
        self._available_models = {
            'minilm': None,     # Will be loaded on demand
            'fasttext': None,   # Will be loaded on demand
            'spacy': None       # Will be loaded on demand
        }
        
        # Embedding cache
        self.embedding_cache = {}
        self.cache_enabled = True
        
        # Vector operations configuration
        self.config = {
            'similarity_metric': 'cosine',
            'normalization': True,
            'dimension_reduction': None,  # PCA, None
            'batch_size': 32,
            'cache_size_limit': 10000  # Max number of cached embeddings
        }
        
        # Precomputed embeddings for common heading patterns
        self.heading_embeddings = {}
        self.heading_keywords = [
            'introduction', 'conclusion', 'methodology', 'results', 'discussion',
            'abstract', 'summary', 'overview', 'background', 'literature review',
            'data analysis', 'experimental setup', 'findings', 'recommendations',
            'chapter', 'section', 'subsection', 'appendix', 'bibliography'
        ]
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def initialize_models(self, models_to_load: Optional[List[str]] = None) -> Dict[str, bool]:
        """Initialize embedding models"""
        if models_to_load is None:
            models_to_load = list(self._available_models.keys())
        
        initialization_results = {}
        
        for model_name in models_to_load:
            try:
                if model_name == 'minilm':
                    initialization_results[model_name] = self._init_minilm()
                elif model_name == 'fasttext':
                    initialization_results[model_name] = self._init_fasttext()
                elif model_name == 'spacy':
                    initialization_results[model_name] = self._init_spacy()
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    initialization_results[model_name] = False
                    
            except Exception as e:
                logger.error(f"Failed to initialize {model_name}: {e}")
                initialization_results[model_name] = False
        
        # Precompute heading keyword embeddings
        self._precompute_heading_embeddings()
        
        return initialization_results
    
    def _init_minilm(self) -> bool:
        """Initialize MiniLM model"""
        try:
            from models.minilm.model_handler import MiniLMHandler
            handler = MiniLMHandler()
            if handler.load_model():
                self._model_handlers['minilm'] = handler
                self._available_models['minilm'] = True
                logger.info("MiniLM model initialized successfully")
                return True
        except Exception as e:
            logger.warning(f"MiniLM initialization failed: {e}")
        
        self._available_models['minilm'] = False
        return False
    
    def _init_fasttext(self) -> bool:
        """Initialize FastText model"""
        try:
            from models.fasttext.model_handler import FastTextHandler
            handler = FastTextHandler()
            # FastText needs to be trained or loaded first
            self._model_handlers['fasttext'] = handler
            self._available_models['fasttext'] = True
            logger.info("FastText model handler initialized")
            return True
        except Exception as e:
            logger.warning(f"FastText initialization failed: {e}")
        
        self._available_models['fasttext'] = False
        return False
    
    def _init_spacy(self) -> bool:
        """Initialize spaCy model"""
        try:
            from models.spacy.model_handler import SpacyHandler
            handler = SpacyHandler()
            if handler.load_model():
                self._model_handlers['spacy'] = handler
                self._available_models['spacy'] = True
                logger.info("spaCy model initialized successfully")
                return True
        except Exception as e:
            logger.warning(f"spaCy initialization failed: {e}")
        
        self._available_models['spacy'] = False
        return False
    
    def get_embeddings(self, texts: Union[str, List[str]], 
                      model_name: str = 'minilm',
                      use_cache: bool = True) -> Optional[np.ndarray]:
        """Get embeddings for texts using specified model"""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Check if model is available
        if model_name not in self._available_models or not self._available_models[model_name]:
            logger.warning(f"Model {model_name} is not available")
            return None
        
        # Try to get from cache first
        if use_cache and self.cache_enabled:
            cached_embeddings = self._get_from_cache(texts, model_name)
            if cached_embeddings is not None:
                return cached_embeddings
        
        # Generate embeddings
        embeddings = self._generate_embeddings(texts, model_name)
        
        # Cache results
        if embeddings is not None and use_cache and self.cache_enabled:
            self._add_to_cache(texts, embeddings, model_name)
        
        return embeddings
    
    def _generate_embeddings(self, texts: List[str], model_name: str) -> Optional[np.ndarray]:
        """Generate embeddings using specified model"""
        handler = self._model_handlers.get(model_name)
        if not handler:
            return None
        
        try:
            if model_name == 'minilm':
                return handler.encode_texts(texts)
            elif model_name == 'fasttext':
                # Get sentence vectors from FastText
                embeddings = []
                for text in texts:
                    vector = handler.get_sentence_vector(text)
                    if vector is not None:
                        embeddings.append(vector)
                    else:
                        # Fallback: use zero vector
                        embeddings.append(np.zeros(100))  # Default FastText dimension
                return np.array(embeddings) if embeddings else None
            elif model_name == 'spacy':
                # Process texts and get document vectors
                embeddings = []
                for text in texts:
                    doc = handler.process_text(text)
                    if doc and doc.has_vector:
                        embeddings.append(doc.vector)
                    else:
                        # Fallback: use zero vector
                        embeddings.append(np.zeros(300))  # Default spaCy vector size
                return np.array(embeddings) if embeddings else None
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings with {model_name}: {e}")
            return None
    
    def calculate_similarity(self, text1: str, text2: str, 
                           model_name: str = 'minilm') -> Optional[float]:
        """Calculate semantic similarity between two texts"""
        embeddings = self.get_embeddings([text1, text2], model_name)
        
        if embeddings is None or len(embeddings) != 2:
            return None
        
        return self._compute_similarity(embeddings[0], embeddings[1])
    
    def calculate_similarity_matrix(self, texts: List[str], 
                                  model_name: str = 'minilm') -> Optional[np.ndarray]:
        """Calculate pairwise similarity matrix for list of texts"""
        embeddings = self.get_embeddings(texts, model_name)
        
        if embeddings is None:
            return None
        
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                sim = self._compute_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric matrix
        
        return similarity_matrix
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str],
                         model_name: str = 'minilm', top_k: int = 5) -> List[Tuple[str, float, int]]:
        """Find most similar texts to query"""
        if not candidate_texts:
            return []
        
        # Get embeddings
        all_texts = [query_text] + candidate_texts
        embeddings = self.get_embeddings(all_texts, model_name)
        
        if embeddings is None:
            return []
        
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        # Calculate similarities
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            sim = self._compute_similarity(query_embedding, candidate_embedding)
            similarities.append((candidate_texts[i], sim, i))
        
        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_heading_similarity_scores(self, texts: List[str], 
                                    model_name: str = 'minilm') -> List[float]:
        """Get similarity scores against known heading patterns"""
        if not texts:
            return []
        
        # Ensure heading embeddings are computed
        if model_name not in self.heading_embeddings:
            self._precompute_heading_embeddings_for_model(model_name)
        
        heading_embeddings = self.heading_embeddings.get(model_name)
        if heading_embeddings is None:
            return [0.0] * len(texts)
        
        # Get embeddings for input texts
        text_embeddings = self.get_embeddings(texts, model_name)
        if text_embeddings is None:
            return [0.0] * len(texts)
        
        # Calculate maximum similarity to any heading keyword
        similarity_scores = []
        for text_embedding in text_embeddings:
            max_similarity = 0.0
            for heading_embedding in heading_embeddings:
                sim = self._compute_similarity(text_embedding, heading_embedding)
                max_similarity = max(max_similarity, sim)
            similarity_scores.append(max_similarity)
        
        return similarity_scores
    
    def cluster_texts_by_similarity(self, texts: List[str], 
                                   model_name: str = 'minilm',
                                   n_clusters: Optional[int] = None,
                                   similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Cluster texts by semantic similarity"""
        if len(texts) < 2:
            return {'clusters': [texts], 'n_clusters': 1}
        
        # Get embeddings
        embeddings = self.get_embeddings(texts, model_name)
        if embeddings is None:
            return {'clusters': [[text] for text in texts], 'n_clusters': len(texts)}
        
        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Determine number of clusters
            if n_clusters is None:
                # Use similarity threshold to estimate clusters
                similarity_matrix = cosine_similarity(embeddings)
                n_clusters = self._estimate_num_clusters(similarity_matrix, similarity_threshold)
            
            # Perform clustering
            if n_clusters >= len(texts):
                # Each text in its own cluster
                clusters = [[text] for text in texts]
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='average',
                    metric='cosine'
                )
                cluster_labels = clustering.fit_predict(embeddings)
                
                # Group texts by cluster
                clusters = defaultdict(list)
                for text, label in zip(texts, cluster_labels):
                    clusters[label].append(text)
                
                clusters = list(clusters.values())
            
            return {
                'clusters': clusters,
                'n_clusters': len(clusters),
                'cluster_sizes': [len(cluster) for cluster in clusters]
            }
            
        except ImportError:
            logger.warning("scikit-learn not available for clustering")
            return {'clusters': [texts], 'n_clusters': 1}
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {'clusters': [[text] for text in texts], 'n_clusters': len(texts)}
    
    def _estimate_num_clusters(self, similarity_matrix: np.ndarray, 
                              threshold: float) -> int:
        """Estimate number of clusters based on similarity threshold"""
        n = similarity_matrix.shape[0]
        
        # Count pairs above threshold
        high_similarity_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    high_similarity_pairs += 1
        
        # Estimate clusters (heuristic)
        total_pairs = n * (n - 1) // 2
        if total_pairs == 0:
            return 1
        
        similarity_ratio = high_similarity_pairs / total_pairs
        
        if similarity_ratio > 0.7:
            return max(1, n // 4)  # Few large clusters
        elif similarity_ratio > 0.3:
            return max(1, n // 2)  # Medium clusters
        else:
            return n  # Many small clusters
    
    def reduce_dimensions(self, embeddings: np.ndarray, 
                         target_dim: int = 50,
                         method: str = 'pca') -> Optional[np.ndarray]:
        """Reduce dimensionality of embeddings"""
        if embeddings.shape[1] <= target_dim:
            return embeddings
        
        try:
            if method.lower() == 'pca':
                from sklearn.decomposition import PCA
                pca = PCA(n_components=target_dim)
                reduced_embeddings = pca.fit_transform(embeddings)
                return reduced_embeddings
            else:
                logger.warning(f"Unknown dimension reduction method: {method}")
                return embeddings
                
        except ImportError:
            logger.warning("scikit-learn not available for dimension reduction")
            return embeddings
        except Exception as e:
            logger.error(f"Dimension reduction failed: {e}")
            return embeddings
    
    def _compute_similarity(self, embedding1: np.ndarray, 
                           embedding2: np.ndarray) -> float:
        """Compute similarity between two embeddings"""
        if self.config['similarity_metric'] == 'cosine':
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        elif self.config['similarity_metric'] == 'euclidean':
            # Convert Euclidean distance to similarity
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)
        
        else:
            # Default to cosine
            return self._compute_similarity(embedding1, embedding2)
    
    def _precompute_heading_embeddings(self):
        """Precompute embeddings for heading keywords"""
        for model_name in self._available_models:
            if self._available_models[model_name]:
                self._precompute_heading_embeddings_for_model(model_name)
    
    def _precompute_heading_embeddings_for_model(self, model_name: str):
        """Precompute heading embeddings for specific model"""
        if model_name not in self._available_models or not self._available_models[model_name]:
            return
        
        try:
            embeddings = self.get_embeddings(self.heading_keywords, model_name, use_cache=False)
            if embeddings is not None:
                self.heading_embeddings[model_name] = embeddings
                logger.debug(f"Precomputed heading embeddings for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to precompute heading embeddings for {model_name}: {e}")
    
    def _get_from_cache(self, texts: List[str], model_name: str) -> Optional[np.ndarray]:
        """Get embeddings from cache"""
        cache_key = self._generate_cache_key(texts, model_name)
        return self.embedding_cache.get(cache_key)
    
    def _add_to_cache(self, texts: List[str], embeddings: np.ndarray, model_name: str):
        """Add embeddings to cache"""
        if len(self.embedding_cache) >= self.config['cache_size_limit']:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.embedding_cache.keys())[:len(self.embedding_cache) // 2]
            for key in keys_to_remove:
                del self.embedding_cache[key]
        
        cache_key = self._generate_cache_key(texts, model_name)
        self.embedding_cache[cache_key] = embeddings
    
    def _generate_cache_key(self, texts: List[str], model_name: str) -> str:
        """Generate cache key for texts and model"""
        import hashlib
        combined_text = '||'.join(texts)
        text_hash = hashlib.md5(combined_text.encode()).hexdigest()
        return f"{model_name}_{text_hash}"
    
    def save_cache(self, filepath: Optional[str] = None) -> bool:
        """Save embedding cache to disk"""
        if filepath is None:
            filepath = os.path.join(self.cache_dir, 'embedding_cache.pkl')
        
        try:
            cache_data = {
                'embedding_cache': self.embedding_cache,
                'heading_embeddings': self.heading_embeddings,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Embedding cache saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
            return False
    
    def load_cache(self, filepath: Optional[str] = None) -> bool:
        """Load embedding cache from disk"""
        if filepath is None:
            filepath = os.path.join(self.cache_dir, 'embedding_cache.pkl')
        
        if not os.path.exists(filepath):
            logger.info("No embedding cache file found")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.embedding_cache = cache_data.get('embedding_cache', {})
            self.heading_embeddings = cache_data.get('heading_embeddings', {})
            saved_config = cache_data.get('config', {})
            
            # Update config with saved settings
            self.config.update(saved_config)
            
            logger.info(f"Embedding cache loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding cache: {e}")
            return False
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        self.heading_embeddings.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.embedding_cache),
            'cache_size_limit': self.config['cache_size_limit'],
            'heading_embeddings_models': list(self.heading_embeddings.keys()),
            'available_models': {k: v for k, v in self._available_models.items() if v},
            'cache_enabled': self.cache_enabled
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        model_info = {}
        
        for model_name, is_available in self._available_models.items():
            info = {
                'available': is_available,
                'loaded': model_name in self._model_handlers
            }
            
            if model_name in self._model_handlers:
                handler = self._model_handlers[model_name]
                if hasattr(handler, 'get_model_info'):
                    info.update(handler.get_model_info())
            
            model_info[model_name] = info
        
        return model_info
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update embedding handler configuration"""
        self.config.update(new_config)
        
        # Apply configuration changes
        if 'cache_enabled' in new_config:
            self.cache_enabled = new_config['cache_enabled']
    
    def extract_embedding_features(self, texts: List[str], 
                                 model_name: str = 'minilm') -> Optional[Dict[str, Any]]:
        """Extract embedding-based features for heading detection"""
        if not texts:
            return None
        
        embeddings = self.get_embeddings(texts, model_name)
        if embeddings is None:
            return None
        
        # Calculate various embedding-based features
        features = {
            'embeddings': embeddings,
            'embedding_means': np.mean(embeddings, axis=1),
            'embedding_stds': np.std(embeddings, axis=1),
            'embedding_norms': np.linalg.norm(embeddings, axis=1),
            'pairwise_similarities': self.calculate_similarity_matrix(texts, model_name),
            'heading_similarity_scores': self.get_heading_similarity_scores(texts, model_name)
        }
        
        # Add clustering information
        if len(texts) > 1:
            clustering_result = self.cluster_texts_by_similarity(texts, model_name)
            features['clustering'] = clustering_result
        
        return features
    
    def __del__(self):
        """Cleanup when handler is destroyed"""
        # Save cache before destruction
        if hasattr(self, 'cache_enabled') and self.cache_enabled:
            try:
                self.save_cache()
            except:
                pass  # Ignore errors during cleanup
