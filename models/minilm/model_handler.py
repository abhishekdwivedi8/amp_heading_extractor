"""
MiniLM Sentence Transformer Model Handler
Provides efficient sentence embeddings for heading extraction pipeline.
"""

import os
import logging
import numpy as np
import pickle
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class MiniLMHandler:
    """
    Handler for MiniLM sentence transformer model optimized for heading extraction.
    Supports both online and offline usage with memory-efficient operations.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize MiniLM handler.
        
        Args:
            model_name: HuggingFace model name or local path
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "cache")
        self.model: Optional[SentenceTransformer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
        # Model configuration
        self.config = {
            "max_seq_length": 512,
            "batch_size": 32,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False
        }
        
    def load_model(self, force_download: bool = False) -> bool:
        """
        Load the MiniLM model with error handling and fallbacks.
        
        Args:
            force_download: Force re-download of model
            
        Returns:
            bool: True if model loaded successfully
        """
        if self.is_loaded and not force_download:
            return True
            
        if SentenceTransformer is None:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            return False
            
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Try loading from cache first
            cached_model_path = os.path.join(self.cache_dir, self.model_name.replace("/", "_"))
            
            if os.path.exists(cached_model_path) and not force_download:
                logger.info(f"Loading cached model from {cached_model_path}")
                self.model = SentenceTransformer(cached_model_path, device=self.device)
            else:
                logger.info(f"Downloading model {self.model_name}")
                self.model = SentenceTransformer(self.model_name, device=self.device, cache_folder=self.cache_dir)
                
                # Save to cache
                if cached_model_path != self.model_name:
                    self.model.save(cached_model_path)
                    
            # Configure model settings
            self.model.max_seq_length = self.config["max_seq_length"]
            self.is_loaded = True
            
            logger.info(f"MiniLM model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MiniLM model: {e}")
            self.model = None
            self.is_loaded = False
            return False
    
    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: Optional[int] = None,
               show_progress: bool = False) -> Optional[np.ndarray]:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings or None if failed
        """
        if not self.is_loaded and not self.load_model():
            logger.error("Cannot encode: model not loaded")
            return None
            
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            batch_size = batch_size or self.config["batch_size"]
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.config["normalize_embeddings"],
                convert_to_numpy=self.config["convert_to_numpy"],
                show_progress_bar=show_progress
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            return None
    
    def similarity(self, text1: str, text2: str) -> Optional[float]:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1, or None if failed
        """
        try:
            embeddings = self.encode([text1, text2])
            if embeddings is None or len(embeddings) != 2:
                return None
                
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1])
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return None
    
    def batch_similarity(self, 
                        query: str, 
                        candidates: List[str],
                        top_k: Optional[int] = None) -> List[tuple]:
        """
        Find most similar texts to query from candidates.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (text, similarity_score) tuples sorted by similarity
        """
        try:
            all_texts = [query] + candidates
            embeddings = self.encode(all_texts)
            
            if embeddings is None:
                return []
                
            query_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]
            
            # Calculate similarities
            similarities = np.dot(candidate_embeddings, query_embedding)
            
            # Create results with indices
            results = [(candidates[i], float(similarities[i])) for i in range(len(candidates))]
            results.sort(key=lambda x: x[1], reverse=True)
            
            if top_k:
                results = results[:top_k]
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate batch similarity: {e}")
            return []
    
    def extract_heading_features(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract heading-specific features using embeddings.
        
        Args:
            text: Text to analyze
            context: Optional surrounding context
            
        Returns:
            Dictionary of features for heading classification
        """
        try:
            features = {}
            
            # Basic embedding
            embedding = self.encode(text)
            if embedding is not None:
                features['embedding'] = embedding.flatten()
                features['embedding_norm'] = np.linalg.norm(embedding)
                features['embedding_mean'] = np.mean(embedding)
                features['embedding_std'] = np.std(embedding)
            
            # Context similarity if provided
            if context:
                similarity = self.similarity(text, context)
                if similarity is not None:
                    features['context_similarity'] = similarity
            
            # Text characteristics
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract heading features: {e}")
            return {}
    
    def save_embeddings(self, texts: List[str], filepath: str) -> bool:
        """
        Save pre-computed embeddings to file.
        
        Args:
            texts: List of texts
            filepath: Path to save embeddings
            
        Returns:
            True if saved successfully
        """
        try:
            embeddings = self.encode(texts)
            if embeddings is None:
                return False
                
            data = {
                'texts': texts,
                'embeddings': embeddings,
                'model_name': self.model_name
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"Embeddings saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return False
    
    def load_embeddings(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load pre-computed embeddings from file.
        
        Args:
            filepath: Path to embeddings file
            
        Returns:
            Dictionary with texts and embeddings, or None if failed
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Embeddings file not found: {filepath}")
                return None
                
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            logger.info(f"Embeddings loaded from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_name': self.model_name,
            'is_loaded': self.is_loaded,
            'device': self.device,
            'cache_dir': self.cache_dir,
            'config': self.config
        }
        
        if self.is_loaded and self.model:
            info['max_seq_length'] = self.model.max_seq_length
            info['embedding_dimension'] = self.model.get_sentence_embedding_dimension()
            
        return info
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.is_loaded = False
        logger.info("MiniLM model resources cleaned up")


# Convenience functions for easy usage
def create_minilm_handler(model_name: str = "all-MiniLM-L6-v2") -> MiniLMHandler:
    """Create and load a MiniLM handler."""
    handler = MiniLMHandler(model_name)
    handler.load_model()
    return handler

def encode_texts(texts: Union[str, List[str]], model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """Quick function to encode texts with MiniLM."""
    handler = create_minilm_handler(model_name)
    return handler.encode(texts)

def calculate_similarity(text1: str, text2: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[float]:
    """Quick function to calculate text similarity."""
    handler = create_minilm_handler(model_name)
    return handler.similarity(text1, text2)
