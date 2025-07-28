#!/usr/bin/env python3
"""
Startup Optimizer: Pre-loads all models at container startup
Eliminates model loading time during processing
"""
import os
import time
import logging

logger = logging.getLogger(__name__)

class StartupOptimizer:
    """Optimizes startup by pre-loading all heavy components"""
    
    @staticmethod
    def preload_all_models():
        """Pre-load all models to eliminate runtime loading"""
        total_start = time.time()
        
        print("[STARTUP] Pre-loading models for optimal performance...")
        
        # 1. Pre-load SentenceTransformer
        try:
            start = time.time()
            from sentence_transformers import SentenceTransformer
            
            # Check if local model exists
            model_path = "models/all-MiniLM-L6-v2"
            if os.path.exists(model_path):
                model = SentenceTransformer(model_path)
            else:
                model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Store in global cache
            globals()['_cached_sentence_transformer'] = model
            load_time = time.time() - start
            print(f"[STARTUP] SentenceTransformer loaded: {load_time:.2f}s")
            
        except Exception as e:
            print(f"[STARTUP] SentenceTransformer failed: {e}")
            globals()['_cached_sentence_transformer'] = None
        
        # 2. Pre-load spaCy
        try:
            start = time.time()
            import spacy
            nlp = spacy.load('en_core_web_sm')
            
            # Store in global cache
            globals()['_cached_spacy_model'] = nlp
            load_time = time.time() - start
            print(f"[STARTUP] spaCy loaded: {load_time:.2f}s")
            
        except Exception as e:
            print(f"[STARTUP] spaCy failed: {e}")
            globals()['_cached_spacy_model'] = None
        
        # 3. Pre-compute embeddings
        try:
            if globals().get('_cached_sentence_transformer'):
                start = time.time()
                keywords = [
                    "introduction", "conclusion", "summary", "abstract", "overview",
                    "background", "methodology", "results", "discussion", "references",
                    "chapter", "section", "subsection", "appendix", "table of contents",
                    "application", "form", "document", "government", "servant"
                ]
                embeddings = globals()['_cached_sentence_transformer'].encode(keywords)
                globals()['_cached_embeddings'] = {'keywords': keywords, 'embeddings': embeddings}
                
                embed_time = time.time() - start
                print(f"[STARTUP] Embeddings computed: {embed_time:.2f}s")
            
        except Exception as e:
            print(f"[STARTUP] Embedding computation failed: {e}")
            globals()['_cached_embeddings'] = None
        
        total_time = time.time() - total_start
        print(f"[STARTUP] All models pre-loaded in {total_time:.2f}s")
        
        # Set environment flag
        os.environ['MODELS_PRELOADED'] = 'true'
    
    @staticmethod
    def get_cached_sentence_transformer():
        """Get pre-loaded SentenceTransformer model"""
        return globals().get('_cached_sentence_transformer')
    
    @staticmethod
    def get_cached_spacy_model():
        """Get pre-loaded spaCy model"""
        return globals().get('_cached_spacy_model')
    
    @staticmethod
    def get_cached_embeddings():
        """Get pre-computed embeddings"""
        return globals().get('_cached_embeddings')

# Pre-load models immediately when module is imported with timeout protection
if os.environ.get('DOCKER_CONTAINER') or os.environ.get('PRELOAD_MODELS'):
    try:
        import signal
        def startup_timeout_handler(signum, frame):
            print('[STARTUP] Model loading timeout - using fallback mode')
            os.environ['DISABLE_HEAVY_MODELS'] = 'true'
        
        signal.signal(signal.SIGALRM, startup_timeout_handler)
        signal.alarm(45)  # 45 second startup timeout
        
        try:
            StartupOptimizer.preload_all_models()
        finally:
            signal.alarm(0)
    except Exception as e:
        print(f'[STARTUP] Model loading failed: {e} - using fallback mode')
        os.environ['DISABLE_HEAVY_MODELS'] = 'true'