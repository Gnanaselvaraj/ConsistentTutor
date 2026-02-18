"""
cache_manager.py: Intelligent caching for embeddings and responses
"""
import hashlib
import pickle
import os
from typing import Any, Optional
from functools import lru_cache
import numpy as np

class CacheManager:
    """Persistent cache for embeddings and common responses"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.embedding_cache_file = os.path.join(cache_dir, "embeddings.pkl")
        self.response_cache_file = os.path.join(cache_dir, "responses.pkl")
        
        # Load existing caches
        self.embedding_cache = self._load_cache(self.embedding_cache_file)
        self.response_cache = self._load_cache(self.response_cache_file)
    
    def _load_cache(self, filepath: str) -> dict:
        """Load cache from disk"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self, cache: dict, filepath: str):
        """Save cache to disk"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text"""
        key = self._hash_text(text)
        return self.embedding_cache.get(key)
    
    def set_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding for text"""
        key = self._hash_text(text)
        self.embedding_cache[key] = embedding
        
        # Save periodically (every 10 additions)
        if len(self.embedding_cache) % 10 == 0:
            self._save_cache(self.embedding_cache, self.embedding_cache_file)
    
    def get_response(self, question: str, subject: str) -> Optional[str]:
        """Get cached response for question+subject"""
        key = self._hash_text(f"{subject}::{question}")
        return self.response_cache.get(key)
    
    def set_response(self, question: str, subject: str, response: str):
        """Cache response for question+subject"""
        key = self._hash_text(f"{subject}::{question}")
        self.response_cache[key] = response
        
        # Save periodically
        if len(self.response_cache) % 5 == 0:
            self._save_cache(self.response_cache, self.response_cache_file)
    
    def save_all(self):
        """Force save all caches"""
        self._save_cache(self.embedding_cache, self.embedding_cache_file)
        self._save_cache(self.response_cache, self.response_cache_file)
    
    def clear_cache(self):
        """Clear all caches"""
        self.embedding_cache = {}
        self.response_cache = {}
        self.save_all()
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            'embeddings_cached': len(self.embedding_cache),
            'responses_cached': len(self.response_cache),
            'cache_dir': self.cache_dir
        }
