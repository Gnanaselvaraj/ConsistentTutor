"""
subject_cache.py: Intelligent subject knowledge base cache for sub-20s query responses

Strategy:
- Pre-load entire subject vector DBs (FAISS indices + metadata) into RAM
- Cache keyed by subject name with LRU eviction
- With 64GB RAM, can cache multiple subjects simultaneously
- Eliminates disk I/O bottleneck on every query
"""
import os
import logging
from typing import Dict, Optional
from collections import OrderedDict
import numpy as np
from .multimodal_vector_store import MultimodalVectorStore

logger = logging.getLogger(__name__)

class SubjectKnowledgeCache:
    """
    Pre-loads and caches entire subject knowledge bases in memory.
    
    Performance benefit:
    - Without cache: Load FAISS from disk on every query (~5-10s disk I/O)
    - With cache: Instant memory lookup (<0.01s)
    - Target: Reduce response time from 30s to <20s
    
    MEMORY OPTIMIZATION (v2): Increased from 10 to 30 subjects to fully utilize 64GB RAM.
    With 4 subjects using ~17GB, we have ~47GB free - can cache 30 subjects comfortably.
    """
    
    def __init__(self, max_subjects: int = 30, text_dim: int = 384, image_dim: int = 512, db_dir: str = "vector_db"):
        """
        Initialize subject cache with expanded capacity.
        
        Args:
            max_subjects: Maximum subjects to cache (LRU eviction) - default 30 for 64GB RAM
            text_dim: Text embedding dimension (384 for MiniLM)
            image_dim: Image embedding dimension (512 for CLIP)
            db_dir: Vector database directory
        """
        self.max_subjects = max_subjects
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.db_dir = db_dir
        
        # LRU cache: OrderedDict maintains insertion order
        self._cache: OrderedDict[str, MultimodalVectorStore] = OrderedDict()
        
        # Query embedding cache for frequent patterns (saves ~0.5s per repeated query type)
        self._query_embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_query_cache = 100  # Cache 100 most recent query embeddings
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'subjects_loaded': 0,
            'query_embedding_hits': 0,
            'query_embedding_misses': 0
        }
        
        logger.info(f"🗄️ Subject cache initialized: max={max_subjects} subjects, query_cache={self._max_query_cache}, text_dim={text_dim}, image_dim={image_dim}")
    
    def get_vector_store(self, subject: str) -> Optional[MultimodalVectorStore]:
        """
        Get vector store for subject (from cache or load from disk).
        
        Args:
            subject: Subject name (e.g., "Physics - 12 - TN")
        
        Returns:
            MultimodalVectorStore instance or None if not found
        """
        subject_dir = os.path.join(self.db_dir, subject)
        
        # Check if subject exists on disk
        if not os.path.exists(subject_dir):
            logger.warning(f"Subject not found: {subject}")
            return None
        
        # Cache hit: Return cached vector store
        if subject in self._cache:
            self.stats['hits'] += 1
            # Move to end (most recently used)
            self._cache.move_to_end(subject)
            logger.debug(f"✅ Cache HIT: {subject} (hits={self.stats['hits']}, misses={self.stats['misses']})")
            return self._cache[subject]
        
        # Cache miss: Load from disk
        self.stats['misses'] += 1
        logger.info(f"⚠️ Cache MISS: {subject} - Loading from disk... (hits={self.stats['hits']}, misses={self.stats['misses']})")
        
        try:
            # Create new vector store instance
            vector_store = MultimodalVectorStore(self.text_dim, self.image_dim, self.db_dir)
            
            # Load FAISS indices and metadata from disk
            vector_store.load(subject)
            
            # Check cache size and evict if needed (LRU)
            if len(self._cache) >= self.max_subjects:
                # Remove oldest (least recently used)
                evicted_subject, evicted_store = self._cache.popitem(last=False)
                self.stats['evictions'] += 1
                logger.info(f"🗑️ Cache FULL: Evicted {evicted_subject} (LRU)")
            
            # Add to cache
            self._cache[subject] = vector_store
            self.stats['subjects_loaded'] += 1
            
            # Log success
            num_texts = len(vector_store.texts)
            num_images = len(vector_store.images) if vector_store.images else 0
            logger.info(f"✅ Loaded {subject}: {num_texts} texts, {num_images} images")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"❌ Failed to load {subject}: {e}")
            return None
    
    def preload_subjects(self, subjects: list):
        """
        Pre-load multiple subjects into cache on startup.
        
        Args:
            subjects: List of subject names to pre-load
        """
        logger.info(f"🚀 Pre-loading {len(subjects)} subjects...")
        
        for i, subject in enumerate(subjects, 1):
            vector_store = self.get_vector_store(subject)
            if vector_store:
                logger.info(f"[{i}/{len(subjects)}] ✅ Pre-loaded: {subject}")
            else:
                logger.warning(f"[{i}/{len(subjects)}] ❌ Failed: {subject}")
        
        logger.info(f"🎉 Pre-loading complete: {len(self._cache)}/{len(subjects)} subjects in cache")
    
    def clear_cache(self):
        """Clear entire cache"""
        num_subjects = len(self._cache)
        self._cache.clear()
        logger.info(f"🗑️ Cache cleared: {num_subjects} subjects removed")
    
    def get_cached_subjects(self) -> list:
        """Get list of currently cached subjects"""
        return list(self._cache.keys())
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
        query_hit_rate = self.stats['query_embedding_hits'] / (self.stats['query_embedding_hits'] + self.stats['query_embedding_misses']) if (self.stats['query_embedding_hits'] + self.stats['query_embedding_misses']) > 0 else 0
        
        return {
            **self.stats,
            'cached_subjects': len(self._cache),
            'cache_size': f"{len(self._cache)}/{self.max_subjects}",
            'hit_rate': f"{hit_rate:.1%}",
            'query_embedding_cache_size': f"{len(self._query_embedding_cache)}/{self._max_query_cache}",
            'query_embedding_hit_rate': f"{query_hit_rate:.1%}",
            'subjects': list(self._cache.keys())
        }
    
    def get_query_embedding(self, query_hash: str) -> Optional[np.ndarray]:
        """
        Get cached query embedding.
        
        Args:
            query_hash: Hash of the query text
        
        Returns:
            Cached embedding or None if not found
        """
        if query_hash in self._query_embedding_cache:
            self.stats['query_embedding_hits'] += 1
            self._query_embedding_cache.move_to_end(query_hash)
            return self._query_embedding_cache[query_hash]
        
        self.stats['query_embedding_misses'] += 1
        return None
    
    def cache_query_embedding(self, query_hash: str, embedding: np.ndarray):
        """
        Cache query embedding for reuse.
        
        Args:
            query_hash: Hash of the query text
            embedding: Query embedding vector
        """
        # Check cache size and evict if needed (LRU)
        if len(self._query_embedding_cache) >= self._max_query_cache:
            self._query_embedding_cache.popitem(last=False)
        
        self._query_embedding_cache[query_hash] = embedding
    
    def remove_subject(self, subject: str) -> bool:
        """
        Remove specific subject from cache.
        
        Args:
            subject: Subject name to remove
        
        Returns:
            True if removed, False if not in cache
        """
        if subject in self._cache:
            del self._cache[subject]
            logger.info(f"🗑️ Removed {subject} from cache")
            return True
        return False
