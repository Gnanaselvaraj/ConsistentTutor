"""
embeddings.py: Embedding logic and batching
"""
import sys
import os

# Import from parent directory's ollama_embeddings module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ollama_embeddings import embed_texts_batched

# Re-export for backward compatibility
__all__ = ['embed_texts_batched']
