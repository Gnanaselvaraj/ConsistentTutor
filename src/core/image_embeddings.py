"""
image_embeddings.py: CLIP-based image embedding for multimodal RAG
OPTIMIZATION: Pre-load CLIP model at import for sub-20s response times
"""
import torch
import numpy as np
from typing import List, Union
from PIL import Image
import io

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")

# Use CLIP model for multimodal embeddings
CLIP_MODEL = "clip-ViT-B-32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# EAGER LOADING: Pre-load CLIP model at import time (not lazy)
# With 64GB RAM, keep model resident in memory for instant access
# Eliminates 10-15s CLIP loading delay on first query
print(f"🚀 Pre-loading CLIP model ({CLIP_MODEL}) on {DEVICE}...")
_CLIP_MODEL = SentenceTransformer(CLIP_MODEL, device=DEVICE)
print(f"✅ CLIP model ready in memory")

def get_clip_model():
    """Get pre-loaded CLIP model (no lazy loading)"""
    return _CLIP_MODEL

def embed_images_batched(images: List[Union[Image.Image, str]], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of PIL Images or image paths using CLIP.
    
    MEMORY OPTIMIZATION (v2): Increased batch size from 16 to 32 for better GPU utilization.
    
    Args:
        images: List of PIL Image objects or paths to images
        batch_size: Number of images to process at once (default 32 for GPU)
    
    Returns:
        numpy array of shape (len(images), 512) with float32 embeddings
    """
    model = get_clip_model()
    
    # Convert paths to PIL Images if needed
    pil_images = []
    for i, img in enumerate(images):
        try:
            if isinstance(img, str):
                pil_img = Image.open(img).convert('RGB')
            elif isinstance(img, bytes):
                pil_img = Image.open(io.BytesIO(img)).convert('RGB')
            else:
                pil_img = img
            
            # Test if image is valid by trying to load it
            pil_img.load()
            pil_images.append(pil_img)
        except (OSError, IOError) as e:
            # Skip corrupted images (JPEG2000, broken streams, etc.)
            print(f"Warning: Skipping corrupted image {i+1}/{len(images)}: {str(e)}")
            continue
    
    if not pil_images:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype='float32')
    
    print(f"Encoding {len(pil_images)}/{len(images)} valid images")
    
    # Encode in batches
    embeddings = model.encode(
        pil_images,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    return embeddings.astype('float32')

def embed_text_for_image_search(text: str) -> np.ndarray:
    """
    Embed text query using CLIP for cross-modal search (text -> image).
    Uses the same embedding space as images.
    
    Args:
        text: Query text
    
    Returns:
        numpy array of shape (1, 512)
    """
    model = get_clip_model()
    embedding = model.encode(
        [text],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embedding.astype('float32')
