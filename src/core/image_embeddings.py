"""
image_embeddings.py: CLIP-based image embedding for multimodal RAG
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

# Global model instance (lazy loaded)
_CLIP_MODEL = None

def get_clip_model():
    """Lazy load CLIP model"""
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        _CLIP_MODEL = SentenceTransformer(CLIP_MODEL, device=DEVICE)
    return _CLIP_MODEL

def embed_images_batched(images: List[Union[Image.Image, str]], batch_size: int = 16) -> np.ndarray:
    """
    Embed a list of PIL Images or image paths using CLIP.
    
    Args:
        images: List of PIL Image objects or paths to images
        batch_size: Number of images to process at once
    
    Returns:
        numpy array of shape (len(images), 512) with float32 embeddings
    """
    model = get_clip_model()
    
    # Convert paths to PIL Images if needed
    pil_images = []
    for img in images:
        if isinstance(img, str):
            pil_images.append(Image.open(img).convert('RGB'))
        elif isinstance(img, bytes):
            pil_images.append(Image.open(io.BytesIO(img)).convert('RGB'))
        else:
            pil_images.append(img)
    
    if not pil_images:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype='float32')
    
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
