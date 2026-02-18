"""
multimodal_vector_store.py: Vector store for both text and images (CLIP-based)
"""
import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image

class MultimodalVectorStore:
    """
    Dual vector store for text and images using CLIP embedding space.
    Text uses 384-dim (all-MiniLM-L6-v2), images use 512-dim (CLIP ViT-B-32).
    """
    
    def __init__(self, text_dim: int, image_dim: int, db_dir: str):
        self.db_dir = db_dir
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.current_subject_dir = None  # Track current loaded subject directory
        
        # Separate indices for text and images with different dimensions
        self.text_index = faiss.IndexFlatIP(text_dim)
        self.image_index = faiss.IndexFlatIP(image_dim)
        
        # Storage
        self.texts = []
        self.text_metadata = []
        self.images = []  # Store image paths or PIL images
        self.image_metadata = []
    
    def add_texts(self, vectors: np.ndarray, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Add text embeddings"""
        self.text_index.add(vectors)
        self.texts.extend(texts)
        if metadata:
            self.text_metadata.extend(metadata)
        else:
            self.text_metadata.extend([{} for _ in texts])
    
    def add_images(self, vectors: np.ndarray, images: List[Any], metadata: List[Dict[str, Any]] = None):
        """Add image embeddings"""
        self.image_index.add(vectors)
        self.images.extend(images)
        if metadata:
            self.image_metadata.extend(metadata)
        else:
            self.image_metadata.extend([{} for _ in images])
    
    def save(self, subject: str):
        """Save both indices and persist images"""
        subject_dir = os.path.join(self.db_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)
        
        # Save text index
        faiss.write_index(self.text_index, f"{subject_dir}/text_index.faiss")
        with open(f"{subject_dir}/texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)
        with open(f"{subject_dir}/text_metadata.pkl", "wb") as f:
            pickle.dump(self.text_metadata, f)
        
        # Save image index and persist images to disk
        faiss.write_index(self.image_index, f"{subject_dir}/image_index.faiss")
        
        # Create images directory
        image_dir = os.path.join(subject_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        
        # Save each image as PNG and store path in metadata
        for i, (img, metadata) in enumerate(zip(self.images, self.image_metadata)):
            image_filename = f"img_{i:04d}.png"
            image_path = os.path.join(image_dir, image_filename)
            
            # Extract PIL Image if it's in dict format
            if isinstance(img, dict) and 'image' in img:
                pil_img = img['image']
            else:
                pil_img = img
            
            # Convert CMYK/other modes to RGB before saving as PNG
            if pil_img.mode not in ('RGB', 'RGBA', 'L', 'LA'):
                pil_img = pil_img.convert('RGB')
            
            # Save image to disk
            pil_img.save(image_path)
            
            # Store relative path in metadata (relative to subject_dir)
            metadata['image_path'] = os.path.join("images", image_filename)
        
        # Save updated metadata with image paths
        with open(f"{subject_dir}/image_metadata.pkl", "wb") as f:
            pickle.dump(self.image_metadata, f)
    
    def load(self, subject: str):
        """Load both indices"""
        subject_dir = os.path.join(self.db_dir, subject)
        self.current_subject_dir = subject_dir  # Track current subject directory
        
        # Load text index
        if os.path.exists(f"{subject_dir}/text_index.faiss"):
            self.text_index = faiss.read_index(f"{subject_dir}/text_index.faiss")
            with open(f"{subject_dir}/texts.pkl", "rb") as f:
                self.texts = pickle.load(f)
            with open(f"{subject_dir}/text_metadata.pkl", "rb") as f:
                self.text_metadata = pickle.load(f)
        
        # Load image index
        if os.path.exists(f"{subject_dir}/image_index.faiss"):
            self.image_index = faiss.read_index(f"{subject_dir}/image_index.faiss")
            with open(f"{subject_dir}/image_metadata.pkl", "rb") as f:
                self.image_metadata = pickle.load(f)
    
    def search_text(self, q_vec: np.ndarray, k: int = 10, threshold: float = 0.20) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search in text index.
        
        Philosophy: Use MINIMAL thresholds (0.20) to filter only obvious noise.
        Let LLM's _is_relevant_answer() be the true gatekeeper for quality.
        This prevents artificial scarcity where good content gets filtered out.
        """
        D, I = self.text_index.search(q_vec, k)
        results = []
        for i, d in zip(I[0], D[0]):
            if d >= threshold and i < len(self.texts):
                metadata = self.text_metadata[i] if i < len(self.text_metadata) else {}
                results.append((self.texts[i], d, metadata))
        return results
    
    def search_images(self, q_vec: np.ndarray, k: int = 5, threshold: float = 0.15) -> List[Tuple[Any, Dict[str, Any], float]]:
        """
        Search in image index (returns image data, metadata, and score).
        
        Philosophy: Use VERY LOW thresholds (0.15) to be lenient with diagrams.
        Let LLM decide final relevance. Better to show more diagrams than miss relevant ones.
        """
        D, I = self.image_index.search(q_vec, k)
        results = []
        for i, d in zip(I[0], D[0]):
            if d >= threshold and i < len(self.image_metadata):
                metadata = self.image_metadata[i]
                
                # Load image on-demand from path in metadata
                image_data = None
                if i < len(self.images) and self.images[i] is not None:
                    # Images already loaded in memory
                    image_data = self.images[i]
                elif 'image_path' in metadata:
                    # Load image from path stored in metadata
                    try:
                        # Construct full path using current subject directory
                        relative_path = metadata['image_path']
                        if self.current_subject_dir:
                            full_image_path = os.path.join(self.current_subject_dir, relative_path)
                            if os.path.exists(full_image_path):
                                from PIL import Image
                                pil_img = Image.open(full_image_path)
                                # Return as dict for compatibility with _format_image_references
                                image_data = {'image': pil_img}
                            else:
                                print(f"Warning: Image path not found: {full_image_path}")
                        else:
                            print(f"Warning: current_subject_dir not set")
                    except Exception as e:
                        print(f"Warning: Could not load image from {relative_path}: {e}")
                        image_data = None
                
                results.append((image_data, metadata, d))
        return results
    
    def search_multimodal(self, q_vec_text: np.ndarray, q_vec_image: np.ndarray, 
                          k_text: int = 10, k_images: int = 3, 
                          text_threshold: float = 0.5, image_threshold: float = 0.4) -> Dict[str, Any]:
        """
        Search both text and images with separate query vectors.
        
        Args:
            q_vec_text: Text query embedding (384-dim for all-MiniLM-L6-v2)
            q_vec_image: Image query embedding (512-dim for CLIP)
            k_text: Number of text results
            k_images: Number of image results
            text_threshold: Minimum score for text results
            image_threshold: Minimum score for image results
        
        Returns:
            {
                'texts': [(text, score, metadata), ...],
                'images': [(metadata, score), ...],
                'has_visual': bool
            }
        """
        text_results = self.search_text(q_vec_text, k=k_text, threshold=text_threshold)
        image_results = self.search_images(q_vec_image, k=k_images, threshold=image_threshold)
        
        return {
            'texts': text_results,
            'images': image_results,
            'has_visual': len(image_results) > 0
        }
