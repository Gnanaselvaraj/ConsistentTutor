"""
vector_store.py: Vector DB and retrieval logic
"""
import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple

class VectorStore:
    def __init__(self, dim: int, db_dir: str):
        self.db_dir = db_dir
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []
        self.metadata = []  # Store page numbers, source files, etc.

    def add(self, vectors: np.ndarray, texts: List[str], metadata: List[Dict[str, Any]] = None):
        self.index.add(vectors)
        self.texts.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            # Default empty metadata
            self.metadata.extend([{} for _ in texts])

    def save(self, subject: str):
        subject_dir = os.path.join(self.db_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)
        faiss.write_index(self.index, f"{subject_dir}/index.faiss")
        with open(f"{subject_dir}/texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)
        with open(f"{subject_dir}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, subject: str):
        subject_dir = os.path.join(self.db_dir, subject)
        self.index = faiss.read_index(f"{subject_dir}/index.faiss")
        with open(f"{subject_dir}/texts.pkl", "rb") as f:
            self.texts = pickle.load(f)
        # Load metadata if exists (backward compatibility)
        metadata_path = f"{subject_dir}/metadata.pkl"
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = [{} for _ in self.texts]

    def search(self, q_vec: np.ndarray, k: int = 10, threshold: float = 0.5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search and return (text, score, metadata) tuples"""
        D, I = self.index.search(q_vec, k)
        results = []
        for i, d in zip(I[0], D[0]):
            if d >= threshold:
                metadata = self.metadata[i] if i < len(self.metadata) else {}
                results.append((self.texts[i], d, metadata))
        return results
