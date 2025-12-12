
# src/vector_store.py
import faiss
import numpy as np
from typing import List, Dict, Any

class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # using inner product on normalized vectors
        self.id_map = []

    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
        arr = np.array(vectors, dtype='float32')
        # ensure normalization
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        arr = arr / norms
        self.index.add(arr)
        self.id_map.extend(metadatas)

    def search(self, query_vector: List[float], top_k: int = 5):
        q = np.array([query_vector], dtype='float32')
        q = q / (np.linalg.norm(q) + 1e-9)
        D, I = self.index.search(q, top_k)
        results = []
        for dist, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            meta = self.id_map[idx]
            results.append({
                'score': float(dist),
                'metadata': meta
            })
        return results
