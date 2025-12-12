from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        if _HAS_ST:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None
        else:
            self.model = None

    def encode(self, texts: List[str]) -> List[List[float]]:
        if self.model:
            vecs = self.model.encode(texts, show_progress_bar=False)
            return [v.tolist() if hasattr(v, 'tolist') else v for v in vecs]
        # fallback: simple TF-IDF-ish via hashing (very rough)
        return [self._simple_hash(t) for t in texts]

    def _simple_hash(self, text: str, dim: int = 384):
        # deterministic naive hashing to vector
        vec = np.zeros(dim, dtype=float)
        for i, ch in enumerate(text.lower()):
            vec[ord(ch) % dim] += 1.0
        # normalize
        norm = np.linalg.norm(vec) + 1e-9
        return (vec / norm).tolist()