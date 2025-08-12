from __future__ import annotations
import json
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss
import os

class VectorStore:
    def __init__(self, dim: int, index_path: str, meta_path: str):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatIP(dim)
        self._metas: List[Dict[str, Any]] = []
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.load()

    def add(self, embeddings: List[List[float]], metas: List[Dict[str, Any]]):
        arr = np.array(embeddings, dtype="float32")
        # Normalize for cosine similarity using inner product
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self._metas.extend(metas)

    def search(self, query_emb: List[float], top_k: int) -> List[Tuple[int, float]]:
        q = np.array([query_emb], dtype="float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self._metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def load(self):
        self.index = faiss.read_index(self.index_path)
        self._metas = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self._metas.append(json.loads(line))

    @property
    def metas(self) -> List[Dict[str, Any]]:
        return self._metas