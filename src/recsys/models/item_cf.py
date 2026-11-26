"""Item-based коллаборативная фильтрация.

Симметричная косинусная близость по матрице со-встречаемости + top-K соседей:
хранить полную матрицу items x items в 100+ тыс. товаров бессмысленно.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.data.dataset import InteractionData
from recsys.models.base import BaseRecommender


def top_k_neighbors(similarity: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Оставляет k самых сильных соседей в каждой строке."""
    rows, cols, data = [], [], []
    for row in range(similarity.shape[0]):
        start, end = similarity.indptr[row], similarity.indptr[row + 1]
        if start == end:
            continue
        idx = similarity.indices[start:end]
        vals = similarity.data[start:end]
        if len(vals) > k:
            keep = np.argpartition(-vals, k)[:k]
            idx, vals = idx[keep], vals[keep]
        rows.append(np.full(len(idx), row, dtype=np.int64))
        cols.append(idx.astype(np.int64))
        data.append(vals.astype(np.float32))
    if not rows:
        return sp.csr_matrix(similarity.shape, dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=similarity.shape,
    )


class ItemCFRecommender(BaseRecommender):
    name = "item_cf"

    def __init__(self, neighbors: int = 200, shrink: float = 10.0) -> None:
        super().__init__()
        self.neighbors = neighbors
        self.shrink = shrink
        self.item_similarity: sp.csr_matrix | None = None

    def fit(self, data: InteractionData) -> ItemCFRecommender:
        super().fit(data)
        cooc = (data.train_csr.T @ data.train_csr).tocsr()
        counts = np.asarray(data.train_csr.sum(axis=0)).ravel().astype(np.float64)

        # Косинус со shrinkage: у редких товаров близость занижается, иначе
        # один совместный отзыв даёт «идеальную» пару.
        rows = np.repeat(np.arange(cooc.shape[0]), np.diff(cooc.indptr))
        cols = cooc.indices
        values = cooc.data.astype(np.float64)
        denom = np.sqrt(counts[rows] * counts[cols]) + self.shrink
        values = values / denom
        sim = sp.csr_matrix((values, (rows, cols)), shape=cooc.shape)
        sim.setdiag(0)
        sim.eliminate_zeros()
        self.item_similarity = top_k_neighbors(sim, self.neighbors)
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        if self.item_similarity is None:
            raise RuntimeError("сначала вызови fit()")
        data = self._require_data()
        scores = data.train_csr[user_idx] @ self.item_similarity
        return np.asarray(scores.todense(), dtype=np.float32)
