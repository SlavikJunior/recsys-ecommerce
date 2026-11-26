"""User-based коллаборативная фильтрация.

Нужна как второй классический подход и как точка сравнения: на больших
каталогах user-CF дороже item-CF, и это видно по latency в отчёте.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.data.dataset import InteractionData
from recsys.models.base import BaseRecommender


class UserCFRecommender(BaseRecommender):
    name = "user_cf"

    def __init__(self, neighbors: int = 100, shrink: float = 10.0) -> None:
        super().__init__()
        self.neighbors = neighbors
        self.shrink = shrink
        self.user_similarity: sp.csr_matrix | None = None

    def fit(self, data: InteractionData) -> UserCFRecommender:
        super().fit(data)
        cooc = (data.train_csr @ data.train_csr.T).tocsr()
        counts = np.asarray(data.train_csr.sum(axis=1)).ravel().astype(np.float64)

        rows = np.repeat(np.arange(cooc.shape[0]), np.diff(cooc.indptr))
        cols = cooc.indices
        values = cooc.data.astype(np.float64) / (np.sqrt(counts[rows] * counts[cols]) + self.shrink)
        sim = sp.csr_matrix((values, (rows, cols)), shape=cooc.shape)
        sim.setdiag(0)
        sim.eliminate_zeros()
        self.user_similarity = self._top_k_rows(sim, self.neighbors)
        return self

    @staticmethod
    def _top_k_rows(similarity: sp.csr_matrix, k: int) -> sp.csr_matrix:
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

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        if self.user_similarity is None:
            raise RuntimeError("сначала вызови fit()")
        data = self._require_data()
        # Интерпретируемо: скор товара = сумма близостей пользователей,
        # которые этот товар покупали. Соседи ограничены top-K.
        scores = self.user_similarity[user_idx] @ data.train_csr
        return np.asarray(scores.todense(), dtype=np.float32)
