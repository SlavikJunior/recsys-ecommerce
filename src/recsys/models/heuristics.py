"""Эвристики на атрибутах товара: «товары этого же бренда» и «этой же категории».

Это не ML-baseline, а именно бизнес-правила: они объяснимы, дешёвы и в
отчёте отвечают на вопрос «что показывать, если модель недоступна».
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.data.dataset import InteractionData
from recsys.models.base import BaseRecommender


class AttributeAffinityRecommender(BaseRecommender):
    """Скор = близость истории пользователя по выбранному атрибуту.

    Вес атрибута нормируется по IDF: редкий бренд информативнее, чем бренд,
    который есть в половине каталога.
    """

    def __init__(self, column: str) -> None:
        super().__init__()
        self.column = column
        self.name = f"{column}_affinity"
        self.profile: sp.csr_matrix | None = None
        self.incidence: sp.csr_matrix | None = None

    def fit(self, data: InteractionData) -> AttributeAffinityRecommender:
        super().fit(data)
        values = data.item_meta[self.column].fillna("").astype(str).to_numpy()
        # Пустые значения не индексируем: иначе в словаре остаются дыры и
        # размер матрицы перестаёт соответствовать максимальному индексу.
        vocab = {value: idx for idx, value in enumerate(sorted({v for v in values if v}))}
        item_idx = np.array([i for i, value in enumerate(values) if value], dtype=np.int64)
        value_idx = np.array([vocab[values[i]] for i in item_idx], dtype=np.int64)
        incidence = sp.csr_matrix(
            (np.ones(len(item_idx), dtype=np.float32), (item_idx, value_idx)),
            shape=(data.n_items, len(vocab)),
        )
        df = np.asarray((incidence > 0).sum(axis=0)).ravel().astype(np.float64)
        idf = np.log((1.0 + data.n_items) / (1.0 + df)) + 1.0
        incidence = incidence.multiply(idf[None, :]).tocsr()

        self.incidence = incidence
        self.profile = (data.train_csr @ incidence).tocsr()
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        if self.incidence is None or self.profile is None:
            raise RuntimeError("сначала вызови fit()")
        scores = self.profile[user_idx] @ self.incidence.T
        return np.asarray(scores.todense(), dtype=np.float32)
