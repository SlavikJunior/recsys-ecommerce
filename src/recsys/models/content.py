"""Content-based рекомендации: TF-IDF по тексту карточки товара.

Профиль пользователя — взвешенное среднее эмбеддингов просмотренных товаров.
Этот подход не страдает от холодного старта товара (если есть описание), но
проигрывает коллаборативным моделям там, где важнее совместные покупки.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.data.dataset import InteractionData
from recsys.models.base import BaseRecommender


def build_item_text(data: InteractionData) -> list[str]:
    meta = data.item_meta
    parts = []
    for title, brand, category in zip(
        meta["title"], meta["brand"], meta["top_category"], strict=True
    ):
        # Бренд и категорию дублируем отдельными токенами: названия брендов
        # часто состоят из нескольких слов и иначе «размазываются» по тексту.
        brand_tokens = " ".join(f"brand_{t}" for t in str(brand).lower().split())
        category_tokens = " ".join(f"cat_{t}" for t in str(category).lower().split())
        parts.append(f"{title} {brand_tokens} {category_tokens}")
    return parts


class ContentBasedRecommender(BaseRecommender):
    name = "content_based"

    def __init__(self, max_features: int = 100_000, min_df: int = 2) -> None:
        super().__init__()
        self.max_features = max_features
        self.min_df = min_df
        self.item_vectors: sp.csr_matrix | None = None
        self.profiles: sp.csr_matrix | None = None

    def fit(self, data: InteractionData) -> ContentBasedRecommender:
        super().fit(data)
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.vectorizer = vectorizer
        self.item_vectors = normalize(vectorizer.fit_transform(build_item_text(data))).tocsr()
        # Профиль = сумма векторов истории; нормализуем, чтобы длина истории
        # не превращалась в преимущество в ранжировании.
        self.profiles = normalize(data.train_csr @ self.item_vectors).tocsr()
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        if self.item_vectors is None or self.profiles is None:
            raise RuntimeError("сначала вызови fit()")
        scores = self.profiles[user_idx] @ self.item_vectors.T
        return np.asarray(scores.todense(), dtype=np.float32)
