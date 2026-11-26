"""Гибрид: Reciprocal Rank Fusion нескольких моделей.

Складывать сырые скоры нельзя — у popularity и TF-IDF разный масштаб.
RRF работает только с позициями, поэтому устойчив к калибровке и хорошо
объясняется бизнесу: «товар попал в топ нескольких источников».
"""

from __future__ import annotations

import numpy as np

from recsys.data.dataset import InteractionData
from recsys.models.base import BaseRecommender


class HybridRecommender(BaseRecommender):
    name = "hybrid_rrf"

    def __init__(
        self,
        models: list[BaseRecommender],
        weights: list[float] | None = None,
        top_n: int = 300,
        rrf_k: int = 60,
        batch_size: int = 512,
        fit_sources: bool = True,
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("нужен хотя бы один источник")
        self.models = models
        self.weights = weights or [1.0] * len(models)
        if len(self.weights) != len(models):
            raise ValueError("weights и models должны совпадать по длине")
        self.top_n = top_n
        self.rrf_k = rrf_k
        self.batch_size = batch_size
        # Для подбора весов источники обучаются один раз и переиспользуются.
        self.fit_sources = fit_sources

    def fit(self, data: InteractionData) -> HybridRecommender:
        super().fit(data)
        if self.fit_sources:
            for model in self.models:
                model.fit(data)
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        data = self._require_data()
        fused = np.zeros((len(user_idx), data.n_items), dtype=np.float32)
        seen = data.train_csr[user_idx]

        for model, weight in zip(self.models, self.weights, strict=True):
            scores = model.score_users(user_idx).astype(np.float32, copy=True)
            rows, cols = seen.nonzero()
            scores[rows, cols] = -np.inf
            top_n = min(self.top_n, scores.shape[1])
            top = np.argpartition(-scores, top_n - 1, axis=1)[:, :top_n]
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1)
            ranked = np.take_along_axis(top, order, axis=1)

            ranks = np.arange(top_n, dtype=np.float32)
            contribution = weight / (self.rrf_k + 1.0 + ranks)
            np.put_along_axis(
                fused,
                ranked,
                np.take_along_axis(fused, ranked, axis=1) + contribution[None, :],
                axis=1,
            )
        return fused
