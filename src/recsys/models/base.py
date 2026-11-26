"""Базовый интерфейс рекомендателей и утилиты ранжирования."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from recsys.data.dataset import InteractionData


class BaseRecommender(ABC):
    """Модель отдаёт скоры по всему каталогу, ранжирование общее для всех.

    Такой интерфейс позволяет честно сравнивать модели на одном протоколе:
    отличие только в скорах, top-k и исключение просмотренного — одинаковые.
    """

    name: str = "base"

    def __init__(self) -> None:
        self.data: InteractionData | None = None

    def fit(self, data: InteractionData) -> BaseRecommender:
        self.data = data
        return self

    @abstractmethod
    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        """Скоры (len(user_idx), n_items) для батча пользователей."""

    def _require_data(self) -> InteractionData:
        if self.data is None:
            raise RuntimeError(f"{self.name}: сначала вызови fit()")
        return self.data

    def recommend(
        self,
        user_idx: np.ndarray,
        k: int = 10,
        exclude_seen: bool = True,
        batch_size: int = 512,
    ) -> np.ndarray:
        data = self._require_data()
        out = np.empty((len(user_idx), k), dtype=np.int64)

        for start in range(0, len(user_idx), batch_size):
            batch = user_idx[start : start + batch_size]
            scores = self.score_users(batch).astype(np.float32, copy=True)
            if exclude_seen:
                seen = data.train_csr[batch]
                # Просмотренное не рекомендуем: это стандарт для offline-оценки.
                rows, cols = seen.nonzero()
                scores[rows, cols] = -np.inf

            k_actual = min(k, scores.shape[1])
            if k_actual < k:
                raise ValueError(f"{self.name}: каталог меньше k={k}")

            # argpartition дешевле полной сортировки, а порядок внутри top-k
            # восстанавливаем сортировкой только этих k элементов.
            top = np.argpartition(-scores, k_actual - 1, axis=1)[:, :k_actual]
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1)
            out[start : start + len(batch)] = np.take_along_axis(top, order, axis=1)

        return out


class PopularityFallbackMixin:
    """Общий fallback: пользователь без истории получает популярное.

    В датасете после 5-core таких пользователей нет в train, но в проде
    холодный старт — норма, поэтому fallback встроен в оценку, а не в отчёт.
    """

    popularity_order: np.ndarray

    def fill_unknown_users(self, scores: np.ndarray, user_idx: np.ndarray) -> None:
        data = self._require_data()  # type: ignore[attr-defined]
        empty = np.asarray(data.train_csr[user_idx].sum(axis=1)).ravel() == 0
        if empty.any():
            scores[empty] = 0.0
            scores[np.ix_(empty, self.popularity_order[:100])] = 1.0
