"""Популярность: глобальная и time-decayed.

Это baseline, который в проде всегда остаётся fallback'ом: он дешёвый,
объяснимый и работает при пустом кэше/холодном старте.
"""

from __future__ import annotations

import numpy as np

from recsys.data.dataset import InteractionData
from recsys.models.base import BaseRecommender, PopularityFallbackMixin

# Полураспад по умолчанию ~30 дней: сезонные тренды учитываются, но каталог
# не «схлопывается» в последнюю неделю. На историческом holdout сильное
# затухание проигрывает, и это отдельный вывод в отчёте.
DEFAULT_HALF_LIFE_DAYS = 30.0


class PopularityRecommender(PopularityFallbackMixin, BaseRecommender):
    name = "popularity"

    def __init__(self, decay: bool = False, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> None:
        super().__init__()
        self.decay = decay
        self.half_life_days = half_life_days
        self.popularity_order = np.array([], dtype=np.int64)

    def fit(self, data: InteractionData) -> PopularityRecommender:
        super().fit(data)
        if self.decay:
            ref_ts = int(data.item_last_ts.max())
            age = np.maximum(ref_ts - data.item_last_ts, 0).astype(np.float64)
            half_life_seconds = self.half_life_days * 24 * 3600
            scores = np.exp(-np.log(2) * age / half_life_seconds)
            scores *= data.item_counts > 0
        else:
            scores = data.item_counts.astype(np.float64)
        self.scores = scores.astype(np.float32)
        self.popularity_order = np.argsort(-self.scores).astype(np.int64)
        self.name = f"popularity_decay_{int(self.half_life_days)}d" if self.decay else "popularity"
        return self

    def score_users(self, user_idx: np.ndarray) -> np.ndarray:
        return np.tile(self.scores, (len(user_idx), 1))
