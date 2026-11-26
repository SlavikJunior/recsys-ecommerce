"""Тесты рекомендателей на синтетическом каталоге."""

from __future__ import annotations

import numpy as np

from recsys.data.dataset import InteractionData
from recsys.models.heuristics import AttributeAffinityRecommender
from recsys.models.hybrid import HybridRecommender
from recsys.models.item_cf import ItemCFRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.user_cf import UserCFRecommender


def test_popularity_scores_shape(toy_data: InteractionData) -> None:
    model = PopularityRecommender().fit(toy_data)
    scores = model.score_users(np.array([0, 1]))
    assert scores.shape == (2, toy_data.n_items)
    assert np.all(scores[0] == scores[1])


def test_recommend_excludes_seen(toy_data: InteractionData) -> None:
    model = PopularityRecommender().fit(toy_data)
    recs = model.recommend(np.array([0]), k=2)
    seen = set(toy_data.train_csr[0].indices)
    assert not seen & set(recs[0].tolist())


def test_brand_affinity_prefers_same_brand(toy_data: InteractionData) -> None:
    model = AttributeAffinityRecommender("brand").fit(toy_data)
    scores = model.score_users(np.array([0]))
    # Пользователь 0 покупал brandA (i1, i2); i3/i4 — brandB.
    assert scores[0, 0] > scores[0, 2]


def test_item_and_user_cf_shapes(toy_data: InteractionData) -> None:
    for model in (ItemCFRecommender(neighbors=3), UserCFRecommender(neighbors=2)):
        model.fit(toy_data)
        assert model.score_users(np.array([0, 1])).shape == (2, toy_data.n_items)


def test_hybrid_returns_ranking(toy_data: InteractionData) -> None:
    model = HybridRecommender([PopularityRecommender(), ItemCFRecommender(neighbors=2)])
    model.fit(toy_data)
    recs = model.recommend(np.array([0, 1]), k=3)
    assert recs.shape == (2, 3)
    assert len(set(recs[0].tolist())) == 3
