"""Общие фикстуры: маленький синтетический датасет без чтения с диска."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.config import SplitConfig
from recsys.data.dataset import InteractionData
from recsys.data.prepare import five_core, make_splits


@pytest.fixture
def toy_interactions() -> pd.DataFrame:
    rows = []
    ts = 1_600_000_000
    for user in ["u1", "u2", "u3"]:
        for i, item in enumerate(["i1", "i2", "i3", "i4", "i5"]):
            rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "rating": 5.0,
                    "timestamp": ts + i + hash(user) % 10,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def toy_data(toy_interactions: pd.DataFrame) -> InteractionData:
    train, valid, test = make_splits(toy_interactions, SplitConfig(min_train_interactions=3))
    users = pd.Index(sorted(toy_interactions["user_id"].unique()))
    items = pd.Index(sorted(toy_interactions["item_id"].unique()))
    csr = sp.csr_matrix(
        (
            np.ones(len(train), dtype=np.float32),
            (users.get_indexer(train["user_id"]), items.get_indexer(train["item_id"])),
        ),
        shape=(len(users), len(items)),
    )
    meta = pd.DataFrame(
        {
            "item_id": items,
            "title": [f"title {i}" for i in items],
            "brand": ["brandA", "brandA", "brandB", "brandB", "brandC"],
            "top_category": ["c1", "c1", "c1", "c2", "c2"],
            "store": ["", "", "", "", ""],
            "categories": ["", "", "", "", ""],
            "price": [1.0, 2.0, 3.0, 4.0, 5.0],
            "average_rating": [4.0] * 5,
            "rating_number": [10] * 5,
        }
    )
    return InteractionData(
        users=users,
        items=items,
        item_meta=meta,
        train_csr=csr,
        train_frame=train,
        item_counts=np.bincount(items.get_indexer(train["item_id"]), minlength=len(items)).astype(
            np.float32
        ),
        item_last_ts=np.array([1_600_000_000] * len(items), dtype=np.int64),
        valid=valid,
        test=test,
        stats={},
    )


def test_five_core_drops_rare(toy_interactions: pd.DataFrame) -> None:
    filtered = five_core(toy_interactions, min_interactions=3)
    assert len(filtered) == len(toy_interactions)


def test_make_splits_last_out(toy_interactions: pd.DataFrame) -> None:
    train, valid, test = make_splits(toy_interactions, SplitConfig())
    assert len(test) == 3
    assert len(valid) == 3
    # Тестовый товар не должен встречаться в train у того же пользователя.
    for user, item in zip(test["user_id"], test["item_id"], strict=True):
        assert item not in set(train[train["user_id"] == user]["item_id"])
