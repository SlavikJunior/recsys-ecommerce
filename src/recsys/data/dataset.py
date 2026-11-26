"""Загрузка подготовленного датасета и построение матриц взаимодействий."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.config import processed_dir


@dataclass
class InteractionData:
    """Индексированные взаимодействия: user_id/item_id -> внутренние индексы.

    train_csr — бинарная матрица (n_users x n_items) для обучения и исключения
    уже просмотренного при ранжировании.
    """

    users: pd.Index
    items: pd.Index
    item_meta: pd.DataFrame
    train_csr: sp.csr_matrix
    train_frame: pd.DataFrame
    item_counts: np.ndarray
    item_last_ts: np.ndarray
    valid: pd.DataFrame
    test: pd.DataFrame
    stats: dict[str, object]

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def user_index(self, user_id: str) -> int:
        return int(self.users.get_loc(user_id))

    def test_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """Пары (user_idx, item_idx) для leave-last-out оценки."""
        user_idx = self.users.get_indexer(self.test["user_id"])
        item_idx = self.items.get_indexer(self.test["item_id"])
        mask = (user_idx >= 0) & (item_idx >= 0)
        return user_idx[mask], item_idx[mask]

    def valid_targets(self) -> tuple[np.ndarray, np.ndarray]:
        user_idx = self.users.get_indexer(self.valid["user_id"])
        item_idx = self.items.get_indexer(self.valid["item_id"])
        mask = (user_idx >= 0) & (item_idx >= 0)
        return user_idx[mask], item_idx[mask]


def load_dataset(category: str = "All_Beauty") -> InteractionData:
    out_dir = processed_dir(category)
    needed = ["train.parquet", "valid.parquet", "test.parquet", "items.parquet"]
    missing = [name for name in needed if not (out_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"в {out_dir} нет {missing}: сначала запусти python -m recsys.data.prepare"
        )

    train = pd.read_parquet(out_dir / "train.parquet")
    valid = pd.read_parquet(out_dir / "valid.parquet")
    test = pd.read_parquet(out_dir / "test.parquet")
    item_meta = pd.read_parquet(out_dir / "items.parquet")

    # Индексы строим по train+test+valid, чтобы таргет теста всегда был в каталоге.
    all_interactions = pd.concat([train, valid, test], ignore_index=True)
    users = pd.Index(all_interactions["user_id"].unique(), name="user_id")
    items = pd.Index(all_interactions["item_id"].unique(), name="item_id")

    user_idx = users.get_indexer(train["user_id"])
    item_idx = items.get_indexer(train["item_id"])
    train_csr = sp.csr_matrix(
        (np.ones(len(train), dtype=np.float32), (user_idx, item_idx)),
        shape=(len(users), len(items)),
    )
    # Дубликаты в CSR складываются: для бинарной матрицы это не проблема, но
    # нормируем к 1, чтобы веса не росли от повторных покупок одного товара.
    train_csr.data = np.ones_like(train_csr.data)

    item_meta = item_meta.set_index("item_id").reindex(items).reset_index()
    item_meta["title"] = item_meta["title"].fillna("")
    item_meta["brand"] = item_meta["brand"].fillna("")
    item_meta["top_category"] = item_meta["top_category"].fillna("")

    stats = json.loads((out_dir / "stats.json").read_text())
    item_counts = np.bincount(items.get_indexer(train["item_id"]), minlength=len(items)).astype(
        np.float32
    )

    # Для time-decayed популярности: время последнего взаимодействия с товаром.
    last_ts = train.groupby("item_id")["timestamp"].max().reindex(items)
    item_last_ts = last_ts.fillna(0).to_numpy(dtype=np.int64)

    return InteractionData(
        users=users,
        items=items,
        item_meta=item_meta,
        train_csr=train_csr,
        train_frame=train,
        item_counts=item_counts,
        item_last_ts=item_last_ts,
        valid=valid,
        test=test,
        stats=stats,
    )


def catalog_path(category: str = "All_Beauty") -> Path:
    return processed_dir(category) / "items.parquet"
