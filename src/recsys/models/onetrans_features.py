"""Train-only признаки и chronological префиксы для OneTrans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from recsys.data.dataset import InteractionData

IntArray = NDArray[np.int64]


@dataclass
class OneTransFeatures:
    item_features: IntArray
    feature_sizes: tuple[int, int, int]
    histories: list[IntArray]
    history_timestamps: list[IntArray]
    train_examples: IntArray
    train_users: IntArray

    @classmethod
    def from_data(cls, data: InteractionData, metadata: bool = False) -> OneTransFeatures:
        frame = data.train_frame.sort_values(["user_id", "timestamp", "item_id"])
        observed = data.item_counts > 0
        identity = np.zeros(data.n_items, dtype=np.int64)
        identity[observed] = np.arange(1, int(observed.sum()) + 1)
        columns = [identity]
        for name in ("brand", "top_category"):
            values = data.item_meta[name].fillna("").astype(str)
            unique = sorted(set(values[observed]) - {""}) if metadata else []
            vocabulary = {value: index + 1 for index, value in enumerate(unique)}
            columns.append(values.map(vocabulary).fillna(0).to_numpy(dtype=np.int64))
        features = np.vstack([np.zeros((1, 3), dtype=np.int64), np.stack(columns, axis=1)])
        histories = [np.array([], dtype=np.int64) for _ in range(data.n_users)]
        timestamps = [np.array([], dtype=np.int64) for _ in range(data.n_users)]
        examples: list[tuple[int, int, int]] = []
        for user, group in frame.groupby("user_id", sort=False):
            uid = data.user_index(str(user))
            ids = data.items.get_indexer(group["item_id"]).astype(np.int64)
            if np.any(ids < 0):
                raise ValueError("train содержит неизвестные item IDs")
            histories[uid] = ids
            timestamps[uid] = group["timestamp"].to_numpy(dtype=np.int64)
            examples.extend((uid, position, int(ids[position])) for position in range(1, len(ids)))
        return cls(
            features,
            tuple(int(features[:, i].max()) + 1 for i in range(3)),  # type: ignore[arg-type]
            histories,
            timestamps,
            np.asarray(examples, dtype=np.int64).reshape(-1, 3),
            np.flatnonzero(np.asarray(data.train_csr.sum(axis=1)).ravel() > 0),
        )

    def histories_for(
        self, users: IntArray, length: int, positions: IntArray | None = None
    ) -> IntArray:
        out = np.full((len(users), length), -1, dtype=np.int64)
        for row, user in enumerate(users):
            stop = int(positions[row]) if positions is not None else len(self.histories[user])
            history = self.histories[user][max(0, stop - length) : stop]
            if len(history):
                out[row, -len(history) :] = history
        return out

    def torch_table(self) -> torch.Tensor:
        return torch.from_numpy(self.item_features)


def chronological_merge(
    streams: list[tuple[IntArray, IntArray]],
) -> tuple[IntArray, IntArray]:
    """Тип сохраняется при merge: разные реальные потоки не подменяются копиями."""
    rows: list[tuple[int, int, int, int]] = []
    for kind, (items, timestamps) in enumerate(streams):
        if len(items) != len(timestamps):
            raise ValueError("число timestamps не совпадает с числом events")
        rows.extend(
            (int(ts), kind, order, int(item))
            for order, (item, ts) in enumerate(zip(items, timestamps, strict=True))
        )
    rows.sort()
    return (
        np.asarray([row[3] for row in rows], dtype=np.int64),
        np.asarray([row[1] for row in rows], dtype=np.int64),
    )


def target_pairs(frame: pd.DataFrame, data: InteractionData) -> tuple[IntArray, IntArray]:
    users = data.users.get_indexer(frame["user_id"])
    items = data.items.get_indexer(frame["item_id"])
    if np.any(users < 0) or np.any(items < 0):
        raise ValueError("target вне зафиксированного каталога")
    return users.astype(np.int64), items.astype(np.int64)
