"""Подготовка датасета: 5-core фильтрация, сплиты по времени, метаданные.

Протокол повторяет принятый в Amazon Reviews 2023 (и в статье McAuley et al.):
оставляем пользователей и товары с >= 5 взаимодействиями, сортируем историю
пользователя по времени и делаем leave-last-out:

    ... train train | valid | test

Такой сплит проверяет именно предсказание следующего товара, а не случайный
попал/не попал, и не даёт модели подсмотреть будущее.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
from tqdm import tqdm

from recsys.config import SplitConfig, processed_dir, raw_paths


def _read_reviews(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Читает jsonl построчно: целиком файл в память не помещаем."""
    user_ids: list[str] = []
    item_ids: list[str] = []
    ratings: list[float] = []
    timestamps: list[int] = []

    with path.open("rb") as fh:
        for i, line in enumerate(tqdm(fh, desc=f"reviews {path.name}", unit=" lines")):
            if limit is not None and i >= limit:
                break
            row = orjson.loads(line)
            user_ids.append(row["user_id"])
            item_ids.append(row["parent_asin"])
            ratings.append(float(row.get("rating") or 0.0))
            timestamps.append(int(row.get("timestamp") or 0))

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "item_id": item_ids,
            "rating": np.asarray(ratings, dtype=np.float32),
            "timestamp": np.asarray(timestamps, dtype=np.int64),
        }
    )


def _read_meta(path: Path) -> pd.DataFrame:
    """Метаданные товаров: только поля, нужные контентной модели и эвристикам."""
    rows: list[dict[str, object]] = []
    with path.open("rb") as fh:
        for line in tqdm(fh, desc=f"meta {path.name}", unit=" lines"):
            row = orjson.loads(line)
            categories = row.get("categories") or []
            details = row.get("details") or {}
            rows.append(
                {
                    "item_id": row["parent_asin"],
                    "title": row.get("title") or "",
                    "store": row.get("store") or "",
                    "categories": " > ".join(str(c) for c in categories),
                    "top_category": str(categories[0]) if categories else "",
                    "price": _parse_price(row.get("price")),
                    "average_rating": float(row.get("average_rating") or 0.0),
                    "rating_number": int(row.get("rating_number") or 0),
                    "brand": str(details.get("Brand") or row.get("store") or ""),
                }
            )
    meta = pd.DataFrame(rows)
    return meta.drop_duplicates(subset="item_id").reset_index(drop=True)


def _parse_price(value: object) -> float:
    """Цена в датасете приходит строкой вида '$12.99' или None."""
    if value is None:
        return float("nan")
    if isinstance(value, int | float):
        return float(value)
    text = str(value).replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def five_core(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """Итеративная 5-core фильтрация: одновременно по пользователям и товарам."""
    current = df
    while True:
        before = len(current)
        user_counts = current["user_id"].value_counts()
        item_counts = current["item_id"].value_counts()
        current = current[
            current["user_id"].isin(user_counts[user_counts >= min_interactions].index)
            & current["item_id"].isin(item_counts[item_counts >= min_interactions].index)
        ]
        if len(current) == before:
            return current.reset_index(drop=True)


def make_splits(
    df: pd.DataFrame, cfg: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-last-out по времени на пользователя."""
    work = df.copy()
    if cfg.dedupe:
        # Один пользователь мог оставить несколько отзывов на товар: оставляем последний.
        work = work.sort_values("timestamp").drop_duplicates(
            subset=["user_id", "item_id"], keep="last"
        )
    work = work.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    work["rank_from_end"] = work.groupby("user_id").cumcount(ascending=False)

    test = work[work["rank_from_end"] == 0]
    valid = work[work["rank_from_end"] == 1]
    train = work[work["rank_from_end"] >= 2]

    train_counts = train.groupby("user_id").size()
    keep_users = train_counts[train_counts >= cfg.min_train_interactions].index
    return (
        train[train["user_id"].isin(keep_users)].reset_index(drop=True),
        valid[valid["user_id"].isin(keep_users)].reset_index(drop=True),
        test[test["user_id"].isin(keep_users)].reset_index(drop=True),
    )


def prepare(category: str, cfg: SplitConfig | None = None, limit: int | None = None) -> Path:
    cfg = cfg or SplitConfig()
    reviews_path, meta_path = raw_paths(category)
    if not reviews_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"нет данных для {category}: сначала запусти python -m recsys.data.download"
        )

    out_dir = processed_dir(category)
    out_dir.mkdir(parents=True, exist_ok=True)

    reviews = _read_reviews(reviews_path, limit=limit)
    reviews = reviews[reviews["item_id"].notna() & reviews["user_id"].notna()]
    reviews = five_core(reviews, cfg.min_interactions)
    train, valid, test = make_splits(reviews, cfg)

    # Метаданные только по товарам, которые остались после фильтрации.
    items = _read_meta(meta_path)
    known_items = set(reviews["item_id"].unique())
    items = items[items["item_id"].isin(known_items)].reset_index(drop=True)

    train.to_parquet(out_dir / "train.parquet", index=False)
    valid.to_parquet(out_dir / "valid.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)

    stats = {
        "category": category,
        "interactions_raw": int(len(reviews)),
        "users": int(reviews["user_id"].nunique()),
        "items": int(reviews["item_id"].nunique()),
        "items_with_meta": int(len(items)),
        "train": int(len(train)),
        "valid": int(len(valid)),
        "test": int(len(test)),
        "timestamp_min": int(reviews["timestamp"].min()),
        "timestamp_max": int(reviews["timestamp"].max()),
        "avg_interactions_per_user": float(len(train) / max(train["user_id"].nunique(), 1)),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="All_Beauty")
    parser.add_argument("--limit", type=int, default=None, help="только для отладки")
    args = parser.parse_args()
    prepare(args.category, limit=args.limit)


if __name__ == "__main__":
    main()
