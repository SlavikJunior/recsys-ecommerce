"""Сборка сервисных артефактов из обученных моделей.

Один шаг офлайн-пайплайна: обучить/загрузить модели и выгрузить всё, что
нужно онлайн-сервису, чтобы он не зависел от Python-объектов обучения.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.config import ARTIFACTS_DIR
from recsys.data.dataset import load_dataset
from recsys.models.item_cf import ItemCFRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerRecommender
from recsys.serving.registry import SERVING_DIR


def build(category: str) -> None:
    started = time.perf_counter()
    data = load_dataset(category)

    two_tower = TwoTowerRecommender(checkpoint_dir=ARTIFACTS_DIR / "two_tower").fit(data)
    assert two_tower.user_embeddings is not None and two_tower.item_embeddings is not None

    item_cf = ItemCFRecommender().fit(data)
    assert item_cf.item_similarity is not None

    popularity = PopularityRecommender(decay=True)
    popularity.fit(data)
    scores = popularity.scores

    out = SERVING_DIR / category
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"user_id": data.users.to_numpy()}).to_parquet(
        out / "user_ids.parquet", index=False
    )
    pd.DataFrame({"item_id": data.items.to_numpy()}).to_parquet(
        out / "item_ids.parquet", index=False
    )
    data.item_meta.to_parquet(out / "item_meta.parquet", index=False)
    np.save(out / "user_embeddings.npy", two_tower.user_embeddings.astype(np.float32))
    np.save(out / "item_embeddings.npy", two_tower.item_embeddings.astype(np.float32))
    np.save(out / "popularity.npy", scores.astype(np.float32))
    np.save(out / "item_counts.npy", data.item_counts.astype(np.float32))
    sp.save_npz(out / "seen_csr.npz", data.train_csr.tocsr())
    sp.save_npz(out / "item_similarity.npz", item_cf.item_similarity.tocsr())

    manifest = {
        "category": category,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_users": int(data.n_users),
        "n_items": int(data.n_items),
        "embedding_dim": int(two_tower.item_embeddings.shape[1]),
        "models": ["two_tower", "item_cf", "popularity_decay"],
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Video_Games")
    args = parser.parse_args()
    build(args.category)


if __name__ == "__main__":
    main()
