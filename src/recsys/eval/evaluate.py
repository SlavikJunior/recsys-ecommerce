"""Запуск оценки рекомендателей на одном протоколе и сохранение артефактов."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.config import ARTIFACTS_DIR
from recsys.data.dataset import InteractionData, load_dataset
from recsys.eval.metrics import summarize
from recsys.models.base import BaseRecommender
from recsys.models.content import ContentBasedRecommender, build_item_text
from recsys.models.heuristics import AttributeAffinityRecommender
from recsys.models.hybrid import HybridRecommender
from recsys.models.item_cf import ItemCFRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerRecommender
from recsys.models.user_cf import UserCFRecommender

# K фиксируем заранее: сравнение моделей на разных K — частая ошибка,
# при которой «лучшая» модель меняется от строки к строке отчёта.
K_EVAL = (10, 20)


def load_hybrid_weights(category: str) -> tuple[list[float], str]:
    """Веса гибрида из artifacts/eval; иначе — равные веса как безопасный дефолт."""
    path = ARTIFACTS_DIR / "eval" / f"hybrid_tuning_{category}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        return list(payload["best_weights"]), str(path.name)
    return [0.25, 0.25, 0.5, 1.0], "default"


def build_models(names: list[str], category: str = "Video_Games") -> dict[str, BaseRecommender]:
    weights, _ = load_hybrid_weights(category)
    factories: dict[str, Any] = {
        "popularity": lambda: PopularityRecommender(decay=False),
        "popularity_decay": lambda: PopularityRecommender(decay=True, half_life_days=30),
        "popularity_decay_365d": lambda: PopularityRecommender(decay=True, half_life_days=365),
        "brand_affinity": lambda: AttributeAffinityRecommender("brand"),
        "category_affinity": lambda: AttributeAffinityRecommender("top_category"),
        "content_based": lambda: ContentBasedRecommender(),
        "item_cf": lambda: ItemCFRecommender(),
        "user_cf": lambda: UserCFRecommender(),
        "hybrid": lambda: HybridRecommender(
            [
                PopularityRecommender(decay=True, half_life_days=365),
                ContentBasedRecommender(),
                ItemCFRecommender(),
                UserCFRecommender(),
            ],
            weights=weights,
        ),
        "two_tower": lambda: TwoTowerRecommender(checkpoint_dir=ARTIFACTS_DIR / "two_tower"),
    }
    unknown = [name for name in names if name not in factories]
    if unknown:
        raise ValueError(f"неизвестные модели: {unknown}; доступны: {sorted(factories)}")
    return {name: factories[name]() for name in names}


def _segment_report(
    recs: np.ndarray,
    targets: np.ndarray,
    user_activity: np.ndarray,
    item_popularity: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Метрики по сегментам: активность пользователя и популярность таргета."""
    from recsys.eval.metrics import hit_rank, ndcg_at_k, recall_at_k

    ranks = hit_rank(recs, targets)
    buckets = {
        "users_5_9": (user_activity >= 5) & (user_activity < 10),
        "users_10_19": (user_activity >= 10) & (user_activity < 20),
        "users_20_plus": user_activity >= 20,
        "item_tail_lt50": item_popularity < 50,
        "item_mid_50_500": (item_popularity >= 50) & (item_popularity < 500),
        "item_head_500_plus": item_popularity >= 500,
    }
    out: dict[str, dict[str, float]] = {}
    for name, mask in buckets.items():
        if mask.sum() == 0:
            continue
        out[name] = {
            "n": int(mask.sum()),
            "recall@10": recall_at_k(ranks[mask], 10),
            "ndcg@10": ndcg_at_k(ranks[mask], 10),
        }
    return out


def _content_vectors(data: InteractionData) -> sp.csr_matrix:
    """Векторы для diversity: те же, что в content-based, но строятся один раз."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(
        max_features=50_000, min_df=2, ngram_range=(1, 2), sublinear_tf=True
    )
    return normalize(vectorizer.fit_transform(build_item_text(data))).tocsr()


def evaluate(
    model: BaseRecommender,
    data: InteractionData,
    item_vectors: sp.csr_matrix,
    k: int = 20,
    max_users: int | None = None,
    split: str = "test",
) -> dict[str, Any]:
    user_idx, targets = data.test_targets() if split == "test" else data.valid_targets()
    if max_users is not None and len(user_idx) > max_users:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(user_idx), size=max_users, replace=False)
        user_idx, targets = user_idx[pick], targets[pick]

    started = time.perf_counter()
    model.fit(data)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    recs = model.recommend(user_idx, k=k)
    recommend_seconds = time.perf_counter() - started

    metrics = summarize(
        recs,
        targets,
        k_values=K_EVAL,
        item_counts=data.item_counts,
        item_vectors=item_vectors,
        n_items=data.n_items,
    )
    user_activity = np.asarray(data.train_csr[user_idx].sum(axis=1)).ravel()
    item_popularity = data.item_counts[targets]
    metrics["users_evaluated"] = float(len(user_idx))
    metrics["fit_seconds"] = fit_seconds
    metrics["recommend_seconds"] = recommend_seconds
    return {
        "model": model.name,
        "metrics": metrics,
        "segments": _segment_report(recs, targets, user_activity, item_popularity),
    }


def to_markdown(results: list[dict[str, Any]]) -> str:
    rows = []
    for item in results:
        m = item["metrics"]
        rows.append(
            {
                "model": item["model"],
                "recall@10": f"{m['recall@10']:.4f}",
                "ndcg@10": f"{m['ndcg@10']:.4f}",
                "recall@20": f"{m['recall@20']:.4f}",
                "ndcg@20": f"{m['ndcg@20']:.4f}",
                "coverage@20": f"{m['coverage@20']:.4f}",
                "novelty@20": f"{m['novelty@20']:.2f}",
                "diversity@20": f"{m['diversity@20']:.3f}",
                "fit_s": f"{m['fit_seconds']:.1f}",
                "rec_s": f"{m['recommend_seconds']:.1f}",
            }
        )
    frame = pd.DataFrame(rows)
    return str(frame.to_markdown(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument(
        "--models",
        default="popularity,popularity_decay,brand_affinity,category_affinity,"
        "content_based,item_cf,user_cf,hybrid",
    )
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--merge", action="store_true", help="дополнить существующий JSON вместо перезаписи"
    )
    parser.add_argument("--split", choices=["test", "valid"], default="test")
    args = parser.parse_args()

    data = load_dataset(args.category)
    print(f"данные: users={data.n_users} items={data.n_items} train={len(data.train_frame)}")
    vectors = _content_vectors(data)

    results = []
    for model in build_models(
        [m.strip() for m in args.models.split(",")], category=args.category
    ).values():
        result = evaluate(model, data, vectors, max_users=args.max_users, split=args.split)
        results.append(result)
        m = result["metrics"]
        print(f"--- {model.name}")
        print(f"    recall@10={m['recall@10']:.4f} ndcg@10={m['ndcg@10']:.4f}")

    results.sort(key=lambda r: r["metrics"]["ndcg@10"], reverse=True)
    out_path = args.output or (ARTIFACTS_DIR / "eval" / f"baselines_{args.category}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.merge and out_path.exists():
        previous = {row["model"]: row for row in json.loads(out_path.read_text())}
        previous.update({row["model"]: row for row in results})
        results = list(previous.values())
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    table = to_markdown(results)
    out_path.with_suffix(".md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
