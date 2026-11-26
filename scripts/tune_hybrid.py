"""Подбор весов гибрида на validation-сплите.

Тест не участвует ни в выборе весов, ни в ранней остановке: иначе метрики
в отчёте будут завышены, а защита модели на новых данных — фикцией.
Поиск случайный, а не полный перебор: 5^4 вариантов на 15 тыс. пользователей
дороже, чем даёт прирост.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from recsys.config import ARTIFACTS_DIR
from recsys.data.dataset import load_dataset
from recsys.eval.metrics import hit_rank, ndcg_at_k
from recsys.models.content import ContentBasedRecommender
from recsys.models.hybrid import HybridRecommender
from recsys.models.item_cf import ItemCFRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.user_cf import UserCFRecommender

SOURCES = [
    ("popularity", PopularityRecommender(decay=True, half_life_days=365)),
    ("content", ContentBasedRecommender()),
    ("item_cf", ItemCFRecommender()),
    ("user_cf", UserCFRecommender()),
]
GRID = [0.0, 0.25, 0.5, 1.0, 2.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--draws", type=int, default=25)
    parser.add_argument("--users", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = load_dataset(args.category)
    user_idx, targets = data.valid_targets()
    rng = np.random.default_rng(args.seed)
    if len(user_idx) > args.users:
        pick = rng.choice(len(user_idx), size=args.users, replace=False)
        user_idx, targets = user_idx[pick], targets[pick]

    print("обучаю источники на train...")
    for _, model in SOURCES:
        model.fit(data)

    results = []
    seen: set[tuple[float, ...]] = set()
    for _ in range(args.draws):
        weights = tuple(float(rng.choice(GRID)) for _ in SOURCES)
        if sum(weights) == 0 or weights in seen:
            continue
        seen.add(weights)
        hybrid = HybridRecommender(
            [model for _, model in SOURCES],
            weights=list(weights),
            fit_sources=False,
        ).fit(data)
        recs = hybrid.recommend(user_idx, k=10)
        ranks = hit_rank(recs, targets)
        score = ndcg_at_k(ranks, 10)
        results.append({"weights": weights, "ndcg@10": score})
        print(
            " + ".join(f"{n}={w}" for (n, _), w in zip(SOURCES, weights, strict=True)),
            f"-> {score:.4f}",
        )

    # Плюс чистые источники как контроль: гибрид обязан быть не хуже лучшего из них.
    for i, (name, model) in enumerate(SOURCES):
        recs = model.recommend(user_idx, k=10)
        score = ndcg_at_k(hit_rank(recs, targets), 10)
        weights = tuple(1.0 if j == i else 0.0 for j in range(len(SOURCES)))
        results.append({"weights": weights, "ndcg@10": score})
        print(f"{name} alone -> {score:.4f}")

    best = max(results, key=lambda r: r["ndcg@10"])
    payload = {
        "source_order": [name for name, _ in SOURCES],
        "best_weights": list(best["weights"]),
        "best_valid_ndcg@10": best["ndcg@10"],
        "trials": results,
    }
    out = ARTIFACTS_DIR / "eval" / f"hybrid_tuning_{args.category}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in ("best_weights", "best_valid_ndcg@10")}))


if __name__ == "__main__":
    main()
