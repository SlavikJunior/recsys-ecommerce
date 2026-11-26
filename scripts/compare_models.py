"""Статистическое сравнение двух моделей на одном сплите.

Offline-метрики без доверительных интервалов — это лотерея: разница в 0.002
NDCG может быть шумом. Скрипт считает paired bootstrap по пользователям и
тест МакНемара на попаданиях, чтобы в отчёте были не «числа», а выводы.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from recsys.config import ARTIFACTS_DIR
from recsys.data.dataset import load_dataset
from recsys.eval.evaluate import _content_vectors, build_models
from recsys.eval.metrics import hit_rank

RNG = np.random.default_rng(1234)


def bootstrap_ci(
    per_user_a: np.ndarray, per_user_b: np.ndarray, iterations: int = 2000
) -> dict[str, float]:
    """Paired bootstrap: ресемплируем пользователей, считаем разницу средних."""
    diff = per_user_a - per_user_b
    n = len(diff)
    idx = RNG.integers(0, n, size=(iterations, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # p-value: доля бутстреп-выборок, где знак разницы противоположен оценке.
    observed = diff.mean()
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "mean_diff": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "n_users": int(n),
    }


def mcnemar(hits_a: np.ndarray, hits_b: np.ndarray) -> dict[str, float]:
    """Сравнение попаданий: сколько раз A выиграл, B выиграл, и p-value."""
    only_a = int(np.sum(hits_a & ~hits_b))
    only_b = int(np.sum(~hits_a & hits_b))
    # Точный биномиальный тест без scipy.stats: при больших n — нормальная аппроксимация.
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    k = min(only_a, only_b)
    from math import comb, log

    log_pmf = sum(
        log(comb(n, i)) - n * log(2) for i in range(0, k + 1)
    )
    p = min(1.0, 2 * float(np.exp(log_pmf)))
    return {"only_a": only_a, "only_b": only_b, "p_value": p}


def per_user_ndcg(ranks: np.ndarray, k: int) -> np.ndarray:
    safe = np.maximum(ranks, 0)
    return np.where((ranks >= 0) & (ranks < k), 1.0 / np.log2(safe + 2.0), 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--a", default="two_tower")
    parser.add_argument("--b", default="hybrid")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    data = load_dataset(args.category)
    vectors = _content_vectors(data)
    user_idx, targets = data.test_targets()
    models = build_models([args.a, args.b])

    ranks = {}
    for name, model in models.items():
        model.fit(data)
        recs = model.recommend(user_idx, k=args.k)
        ranks[name] = hit_rank(recs, targets)
        hits = (ranks[name] >= 0) & (ranks[name] < args.k)
        print(f"{name}: recall@{args.k}={hits.mean():.4f}")

    result = {
        "a": args.a,
        "b": args.b,
        "k": args.k,
        "ndcg": bootstrap_ci(
            per_user_ndcg(ranks[args.a], args.k), per_user_ndcg(ranks[args.b], args.k)
        ),
        "hit_rate": mcnemar(ranks[args.a][:] >= 0, ranks[args.b][:] >= 0),
    }
    out = ARTIFACTS_DIR / "eval" / f"compare_{args.a}_vs_{args.b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
