"""Метрики ранжирования. Чистые функции — их легко тестировать и переиспользовать.

Все метрики считают один релевантный товар на пользователя (leave-last-out),
поэтому MAP@k и MRR@k совпадают по значению, но обе остаются в отчёте: так
результат сопоставим с публикациями, где протокол другой.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def hit_rank(recommended: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Позиция таргета в рекомендациях (0-based), -1 если не попал.

    recommended: (n_users, k) индексы товаров, target: (n_users,)
    """
    matches = recommended == target[:, None]
    has_hit = matches.any(axis=1)
    ranks = np.where(has_hit, matches.argmax(axis=1), -1)
    return ranks.astype(np.int64)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean((ranks >= 0) & (ranks < k)))


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    # np.maximum убирает деление на ноль в отброшенной ветке np.where.
    safe = np.maximum(ranks, 0)
    gains = np.where((ranks >= 0) & (ranks < k), 1.0 / np.log2(safe + 2.0), 0.0)
    return float(np.mean(gains))


def mrr_at_k(ranks: np.ndarray, k: int) -> float:
    safe = np.maximum(ranks, 0)
    values = np.where((ranks >= 0) & (ranks < k), 1.0 / (safe + 1.0), 0.0)
    return float(np.mean(values))


def coverage_at_k(recommended: np.ndarray, n_items: int) -> float:
    """Доля каталога, попавшая в рекомендации хотя бы одному пользователю."""
    return float(np.unique(recommended).size / max(n_items, 1))


def gini_index(recommended: np.ndarray, n_items: int) -> float:
    """Неравенство распределения показов: 0 — равномерно, 1 — всё одному товару."""
    counts = np.bincount(recommended.ravel(), minlength=n_items).astype(np.float64)
    if counts.sum() == 0:
        return 0.0
    counts = np.sort(counts)
    n = counts.size
    index = np.arange(1, n + 1)
    return float((2 * (index * counts).sum()) / (n * counts.sum()) - (n + 1) / n)


def novelty_at_k(recommended: np.ndarray, item_counts: np.ndarray, n_users: int) -> float:
    """Средняя «неожиданность» рекомендаций: -log2(P(item)).

    item_counts — число взаимодействий товара в train, n_users — база для
    вероятности. Чем популярнее товар, тем ниже novelty.
    """
    probs = np.clip(item_counts / max(n_users, 1), 1e-12, 1.0)
    values = -np.log2(probs[recommended])
    return float(values.mean())


def intra_list_diversity(
    recommended: np.ndarray, item_vectors: sp.csr_matrix, sample_users: int = 5000
) -> float:
    """Средняя внутрисписочная непохожесть: 1 - средний косинус между товарами.

    Для скорости считаем по выборке пользователей: на 10k+ пользователей и
    k=10 полный перебор даёт десятки миллионов скалярных произведений.
    """
    if recommended.shape[0] > sample_users:
        rng = np.random.default_rng(0)
        rows = rng.choice(recommended.shape[0], size=sample_users, replace=False)
        recs = recommended[rows]
    else:
        recs = recommended

    norms = np.sqrt(np.asarray(item_vectors.multiply(item_vectors).sum(axis=1))).ravel()
    norms[norms == 0] = 1.0
    scores: list[float] = []
    for row in recs:
        vecs = item_vectors[row]
        sims = (vecs @ vecs.T).toarray()
        sims = sims / np.outer(norms[row], norms[row])
        k = len(row)
        if k < 2:
            continue
        off_diag = (sims.sum() - np.trace(sims)) / (k * (k - 1))
        scores.append(1.0 - float(off_diag))
    return float(np.mean(scores)) if scores else 0.0


def summarize(
    recommended: np.ndarray,
    targets: np.ndarray,
    k_values: tuple[int, ...] = (10, 20),
    item_counts: np.ndarray | None = None,
    item_vectors: sp.csr_matrix | None = None,
    n_items: int | None = None,
) -> dict[str, float]:
    ranks = hit_rank(recommended, targets)
    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = recall_at_k(ranks, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranks, k)
        out[f"mrr@{k}"] = mrr_at_k(ranks, k)
        out[f"map@{k}"] = mrr_at_k(ranks, k)
    max_k = recommended.shape[1]
    if n_items is not None:
        out[f"coverage@{max_k}"] = coverage_at_k(recommended, n_items)
        out[f"gini@{max_k}"] = gini_index(recommended, n_items)
    if item_counts is not None:
        out[f"novelty@{max_k}"] = novelty_at_k(recommended, item_counts, n_users=len(targets))
    if item_vectors is not None:
        out[f"diversity@{max_k}"] = intra_list_diversity(recommended, item_vectors)
    return out
