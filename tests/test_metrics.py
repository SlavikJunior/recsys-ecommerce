"""Юнит-тесты метрик: проверяем формулы на маленьких примерах."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from recsys.eval.metrics import (
    coverage_at_k,
    gini_index,
    hit_rank,
    intra_list_diversity,
    mrr_at_k,
    ndcg_at_k,
    novelty_at_k,
    recall_at_k,
)


def test_hit_rank_positions() -> None:
    recs = np.array([[3, 5, 7], [1, 2, 3], [0, 1, 2]])
    targets = np.array([5, 2, 9])
    assert hit_rank(recs, targets).tolist() == [1, 1, -1]


def test_recall_and_ndcg() -> None:
    ranks = np.array([0, 1, -1, 2])
    assert recall_at_k(ranks, 2) == 0.5
    # ndcg: 1/1 + 1/log2(3) + 0 + 1/log2(4), делим на 4 пользователя
    expected = (1.0 + 1.0 / np.log2(3) + 1.0 / np.log2(4)) / 4
    assert abs(ndcg_at_k(ranks, 3) - expected) < 1e-9
    assert ndcg_at_k(ranks, 1) == 0.25


def test_mrr() -> None:
    ranks = np.array([0, 1, -1])
    assert abs(mrr_at_k(ranks, 10) - (1 + 0.5) / 3) < 1e-9


def test_coverage_and_gini() -> None:
    recs = np.array([[0, 1], [0, 1]])
    assert coverage_at_k(recs, n_items=4) == 0.5
    # Половина каталога не показывается ни разу — это и есть Gini 0.5.
    assert abs(gini_index(recs, n_items=4) - 0.5) < 1e-9
    # Равномерные показы по всему каталогу — Gini 0.
    uniform = np.array([[0, 1], [2, 3]])
    assert gini_index(uniform, n_items=4) < 1e-9


def test_novelty_prefers_tail() -> None:
    counts = np.array([100, 1, 1, 1], dtype=np.float64)
    tail = novelty_at_k(np.array([[1, 2]]), counts, n_users=100)
    head = novelty_at_k(np.array([[0, 1]]), counts, n_users=100)
    assert tail > head


def test_diversity_identical_items() -> None:
    vectors = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    same = intra_list_diversity(np.array([[0, 1]]), vectors)
    different = intra_list_diversity(np.array([[0, 2]]), vectors)
    assert same < 1e-6
    assert abs(different - 1.0) < 1e-6
