"""Протокол sampled-softmax и bounded bootstrap проверяются отдельно от модели."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.eval.block2 import bootstrap_difference, select_negatives


def test_negative_sampler_excludes_only_observed_prefix_and_target() -> None:
    history = np.array([[-1, 0, 1], [-1, -1, 1]])
    targets = np.array([2, 2])
    negatives = select_negatives(history, targets, np.arange(5), 100, np.random.default_rng(42))
    assert not set(negatives[0]) & {0, 1, 2}
    assert not set(negatives[1]) & {1, 2}
    assert 0 in negatives[1]
    with pytest.raises(ValueError, match="нет train-catalog"):
        select_negatives(
            np.array([[0, 1]]), np.array([2]), np.arange(3), 2, np.random.default_rng(0)
        )


def test_paired_bootstrap_identical_and_constant_difference() -> None:
    same = bootstrap_difference(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert same["ci_low"] == same["ci_high"] == 0
    better = bootstrap_difference(np.array([2.0, 3.0]), np.array([1.0, 2.0]))
    assert better["mean_difference"] == better["ci_low"] == better["ci_high"] == 1
