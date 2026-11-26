"""Инварианты OneTrans: mixed параметры, pyramid, leakage и cache equivalence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from recsys.data.dataset import InteractionData
from recsys.models.onetrans import OneTrans, OneTransBlock, OneTransConfig
from recsys.models.onetrans_features import OneTransFeatures, chronological_merge
from recsys.models.onetrans_recommender import (
    OneTransRecommender,
    dataset_fingerprint,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def network(toy_data: InteractionData) -> OneTrans:
    torch.manual_seed(8)
    torch.set_num_threads(1)
    features = OneTransFeatures.from_data(toy_data)
    return OneTrans(
        OneTransConfig(width=8, heads=2, history_length=4, pyramid=(2, 0)),
        features.torch_table(),
        features.feature_sizes,
        toy_data.n_users,
    ).eval()


def test_cache_matches_combined_graph_with_padding_and_empty(network: OneTrans) -> None:
    history = torch.tensor([[-1, 0, 1, 2], [-1, -1, -1, -1]])
    users = torch.tensor([0, 1])
    candidates = torch.tensor([[3, 4], [4, 3]])
    cached = network(history, users, candidates)
    reference = torch.stack(
        [network.uncached(history, users, candidates[:, index]) for index in range(2)], dim=1
    )
    torch.testing.assert_close(cached, reference, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(cached).all()


def test_pyramid_queries_see_full_keys_before_slicing() -> None:
    torch.manual_seed(2)
    block = OneTransBlock(OneTransConfig(width=8, heads=2)).eval()
    x = torch.randn(2, 9, 8)
    valid = torch.ones(2, 6, dtype=torch.bool)
    full, _ = block(x, valid, 6)
    pruned, retained = block(x, valid, 2)
    torch.testing.assert_close(pruned, full[:, -5:], rtol=1e-5, atol=1e-6)
    assert retained.shape == (2, 2)


def test_causal_future_and_ns_do_not_change_prefix() -> None:
    block = OneTransBlock(OneTransConfig(width=8, heads=2)).eval()
    original = torch.randn(1, 7, 8)
    changed = original.clone()
    changed[:, 2:] += 10
    valid = torch.ones(1, 4, dtype=torch.bool)
    a, _ = block(original, valid, 4)
    b, _ = block(changed, valid, 4)
    torch.testing.assert_close(a[:, :2], b[:, :2])


def test_ns_parameters_are_specific_and_trainable(network: OneTrans) -> None:
    block = network.blocks[0]
    a, b = block.qkv.specific[:2]
    assert a.weight.data_ptr() != b.weight.data_ptr()
    candidates = torch.tensor([[1, 2], [2, 1]])
    loss = F.cross_entropy(
        network(torch.tensor([[0, 1], [1, 0]]), torch.tensor([0, 1]), candidates),
        torch.tensor([0, 0]),
    )
    loss.backward()
    for layer in block.qkv.specific:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
    assert block.qkv.shared.weight.grad is not None


def test_training_reduces_actual_loss(network: OneTrans) -> None:
    optimizer = torch.optim.Adam(network.parameters(), lr=0.02)
    histories = torch.tensor([[0, 1], [1, 0]])
    users = torch.tensor([0, 1])
    candidates = torch.tensor([[1, 2], [2, 1]])
    labels = torch.tensor([0, 0])
    first = float(F.cross_entropy(network(histories, users, candidates), labels).detach())
    for _ in range(12):
        optimizer.zero_grad()
        loss = F.cross_entropy(network(histories, users, candidates), labels)
        loss.backward()
        optimizer.step()
    last = float(F.cross_entropy(network(histories, users, candidates), labels).detach())
    assert last < first * 0.8


def test_train_features_and_examples_ignore_holdout(toy_data: InteractionData) -> None:
    before = OneTransFeatures.from_data(toy_data, metadata=True)
    toy_data.valid["timestamp"] = -100
    toy_data.test["item_id"] = "outside"
    toy_data.item_meta["average_rating"] = 1000
    toy_data.item_meta["rating_number"] = 1_000_000
    after = OneTransFeatures.from_data(toy_data, metadata=True)
    np.testing.assert_array_equal(before.item_features, after.item_features)
    np.testing.assert_array_equal(before.train_examples, after.train_examples)
    for user, position, target in before.train_examples:
        history = before.histories_for(np.array([user]), 4, np.array([position]))[0]
        np.testing.assert_array_equal(history[history >= 0], before.histories[user][:position])
        assert target == before.histories[user][position]
    # Только train товары получают обучаемый ID, остальные — unknown.
    assert np.all(after.item_features[1:][toy_data.item_counts == 0, 0] == 0)


def test_chronological_multistream_merge_preserves_real_type() -> None:
    items, types = chronological_merge(
        [(np.array([1, 3]), np.array([10, 30])), (np.array([2]), np.array([20]))]
    )
    np.testing.assert_array_equal(items, [1, 2, 3])
    np.testing.assert_array_equal(types, [0, 1, 0])


def test_checkpoint_mapping_and_chunk_invariance(
    network: OneTrans, toy_data: InteractionData, tmp_path: Path
) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(path, network, dataset_fingerprint(toy_data), False, {"epoch": 1})
    restored, _ = load_checkpoint(path)
    torch.testing.assert_close(network.head.weight, restored.head.weight)
    small = OneTransRecommender(path, candidate_batch=1).fit(toy_data)
    big = OneTransRecommender(path, candidate_batch=32).fit(toy_data)
    np.testing.assert_allclose(
        small.score_users(np.array([0])), big.score_users(np.array([0])), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_equal(
        small.recommend(np.array([0]), k=2), big.recommend(np.array([0]), k=2)
    )
    assert not set(small.recommend(np.array([0]), k=2)[0]) & set(toy_data.train_csr[0].indices)
    assert small.cache_counters["exact_hits"] > 0
    toy_data.train_frame.loc[0, "timestamp"] += 100
    with pytest.raises(ValueError, match="другом dataset"):
        OneTransRecommender(path).fit(toy_data)


def test_append_reuses_first_layer_kv_and_recomputes_deep(
    network: OneTrans, toy_data: InteractionData, tmp_path: Path
) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(path, network, dataset_fingerprint(toy_data), False, {})
    model = OneTransRecommender(path).fit(toy_data)
    model.prefix_for(0, np.array([0]))
    appended = model.prefix_for(0, np.array([0, 1]))
    fresh = model.model.encode_prefix(torch.tensor([[0, 1]]))
    for actual, expected in zip(appended.layers, fresh.layers, strict=True):
        torch.testing.assert_close(actual.keys, expected.keys, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.values, expected.values, atol=1e-6, rtol=1e-5)
    assert model.cache_counters["append_first_layer_kv_reused"] == 1
    assert model.cache_counters["deep_layers_recomputed"] == 2
    model.prefix_for(0, np.array([1, 2]))
    assert model.cache_counters["invalidations"] == 1
