"""Checkpoint inference и ограниченный по памяти каталог для OneTrans."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from recsys.data.dataset import InteractionData
from recsys.models.base import BaseRecommender
from recsys.models.onetrans import OneTrans, OneTransConfig, PrefixCache
from recsys.models.onetrans_features import IntArray, OneTransFeatures


def dataset_fingerprint(data: InteractionData) -> str:
    payload = {
        "users": data.users.tolist(),
        "items": data.items.tolist(),
        "train": data.train_frame[["user_id", "item_id", "timestamp"]].to_json(orient="split"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_checkpoint(
    path: Path, model: OneTrans, fingerprint: str, metadata: bool, extra: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.payload(),
            "feature_sizes": [field.num_embeddings for field in model.fields],
            "n_users": model.users.num_embeddings - 1,
            "state_dict": {
                key: tensor.detach().cpu() for key, tensor in model.state_dict().items()
            },
            "fingerprint": fingerprint,
            "metadata": metadata,
            "extra": extra,
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[OneTrans, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    raw_config = dict(payload["config"])
    raw_config["pyramid"] = tuple(raw_config["pyramid"])
    model = OneTrans(
        OneTransConfig(**raw_config),
        payload["state_dict"]["item_features"],
        tuple(payload["feature_sizes"]),
        int(payload["n_users"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


class OneTransRecommender(BaseRecommender):
    name = "onetrans"

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cpu",
        candidate_batch: int = 2048,
        cache_capacity: int = 128,
    ) -> None:
        super().__init__()
        if candidate_batch < 1 or cache_capacity < 1:
            raise ValueError("размер batch/cache должен быть положительным")
        self.checkpoint = checkpoint
        self.device = device
        self.candidate_batch = candidate_batch
        self.cache_capacity = cache_capacity
        self.prefix_cache: OrderedDict[int, tuple[tuple[int, ...], PrefixCache]] = OrderedDict()
        self.cache_counters = {
            "exact_hits": 0,
            "misses": 0,
            "append_first_layer_kv_reused": 0,
            "deep_layers_recomputed": 0,
            "invalidations": 0,
        }
        self.timings: list[float] = []

    def fit(self, data: InteractionData) -> OneTransRecommender:
        model, payload = load_checkpoint(self.checkpoint, self.device)
        if payload["fingerprint"] != dataset_fingerprint(data):
            raise ValueError("checkpoint обучен на другом dataset/split/mapping")
        self.model = model
        self.features = OneTransFeatures.from_data(data, metadata=bool(payload["metadata"]))
        self.prefix_cache.clear()
        super().fit(data)
        return self

    @torch.inference_mode()
    def prefix_for(self, user: int, history: IntArray) -> PrefixCache:
        key = tuple(int(v) for v in history[-self.model.config.history_length :])
        old = self.prefix_cache.get(user)
        if old is not None and old[0] == key:
            self.cache_counters["exact_hits"] += 1
            self.prefix_cache.move_to_end(user)
            return old[1]
        tensor = torch.tensor([key or (-1,)], dtype=torch.long, device=self.device)
        reusable = None
        reuse = 0
        if old is not None:
            previous = old[0]
            if previous and len(key) > len(previous) and key[: len(previous)] == previous:
                reusable = old[1]
                reuse = len(previous)
                self.cache_counters["append_first_layer_kv_reused"] += reuse
            else:
                self.cache_counters["invalidations"] += 1
        self.cache_counters["misses"] += 1
        self.cache_counters["deep_layers_recomputed"] += max(0, len(self.model.blocks) - 1)
        cache = self.model.encode_prefix(tensor, reusable, reuse)
        self.prefix_cache[user] = (key, cache)
        self.prefix_cache.move_to_end(user)
        while len(self.prefix_cache) > self.cache_capacity:
            self.prefix_cache.popitem(last=False)
        return cache

    @torch.inference_mode()
    def score_catalog(self, user: int, candidates: IntArray) -> np.ndarray:
        history = self.features.histories[user]
        prefix = self.prefix_for(user, history)
        ids = torch.as_tensor(candidates[None], device=self.device)
        scores = self.model.score_cached(
            prefix,
            torch.tensor([user], device=self.device),
            ids,
            torch.tensor([min(len(history), self.model.config.history_length)], device=self.device),
        )
        out: np.ndarray = scores.cpu().numpy()[0]
        return out

    def score_users(self, user_idx: IntArray) -> np.ndarray:
        data = self._require_data()
        out = np.empty((len(user_idx), data.n_items), dtype=np.float32)
        for row, user in enumerate(user_idx):
            for start in range(0, data.n_items, self.candidate_batch):
                ids = np.arange(start, min(start + self.candidate_batch, data.n_items))
                out[row, ids] = self.score_catalog(int(user), ids)
        return out

    def recommend(
        self,
        user_idx: IntArray,
        k: int = 10,
        exclude_seen: bool = True,
        batch_size: int = 512,
    ) -> IntArray:
        data = self._require_data()
        if k < 1 or k > data.n_items:
            raise ValueError("k вне размера каталога")
        out = np.empty((len(user_idx), k), dtype=np.int64)
        for row, uid in enumerate(user_idx):
            started = time.perf_counter()
            best_ids = np.array([], dtype=np.int64)
            best_scores = np.array([], dtype=np.float32)
            seen = (
                data.train_csr[int(uid)].indices if exclude_seen else np.array([], dtype=np.int64)
            )
            if data.n_items - len(seen) < k:
                raise ValueError("недостаточно unseen товаров для k")
            for start in range(0, data.n_items, self.candidate_batch):
                ids = np.arange(start, min(start + self.candidate_batch, data.n_items))
                scores = self.score_catalog(int(uid), ids)
                scores[np.isin(ids, seen)] = -np.inf
                merged_ids = np.concatenate([best_ids, ids])
                merged_scores = np.concatenate([best_scores, scores])
                # Вторичный item ID делает tie-break независимым от размера chunks.
                order = np.lexsort((merged_ids, -merged_scores))[:k]
                best_ids, best_scores = merged_ids[order], merged_scores[order]
            out[row] = best_ids
            self.timings.append(time.perf_counter() - started)
        return out
