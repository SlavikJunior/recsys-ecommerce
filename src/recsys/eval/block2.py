"""Воспроизводимые train/benchmark/eval команды второго блока без старых artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from recsys.config import processed_dir
from recsys.data.dataset import InteractionData, load_dataset
from recsys.eval.metrics import hit_rank, summarize
from recsys.models.base import BaseRecommender
from recsys.models.content import ContentBasedRecommender
from recsys.models.hybrid import HybridRecommender
from recsys.models.item_cf import ItemCFRecommender
from recsys.models.onetrans import OneTrans, OneTransConfig
from recsys.models.onetrans_features import IntArray, OneTransFeatures
from recsys.models.onetrans_recommender import (
    OneTransRecommender,
    dataset_fingerprint,
    save_checkpoint,
)
from recsys.models.popularity import PopularityRecommender
from recsys.models.user_cf import UserCFRecommender


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    selection_seed: int = 20261126
    epochs: int = 3
    batch_size: int = 256
    negatives: int = 15
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    validation_users: int = 256
    test_users: int = 1000
    candidate_batch: int = 2048
    metadata: bool = False
    threads: int = 4

    def __post_init__(self) -> None:
        values = (
            self.epochs,
            self.batch_size,
            self.negatives,
            self.validation_users,
            self.test_users,
            self.candidate_batch,
            self.threads,
        )
        if any(v < 1 for v in values) or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("неверная конфигурация эксперимента")


def read_config(path: Path) -> tuple[OneTransConfig, ExperimentConfig]:
    payload = tomllib.loads(path.read_text())
    architecture = payload.get("model", {})
    if "pyramid" in architecture:
        architecture["pyramid"] = tuple(architecture["pyramid"])
    return OneTransConfig(**architecture), ExperimentConfig(**payload.get("experiment", {}))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def memory_mb() -> float:
    # Darwin ru_maxrss измеряется байтами, Linux — KiB.
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor)


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def freeze_selection(data: InteractionData, output: Path, config: ExperimentConfig) -> None:
    path = output / "selection.npz"
    if path.exists():
        return
    rng = np.random.default_rng(config.selection_seed)
    arrays = {}
    for split, limit in (("valid", config.validation_users), ("test", config.test_users)):
        users, targets = data.valid_targets() if split == "valid" else data.test_targets()
        selected = np.sort(rng.choice(len(users), size=min(limit, len(users)), replace=False))
        arrays[f"{split}_users"] = users[selected]
        arrays[f"{split}_targets"] = targets[selected]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        valid_users=arrays["valid_users"],
        valid_targets=arrays["valid_targets"],
        test_users=arrays["test_users"],
        test_targets=arrays["test_targets"],
    )


def provenance(
    data: InteractionData, category: str, config_path: Path, device: str
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "models" / "onetrans.py",
        root / "models" / "onetrans_features.py",
        root / "models" / "onetrans_recommender.py",
        Path(__file__).resolve(),
    ]
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "category": category,
        "dataset_fingerprint": dataset_fingerprint(data),
        "input_sha256": {
            p.name: file_sha(p) for p in sorted(processed_dir(category).glob("*.parquet"))
        },
        "source_sha256": {str(p.relative_to(root)): file_sha(p) for p in sources},
        "config_sha256": file_sha(config_path),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": device,
        "users": data.n_users,
        "items": data.n_items,
        "train_events": len(data.train_frame),
        "protocol": "leave-last-out; train-only histories for valid/test; closed-world catalog; "
        "train-seen excluded; full catalog scores; fixed uniform users before training",
        "objective": "sampled next-item softmax; alternatives are not observed negative clicks",
        "metadata_note": "average_rating/rating_number never used; metadata disabled by default",
    }


def select_negatives(
    histories: IntArray,
    targets: IntArray,
    observed: IntArray,
    count: int,
    rng: np.random.Generator,
) -> IntArray:
    out = rng.choice(observed, size=(len(targets), count))
    for row, target in enumerate(targets):
        forbidden = np.append(histories[row][histories[row] >= 0], target)
        if len(forbidden) >= len(observed) and np.isin(observed, forbidden).all():
            raise ValueError("нет train-catalog альтернатив для данного префикса")
        invalid = np.isin(out[row], forbidden)
        while invalid.any():
            out[row, invalid] = rng.choice(observed, size=int(invalid.sum()))
            invalid = np.isin(out[row], forbidden)
    return out.astype(np.int64)


def train_step(
    model: OneTrans,
    features: OneTransFeatures,
    examples: IntArray,
    observed: IntArray,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    rng: np.random.Generator,
    device: str,
) -> float:
    histories = features.histories_for(examples[:, 0], model.config.history_length, examples[:, 1])
    negatives = select_negatives(histories, examples[:, 2], observed, config.negatives, rng)
    candidates = np.column_stack([examples[:, 2], negatives])
    optimizer.zero_grad(set_to_none=True)
    logits = model(
        torch.as_tensor(histories, device=device),
        torch.as_tensor(examples[:, 0], device=device),
        torch.as_tensor(candidates, device=device),
    )
    loss = F.cross_entropy(logits, torch.zeros(len(examples), dtype=torch.long, device=device))
    if not torch.isfinite(loss):
        raise RuntimeError("обучение остановлено: non-finite loss")
    loss.backward()  # type: ignore[no-untyped-call]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach().cpu())


def attached(
    model: OneTrans,
    features: OneTransFeatures,
    data: InteractionData,
    config: ExperimentConfig,
    device: str,
) -> OneTransRecommender:
    wrapper = OneTransRecommender(Path("unused.pt"), device, config.candidate_batch)
    wrapper.model = model.eval()
    wrapper.features = features
    wrapper.data = data
    return wrapper


def benchmark(
    data: InteractionData,
    architecture: OneTransConfig,
    config: ExperimentConfig,
    output: Path,
    device: str,
) -> None:
    features = OneTransFeatures.from_data(data, config.metadata)
    model = OneTrans(architecture, features.torch_table(), features.feature_sizes, data.n_users).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    observed = np.flatnonzero(data.item_counts > 0)
    started = time.perf_counter()
    for _ in range(12):
        rows = rng.choice(len(features.train_examples), config.batch_size, replace=False)
        train_step(
            model, features, features.train_examples[rows], observed, optimizer, config, rng, device
        )
    sync(device)
    train_seconds = time.perf_counter() - started
    wrapper = attached(model, features, data, config, device)
    selected = np.load(output / "selection.npz", allow_pickle=False)["test_users"][:10]
    started = time.perf_counter()
    wrapper.recommend(selected, k=20)
    sync(device)
    scoring_seconds = time.perf_counter() - started
    result = {
        "device": device,
        "batch_size": config.batch_size,
        "train_step_count": 12,
        "train_step_seconds": train_seconds / 12,
        "train_examples": len(features.train_examples),
        "estimated_epoch_seconds": train_seconds
        / 12
        * np.ceil(len(features.train_examples) / config.batch_size),
        "scored_users": len(selected),
        "catalog_items": data.n_items,
        "scoring_seconds": scoring_seconds,
        "estimated_test_seconds": scoring_seconds / len(selected) * config.test_users,
        "peak_rss_mb": memory_mb(),
        "cache_counters": wrapper.cache_counters,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "mps_allocated_mb": torch.mps.current_allocated_memory() / 1024**2
        if device == "mps"
        else None,
    }
    write_json(output / f"benchmark_{device}.json", result)
    print(json.dumps(result), flush=True)


def train(
    data: InteractionData,
    architecture: OneTransConfig,
    config: ExperimentConfig,
    output: Path,
    device: str,
) -> None:
    if (output / "model.pt").exists():
        raise FileExistsError("output уже содержит checkpoint; выбери новую директорию")
    features = OneTransFeatures.from_data(data, config.metadata)
    model = OneTrans(architecture, features.torch_table(), features.feature_sizes, data.n_users).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed)
    observed = np.flatnonzero(data.item_counts > 0)
    selected = np.load(output / "selection.npz", allow_pickle=False)
    valid_users, valid_targets = selected["valid_users"], selected["valid_targets"]
    fingerprint = dataset_fingerprint(data)
    best = -float("inf")
    log = []
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        permutation = rng.permutation(len(features.train_examples))
        total_loss = 0.0
        seen = 0
        for offset in range(0, len(permutation), config.batch_size):
            examples = features.train_examples[permutation[offset : offset + config.batch_size]]
            loss = train_step(model, features, examples, observed, optimizer, config, rng, device)
            total_loss += loss * len(examples)
            seen += len(examples)
            if offset // config.batch_size % 200 == 0:
                print(
                    json.dumps({"epoch": epoch, "examples": seen, "loss": total_loss / seen}),
                    flush=True,
                )
        sync(device)
        train_seconds = time.perf_counter() - started
        validation_started = time.perf_counter()
        wrapper = attached(model, features, data, config, device)
        recommendations = wrapper.recommend(valid_users, k=20)
        metrics = summarize(recommendations, valid_targets, k_values=(10, 20), n_items=data.n_items)
        record = {
            "epoch": epoch,
            "examples": seen,
            "loss": total_loss / seen,
            "train_seconds": train_seconds,
            "validation_seconds": time.perf_counter() - validation_started,
            "validation": metrics,
            "peak_rss_mb": memory_mb(),
        }
        log.append(record)
        write_json(output / "training.json", log)
        if metrics["ndcg@10"] > best:
            best = metrics["ndcg@10"]
            save_checkpoint(
                output / "model.pt",
                model,
                fingerprint,
                config.metadata,
                {"epoch": epoch, "validation_ndcg@10": best, "seed": config.seed},
            )
        print(json.dumps(record), flush=True)
    write_json(
        output / "checkpoint.json",
        {
            "sha256": file_sha(output / "model.pt"),
            "best_valid_ndcg@10": best,
            "epochs": config.epochs,
            "selection": "valid only",
        },
    )


def bootstrap_difference(a: np.ndarray, b: np.ndarray, seed: int = 42) -> dict[str, float]:
    diff = a - b
    if not len(diff):
        raise ValueError("пустая выборка bootstrap")
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "users": float(len(diff)),
        "resamples": 2000.0,
    }


def segment_metrics(
    recs: IntArray, users: IntArray, targets: IntArray, data: InteractionData
) -> dict[str, Any]:
    activity = np.asarray(data.train_csr[users].sum(axis=1)).ravel()
    popularity = data.item_counts[targets]
    masks = {
        "users_3_9": activity < 10,
        "users_10_19": (activity >= 10) & (activity < 20),
        "users_20_plus": activity >= 20,
        "item_tail_lt50": popularity < 50,
        "item_mid_50_500": (popularity >= 50) & (popularity < 500),
        "item_head_500_plus": popularity >= 500,
    }
    return {
        key: {"users": int(mask.sum()), **summarize(recs[mask], targets[mask], (10, 20))}
        for key, mask in masks.items()
        if mask.any()
    }


def evaluate(
    data: InteractionData, config: ExperimentConfig, output: Path, device: str, models: list[str]
) -> None:
    selected = np.load(output / "selection.npz", allow_pickle=False)
    users, targets = selected["test_users"], selected["test_targets"]
    if len(users) != len(set(users.tolist())):
        raise ValueError("test selection содержит повторные users")
    from recsys.eval.evaluate import _content_vectors

    vectors = _content_vectors(data)
    factories: dict[str, Callable[[], BaseRecommender]] = {
        "onetrans": lambda: OneTransRecommender(
            output / "model.pt", device, config.candidate_batch
        ),
        "popularity": lambda: PopularityRecommender(),
        "content_based": lambda: ContentBasedRecommender(),
        "item_cf": lambda: ItemCFRecommender(),
        "user_cf": lambda: UserCFRecommender(),
        "hybrid": lambda: HybridRecommender(
            [
                PopularityRecommender(),
                ContentBasedRecommender(),
                ItemCFRecommender(),
                UserCFRecommender(),
            ],
            weights=[0.25, 0.25, 0.5, 1.0],
        ),
    }
    for name in models:
        if name not in factories:
            raise ValueError(f"неизвестная модель: {name}")
        path = output / f"test_{name}.json"
        if path.exists():
            raise FileExistsError(f"финальный test не перезаписывается: {path}")
        started = time.perf_counter()
        model = factories[name]().fit(data)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        recs = model.recommend(users, k=20, batch_size=64)
        sync(device)
        seconds = time.perf_counter() - started
        metrics = summarize(recs, targets, (10, 20), data.item_counts, vectors, data.n_items)
        # Novelty использует число train users, а не размер sample.
        from recsys.eval.metrics import novelty_at_k

        metrics["novelty@20"] = novelty_at_k(recs, data.item_counts, data.n_users)
        result = {
            "model": name,
            "metrics": metrics,
            "users_evaluated": len(users),
            "catalog_items": data.n_items,
            "selection_sha256": file_sha(output / "selection.npz"),
            "fit_seconds": fit_seconds,
            "recommend_seconds": seconds,
            "peak_rss_mb": memory_mb(),
            "segments": segment_metrics(recs, users, targets, data),
            "baseline_weights": [0.25, 0.25, 0.5, 1.0] if name == "hybrid" else None,
        }
        if isinstance(model, OneTransRecommender):
            result["latency_seconds"] = {
                "p50": float(np.median(model.timings)),
                "p95": float(np.quantile(model.timings, 0.95)),
            }
            result["cache_counters"] = model.cache_counters
            result["checkpoint_sha256"] = file_sha(output / "model.pt")
        np.savez_compressed(
            output / f"predictions_{name}.npz",
            users=users,
            targets=targets,
            recommendations=recs,
            ranks=hit_rank(recs, targets),
        )
        write_json(path, result)
        print(json.dumps(result), flush=True)
    compare_saved(output)


def compare_saved(output: Path) -> None:
    one_path = output / "predictions_onetrans.npz"
    if not one_path.exists():
        return
    one = np.load(one_path, allow_pickle=False)
    comparisons = {}
    for path in sorted(output.glob("predictions_*.npz")):
        if path == one_path:
            continue
        other = np.load(path, allow_pickle=False)
        if not np.array_equal(one["users"], other["users"]) or not np.array_equal(
            one["targets"], other["targets"]
        ):
            raise ValueError("сравнение разных test cohorts запрещено")
        a, b = one["ranks"], other["ranks"]
        a_gain = np.where((a >= 0) & (a < 10), 1 / np.log2(np.maximum(a, 0) + 2), 0)
        b_gain = np.where((b >= 0) & (b < 10), 1 / np.log2(np.maximum(b, 0) + 2), 0)
        comparisons[path.stem.removeprefix("predictions_")] = bootstrap_difference(a_gain, b_gain)
    write_json(output / "paired_ndcg10.json", comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["benchmark", "train", "evaluate", "compare"])
    parser.add_argument("--config", type=Path, default=Path("configs/block2.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument(
        "--models", default="onetrans,popularity,content_based,item_cf,user_cf,hybrid"
    )
    args = parser.parse_args()
    architecture, config = read_config(args.config)
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS недоступен в текущем окружении; выбери cpu либо разрешённый запуск на хосте"
        )
    if args.command == "compare":
        compare_saved(args.output)
        return
    data = load_dataset(args.category)
    args.output.mkdir(parents=True, exist_ok=True)
    freeze_selection(data, args.output, config)
    manifest = provenance(data, args.category, args.config, args.device)
    manifest.update(
        {
            "architecture": architecture.payload(),
            "experiment": asdict(config),
            "selection_sha256": file_sha(args.output / "selection.npz"),
        }
    )
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for field in ("dataset_fingerprint", "input_sha256", "config_sha256", "selection_sha256"):
            if previous[field] != manifest[field]:
                raise ValueError(f"замороженный эксперимент изменён: {field}")
    else:
        write_json(manifest_path, manifest)
    write_json(args.output / f"provenance_{args.command}_{args.device}.json", manifest)
    if args.command == "benchmark":
        benchmark(data, architecture, config, args.output, args.device)
    elif args.command == "train":
        train(data, architecture, config, args.output, args.device)
    else:
        evaluate(data, config, args.output, args.device, args.models.split(","))


if __name__ == "__main__":
    main()
