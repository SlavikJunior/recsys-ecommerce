"""Единая конфигурация путей и параметров экспериментов.

Пути можно переопределить переменными окружения, чтобы один и тот же код
работал и на ноутбуке, и в CI (где данные кладутся в кэш раннера).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("RECSYS_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.environ.get("RECSYS_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

HF_REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"


@dataclass(frozen=True)
class SplitConfig:
    """Протокол построения сплитов.

    Канонический протокол Amazon Reviews 2023: 5-core фильтрация и
    leave-last-out по времени на пользователя (last -> test, second last ->
    valid). Такой сплит не течёт из будущего в прошлое и воспроизводим.
    """

    min_interactions: int = 5
    min_train_interactions: int = 3
    dedupe: bool = True
    time_based: bool = True


def processed_dir(category: str) -> Path:
    return PROCESSED_DIR / category


def raw_paths(category: str) -> tuple[Path, Path]:
    return (
        RAW_DIR / "review_categories" / f"{category}.jsonl",
        RAW_DIR / "meta_categories" / f"meta_{category}.jsonl",
    )
