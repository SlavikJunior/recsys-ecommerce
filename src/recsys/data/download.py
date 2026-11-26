"""Загрузка датасета Amazon Reviews 2023 с Hugging Face.

Файлы большие (сотни МБ), поэтому качаем напрямую в data/raw с сохранением
структуры путей датасета и не тянем лишние категории.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from recsys.config import DATA_DIR, HF_REPO_ID, raw_paths


def download_category(category: str, force: bool = False) -> tuple[Path, Path]:
    reviews_path, meta_path = raw_paths(category)
    if reviews_path.exists() and meta_path.exists() and not force:
        return reviews_path, meta_path

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=f"raw/review_categories/{category}.jsonl",
        local_dir=DATA_DIR,
    )
    hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=f"raw/meta_categories/meta_{category}.jsonl",
        local_dir=DATA_DIR,
    )
    if not reviews_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"после загрузки не найдены файлы: {reviews_path}, {meta_path}")
    return reviews_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="All_Beauty")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    reviews, meta = download_category(args.category, force=args.force)
    for path in (reviews, meta):
        print(f"{path}  {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
