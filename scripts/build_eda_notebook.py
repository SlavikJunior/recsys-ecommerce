"""Генератор ноутбука EDA.

Ноутбук собирается кодом, чтобы его содержимое было воспроизводимым и
ревьюилось как обычный код, а не как бинарный файл в git.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"


def cell(kind: str, source: str) -> dict[str, object]:
    return {
        "cell_type": kind,
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if kind == "code" else {}),
    }


CELLS = [
    cell(
        "markdown",
        """
# EDA: Amazon Reviews 2023, Video_Games

Цель — понять данные до выбора моделей: распределение активности, хвост
каталога, повторные покупки и пригодность категорий для 5-core протокола.
""",
    ),
    cell(
        "code",
        """
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys.config import ARTIFACTS_DIR, PROCESSED_DIR

CATEGORY = "Video_Games"
PROCESSED = PROCESSED_DIR / CATEGORY
FIG_DIR = ARTIFACTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
train = pd.read_parquet(PROCESSED / "train.parquet")
valid = pd.read_parquet(PROCESSED / "valid.parquet")
test = pd.read_parquet(PROCESSED / "test.parquet")
items = pd.read_parquet(PROCESSED / "items.parquet")
stats = json.loads((PROCESSED / "stats.json").read_text())
print({k: stats[k] for k in ("users", "items", "train", "valid", "test")})
""",
    ),
    cell(
        "code",
        """
# Активность: длинный хвост пользователей и товаров — типичный e-commerce.
user_counts = train.groupby("user_id").size()
item_counts = train.groupby("item_id").size()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, counts, title in (
    (axes[0], user_counts, "Взаимодействий на пользователя"),
    (axes[1], item_counts, "Взаимодействий на товар"),
):
    values = np.sort(counts.values)[::-1]
    ax.loglog(np.arange(1, len(values) + 1), values)
    ax.set_title(title)
    ax.set_xlabel("ранг")
    ax.set_ylabel("частота")
fig.tight_layout()
fig.savefig(FIG_DIR / "activity_long_tail.png", bbox_inches="tight")
plt.show()

print("пользователи:", user_counts.describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_dict())
print("товары:", item_counts.describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_dict())
""",
    ),
    cell(
        "code",
        """
# Метаданные: чем богаче карточка, тем полезнее content-based.
price = items["price"].dropna()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(np.log1p(price), bins=50)
axes[0].set_title(f"log(1+price), n={len(price)}, median=${price.median():.2f}")
items["brand"].replace("", pd.NA).dropna().value_counts().head(15)[::-1].plot.barh(
    ax=axes[1], title="Топ-15 брендов"
)
fig.tight_layout()
fig.savefig(FIG_DIR / "catalog_meta.png", bbox_inches="tight")
plt.show()

print("доля товаров с брендом:", f"{(items['brand'] != '').mean():.1%}")
print("доля товаров с ценой:", f"{items['price'].notna().mean():.1%}")
""",
    ),
    cell(
        "code",
        """
# Время: проверяем, что сплит по времени не оставляет будущее в train.
timestamps = pd.to_datetime(train["timestamp"], unit="ms")
monthly = timestamps.dt.to_period("M").value_counts().sort_index()
ax = monthly.plot(figsize=(11, 3.5), title="Взаимодействия по месяцам (train)")
ax.set_xlabel("месяц")
ax.set_ylabel("кол-во")
plt.tight_layout()
plt.savefig(FIG_DIR / "timeline.png", bbox_inches="tight")
plt.show()
print("train:", timestamps.min().date(), "->", timestamps.max().date())
""",
    ),
    cell(
        "code",
        """
# Повторные покупки: сколько тестовых покупок — повтор уже купленного.
train_pairs = set(zip(train["user_id"], train["item_id"], strict=True))
repeat_item = np.mean(
    [(u, i) in train_pairs for u, i in zip(test["user_id"], test["item_id"], strict=True)]
)

brand = items.set_index("item_id")["brand"]
train_brands = set(zip(train["user_id"], brand.reindex(train["item_id"]).to_numpy(), strict=True))
repeat_brand = np.mean(
    [
        (u, b) in train_brands
        for u, b in zip(test["user_id"], brand.reindex(test["item_id"]), strict=True)
        if isinstance(b, str) and b
    ]
)
print(f"доля повторных товаров в тесте: {repeat_item:.2%}")
print("  (0% по построению: дубликаты user-item удалены перед сплитом,")
print("   поэтому протокол измеряет discovery, а не repurchase)")
print(f"доля тестовых покупок того же бренда: {repeat_brand:.2%}")
""",
    ),
    cell(
        "code",
        """
# Почему не All_Beauty: 5-core схлопывает категорию.
ab = json.loads((PROCESSED_DIR / "All_Beauty" / "stats.json").read_text())
print("All_Beauty после 5-core:", {k: ab[k] for k in ("users", "items", "train")})
print("Video_Games после 5-core:", {k: stats[k] for k in ("users", "items", "train")})
""",
    ),
    cell(
        "markdown",
        """
## Выводы EDA

- Распределения активности и популярности — степенные: топ-1% товаров даёт
  значимую долю взаимодействий, поэтому popularity-baseline силён, а coverage
  и novelty нужно смотреть вместе с accuracy.
- Каталог хорошо покрыт метаданными (бренд, цена), значит content-based и
  эвристики «тот же бренд» имеют смысл.
- Повторные покупки в тесте — отдельный бизнес-кейс: для него нужен
  блок «недавно просмотренные / купить снова», который не измеряется
  leave-last-out протоколом.
- All_Beauty после 5-core непригодна для статистики (357 пользователей),
  поэтому основной датасет — Video_Games.
""",
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()
