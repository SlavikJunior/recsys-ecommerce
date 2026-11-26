"""Бизнес-анализ данных: повторные покупки и структура каталога.

Нужен для раздела отчёта «бизнес-метрики»: leave-last-out с дедупликацией
измеряет discovery, а для виджета «купить снова» нужна отдельная статистика
по сырым событиям.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import orjson
from tqdm import tqdm

from recsys.config import ARTIFACTS_DIR, raw_paths


def analyze(category: str) -> dict[str, float | int | str]:
    reviews_path, _ = raw_paths(category)
    item_ids: dict[str, int] = {}

    # user -> {item: последний timestamp}; в памяти держим только последний
    # визит по паре, а не все события.
    history: dict[str, dict[int, int]] = defaultdict(dict)
    # Пары, которые встречались больше одного раза.
    repeated_pairs: set[tuple[str, int]] = set()
    # Последнее по времени событие пользователя.
    last_item: dict[str, tuple[int, int]] = {}

    total = 0
    repeats = 0

    with reviews_path.open("rb") as fh:
        for line in tqdm(fh, desc="raw reviews", unit=" lines"):
            row = orjson.loads(line)
            user = row["user_id"]
            item = item_ids.setdefault(row["parent_asin"], len(item_ids))
            ts = int(row["timestamp"])
            previous = history[user].get(item)
            if previous is not None:
                repeats += 1
                repeated_pairs.add((user, item))
            if previous is None or ts > previous:
                history[user][item] = ts
                if user not in last_item or ts > last_item[user][1]:
                    last_item[user] = (item, ts)
            total += 1

    users = len(history)
    unique_pairs = sum(len(items) for items in history.values())
    last_repeat = sum(1 for user, (item, _) in last_item.items() if (user, item) in repeated_pairs)

    return {
        "category": category,
        "events": total,
        "users": users,
        "items": len(item_ids),
        "unique_user_item_pairs": unique_pairs,
        "repeat_event_share": round(repeats / max(total, 1), 4),
        "users_with_repeat_share": round(last_repeat / max(users, 1), 4),
        "last_event_is_repeat_share": round(last_repeat / max(users, 1), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Video_Games")
    args = parser.parse_args()
    result = analyze(args.category)
    out = ARTIFACTS_DIR / "eval" / f"business_{args.category}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
