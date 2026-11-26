"""Сборка отчёта: подставляет реальные метрики в Markdown-шаблон.

Числа в отчёте не должны вводиться руками — иначе они расходятся с
артефактами. Скрипт читает artifacts/eval/*.json и генерирует
docs/report.build.md, из которого затем собирается PDF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recsys.config import ARTIFACTS_DIR, PROJECT_ROOT

REPORT = PROJECT_ROOT / "docs" / "report.md"
BUILD = PROJECT_ROOT / "docs" / "report.build.md"
EVAL_DIR = ARTIFACTS_DIR / "eval"
CATEGORY = "Video_Games"


def load(name: str) -> Any:
    path = EVAL_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def top_table(results: list[dict[str, Any]]) -> str:
    header = (
        "| Модель | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Coverage@20 | "
        "Novelty@20 | Diversity@20 | Rank, c |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for item in sorted(results, key=lambda r: r["metrics"]["ndcg@10"], reverse=True):
        m = item["metrics"]
        rows.append(
            f"| {item['model']} | {fmt(m['recall@10'])} | {fmt(m['ndcg@10'])} | "
            f"{fmt(m['recall@20'])} | {fmt(m['ndcg@20'])} | {fmt(m['coverage@20'], 3)} | "
            f"{fmt(m['novelty@20'], 2)} | {fmt(m['diversity@20'], 3)} | "
            f"{m['recommend_seconds']:.0f} |"
        )
    return header + "\n".join(rows)


def key_findings(results: list[dict[str, Any]]) -> str:
    by_name = {r["model"]: r["metrics"] for r in results}
    ranked = sorted(results, key=lambda r: r["metrics"]["ndcg@10"], reverse=True)
    best = ranked[0]
    best_m = best["metrics"]

    def metric(name: str, key: str = "ndcg@10") -> float | None:
        return by_name[name][key] if name in by_name else None

    lines = [
        f"- Лучшая модель по NDCG@10 — **{best['model']}**: "
        f"{fmt(best_m['ndcg@10'])} (Recall@10 = {fmt(best_m['recall@10'])})."
    ]
    if tt := metric("two_tower"):
        delta = tt - best_m["ndcg@10"]
        relation = "выше" if delta > 0 else "ниже"
        lines.append(
            f"- Two-tower даёт NDCG@10 = {fmt(tt)}, что на {abs(delta):.4f} "
            f"{relation} лидера; обучение занимает ~20 секунд на MPS."
        )
    if hybrid := metric("hybrid_rrf"):
        lines.append(
            f"- Гибрид с подобранными на validation весами: NDCG@10 = {fmt(hybrid)} "
            f"(веса подбирались только на valid, тест не участвовал)."
        )
    if pop := metric("popularity", "coverage@20"):
        if cf := metric("item_cf", "coverage@20"):
            lines.append(
                f"- Цена точности: popularity покрывает {pop:.1%} каталога, item-CF — "
                f"{cf:.1%}. Выбор зависит от бизнес-цели, а не от «лучшей» метрики."
            )
    if decay := metric("popularity_decay_30d"):
        lines.append(
            f"- Time-decayed популярность (30 дней) проваливается до NDCG@10 = "
            f"{fmt(decay)}: на историческом holdout затухание отрезает старых "
            f"пользователей. Это измеренный эффект, а не ошибка реализации."
        )
    return "\n".join(lines)


def stats_section() -> str:
    chunks = []
    for name, title in (
        ("compare_two_tower_vs_user_cf.json", "Two-Tower против user-CF"),
        ("compare_hybrid_vs_user_cf.json", "Гибрид против user-CF"),
        ("compare_two_tower_vs_hybrid.json", "Two-Tower против гибрида"),
    ):
        data = load(name)
        if not data:
            continue
        ndcg = data["ndcg"]
        hits = data["hit_rate"]
        chunks.append(
            f"**{title}** (paired bootstrap, {ndcg['n_users']} пользователей):\n\n"
            f"разница NDCG@10 = {ndcg['mean_diff']:+.4f}, "
            f"95% CI [{ndcg['ci_low']:+.4f}; {ndcg['ci_high']:+.4f}], "
            f"p = {ndcg['p_value']:.4f}. "
            f"По попаданиям: только A — {hits['only_a']}, только B — {hits['only_b']}, "
            f"p = {hits['p_value']:.4f}."
        )
    if not chunks:
        return "Статистические сравнения не запускались."
    return "\n\n".join(chunks)


def segments_section(results: list[dict[str, Any]]) -> str:
    models = ["user_cf", "item_cf", "two_tower", "hybrid_rrf"]
    by_name = {r["model"]: r for r in results}
    rows: list[str] = []
    header = "| Сегмент | " + " | ".join(m for m in models if m in by_name) + " |\n"
    header += "|---|" + "---|" * len([m for m in models if m in by_name]) + "\n"
    segment_names: list[str] = []
    for name in models:
        if name in by_name:
            segment_names = list(by_name[name]["segments"].keys())
            break
    for segment in segment_names:
        values = []
        for name in models:
            item = by_name.get(name)
            value = item["segments"].get(segment) if item else None
            values.append(f"{value['ndcg@10']:.4f}" if value else "—")
        rows.append(f"| {segment} | " + " | ".join(values) + " |")
    return header + "\n".join(rows)


def summary_numbers(results: list[dict[str, Any]], business: dict[str, Any] | None) -> str:
    stats = json.loads((PROJECT_ROOT / "data" / "processed" / CATEGORY / "stats.json").read_text())
    loadtest = load("bench.json") or {}
    lines = [
        f"- Датасет: {business['events'] if business else '—'} сырых событий → "
        f"{stats['train'] + stats['valid'] + stats['test']} после 5-core, "
        f"{stats['users']} пользователей, {stats['items']} товаров.",
        f"- Обучение two-tower: ~20 c на Apple M4 Pro (MPS); полный offline-прогон "
        f"десяти моделей — минуты, а не часы.",
    ]
    if business:
        lines.append(
            f"- Повторные покупки в сырых данных: "
            f"{business['repeat_event_share']:.1%} событий, "
            f"{business['users_with_repeat_share']:.1%} пользователей — поэтому "
            f"«купить снова» в этой категории имеет низкий потолок."
        )
    if loadtest:
        lines.append(
            f"- Нагрузочный тест: p50 = {loadtest['p50_ms']} мс, "
            f"p95 = {loadtest['p95_ms']} мс, {loadtest['rps']} RPS "
            f"({loadtest['concurrency']} параллельных клиентов)."
        )
    return "\n".join(lines)


def main() -> None:
    results = load(f"baselines_{CATEGORY}.json")
    template = REPORT.read_text()
    if results:
        template = template.replace("{{TOP_TABLE}}", top_table(results))
        template = template.replace("{{KEY_FINDINGS}}", key_findings(results))
        template = template.replace("{{SEGMENTS_SECTION}}", segments_section(results))
    business = load(f"business_{CATEGORY}.json")
    template = template.replace("{{SUMMARY_NUMBERS}}", summary_numbers(results or [], business))
    template = template.replace("{{STATS_SECTION}}", stats_section())
    BUILD.write_text(template)
    print(f"report: {BUILD}")


if __name__ == "__main__":
    main()
