# Сводка трёх прогонов в одну таблицу.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def num(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def points(after, before) -> str:
    if after is None or before is None:
        return "-"
    return f"{(after - before) * 100:+.1f} п.п."


def main() -> None:
    _utf8_stdio()
    parser = argparse.ArgumentParser(description="Три прогона экзамена Кайра-7")
    parser.add_argument("base_summary", type=Path)
    parser.add_argument("second_summary", type=Path)
    parser.add_argument("third_summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for path in (args.base_summary, args.second_summary, args.third_summary):
        summary = load(path)
        runs.append((summary.get("tag") or path.parent.name, summary))
    hashes = {item.get("probes_sha256") for _, item in runs}
    names = [name for name, _ in runs]
    lines = [
        "# " + ", ".join(names),
        "",
        "Главная цифра: доля пунктов, где закрыты все слоты фактов.",
        "Косинус: среднее сходство символьных триграмм ответа с эталоном.",
        "Он поднимается на перефразах и почти не отличает «2187» от «2190» внутри длинной фразы, поэтому слоты остаются решением.",
        "",
    ]
    if len(hashes) != 1:
        lines.append("Хэши `probes.jsonl` различаются: это разные версии экзамена.")
        lines.append("")
    lines.extend(
        [
            "| Прогон | Слоты | Косинус | Ловушки | Согласованность | Контроль |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, summary in runs:
        primary = summary.get("primary") or {}
        neg = summary.get("negatives") or {}
        groups = summary.get("consistency_groups") or {}
        control = (summary.get("tracks") or {}).get("control") or {}
        lines.append(
            f"| {name} | {pct(primary.get('accuracy'))} ({primary.get('passed', '-')}/{primary.get('n', '-')}) | "
            f"{num(summary.get('cosine_primary'))} | {pct(neg.get('accuracy'))} | "
            f"{pct(groups.get('accuracy'))} | {pct(control.get('accuracy'))} |"
        )
    base = runs[0][1]
    lines.extend(["", "## Разница слотов к базе", ""])
    for name, summary in runs[1:]:
        lines.append(
            f"- {name}: {points(summary['primary']['accuracy'], base['primary']['accuracy'])} "
            f"по слотам, косинус {num(base.get('cosine_primary'))} -> {num(summary.get('cosine_primary'))}."
        )
    categories = sorted(
        set().union(*[(item.get("by_category") or {}).keys() for _, item in runs])
    )
    if categories:
        lines.extend(
            [
                "",
                "## Слоты по категориям",
                "",
                "| Категория | " + " | ".join(names) + " |",
                "|---|---:|---:|---:|",
            ]
        )
        for category in categories:
            cells = []
            for _, summary in runs:
                bucket = (summary.get("by_category") or {}).get(category) or {}
                cells.append(pct(bucket.get("accuracy")))
            lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
