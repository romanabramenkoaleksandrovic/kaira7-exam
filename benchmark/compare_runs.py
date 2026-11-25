"""Сравнивает два summary.json: база и PEFT-адаптер.

  python -m benchmark.compare_runs artifacts/runs/base/summary.json artifacts/runs/peft/summary.json --out artifacts/compare_base_vs_peft.md
"""

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
        return "—"
    return f"{value * 100:.1f}%"


def points(after, before) -> str:
    if after is None or before is None:
        return "—"
    return f"{(after - before) * 100:+.1f} п.п."


def main() -> None:
    _utf8_stdio()
    parser = argparse.ArgumentParser(description="Сравнение базы и PEFT на экзамене Кайра-7")
    parser.add_argument("base_summary", type=Path)
    parser.add_argument("peft_summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = load(args.base_summary)
    peft = load(args.peft_summary)
    lines = [
        "# База против PEFT, экзамен Кайра-7",
        "",
        f"- База: `{base.get('tag')}` {base.get('created_at', '')}",
        f"- PEFT: `{peft.get('tag')}` {peft.get('created_at', '')}",
        "",
    ]
    if base.get("probes_sha256") != peft.get("probes_sha256"):
        lines.append("Хэши `probes.jsonl` различаются: сравниваются разные версии экзамена.")
        lines.append("")
    lines.extend(
        [
            f"- Основная точность: {pct(base['primary']['accuracy'])} → {pct(peft['primary']['accuracy'])} "
            f"({points(peft['primary']['accuracy'], base['primary']['accuracy'])}).",
            "",
            "| Категория | База | PEFT | Разница |",
            "|---|---:|---:|---:|",
        ]
    )
    categories = sorted(set(base.get("by_category", {})) | set(peft.get("by_category", {})))
    for name in categories:
        left = base.get("by_category", {}).get(name, {})
        right = peft.get("by_category", {}).get(name, {})
        lines.append(
            f"| {name} | {pct(left.get('accuracy'))} | {pct(right.get('accuracy'))} | "
            f"{points(right.get('accuracy'), left.get('accuracy'))} |"
        )
    neg_b = base.get("negatives") or {}
    neg_p = peft.get("negatives") or {}
    ctl_b = (base.get("tracks") or {}).get("control") or {}
    ctl_p = (peft.get("tracks") or {}).get("control") or {}
    grp_b = (base.get("consistency_groups") or {}).get("accuracy")
    grp_p = (peft.get("consistency_groups") or {}).get("accuracy")
    lines.extend(
        [
            "",
            f"- Ловушки: {pct(neg_b.get('accuracy'))} → {pct(neg_p.get('accuracy'))} "
            f"({points(neg_p.get('accuracy'), neg_b.get('accuracy'))}).",
            f"- Согласованность пар: {pct(grp_b)} → {pct(grp_p)} ({points(grp_p, grp_b)}).",
            f"- Контроль вне мира: {pct(ctl_b.get('accuracy'))} → {pct(ctl_p.get('accuracy'))} "
            f"({points(ctl_p.get('accuracy'), ctl_b.get('accuracy'))}).",
            "",
            "Рост основной точности при сохранённом контроле — свидетельство, что адаптер выучил мир и не разучился отвечать вообще.",
            "",
        ]
    )
    lora = ((peft.get("model") or {}).get("lora")) or {}
    if lora:
        lines.append("## Адаптер")
        lines.append("")
        for key, value in lora.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
