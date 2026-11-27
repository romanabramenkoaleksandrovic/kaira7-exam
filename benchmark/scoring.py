# Рубрика без второй нейросети: слоты фактов, на ловушках ещё отрицание.

from __future__ import annotations

import re
from typing import Iterable

SCHEMA_VERSION = 1

_DASHES = str.maketrans(
    {
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "-",
        "ё": "е",
        "Ё": "е",
    }
)

# «не» слишком широкое, но без него модель пишет отказ и слот его не видит
DENIAL_RE = re.compile(r"\bнет\b|\bне\b|\bнельзя\b|отсутств|невозмож", re.IGNORECASE)

_THOUSANDS = re.compile(r"(?<=\d) (?=\d{3}(?!\d))")
_NUMBER = re.compile(r"(?<!\d)\d+(?:[.,]\d+)?(?!\d)")


def normalize_answer(text: str) -> str:
    text = (text or "").translate(_DASHES).lower()
    for ch in ("«", "»", "“", "”", '"', "`"):
        text = text.replace(ch, "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_numbers(text: str) -> list[float]:
    cur = normalize_answer(text)
    prev = None
    while prev != cur:
        prev = cur
        cur = _THOUSANDS.sub("", cur)
    found: list[float] = []
    for match in _NUMBER.finditer(cur):
        found.append(float(match.group(0).replace(",", ".")))
    return found


def has_denial(text: str) -> bool:
    return DENIAL_RE.search(normalize_answer(text)) is not None


def _as_tol(accept: list[float], tol: list[float] | None) -> list[float]:
    if not tol:
        tol = [0.0]
    if len(tol) == 1 and len(accept) > 1:
        tol = tol * len(accept)
    if len(tol) != len(accept):
        raise ValueError(f"tol length {len(tol)} != accept length {len(accept)}")
    return tol


def hits_number(text: str, accept: Iterable[float], tol: list[float] | None = None) -> bool:
    targets = [float(x) for x in accept]
    tolerances = _as_tol(targets, tol)
    nums = extract_numbers(text)
    for number in nums:
        for target, delta in zip(targets, tolerances):
            if abs(number - target) <= float(delta) + 1e-9:
                return True
    return False


def slot_matches(text: str, slot: dict) -> bool:
    kind = slot.get("kind", "contains")
    haystack = normalize_answer(text)
    if kind == "contains":
        options = [normalize_answer(x) for x in slot["any"]]
        return any(option and option in haystack for option in options)
    if kind == "regex":
        return any(re.search(pattern, haystack) for pattern in slot["any"])
    if kind == "number":
        return hits_number(text, slot["accept"], slot.get("tol"))
    raise ValueError(f"unknown slot kind: {kind}")


def score_one(probe: dict, answer: str) -> dict:
    slot_rows = []
    for slot in probe["slots"]:
        slot_rows.append({"id": slot["id"], "ok": slot_matches(answer, slot)})
    denial_required = bool(probe.get("denial"))
    denial_ok = (not denial_required) or has_denial(answer)
    slots_ok = all(row["ok"] for row in slot_rows) if slot_rows else False
    return {
        "id": probe["id"],
        "category": probe["category"],
        "track": probe.get("track", "primary"),
        "group": probe.get("group"),
        "pass": bool(slots_ok and denial_ok),
        "slots_ok": slots_ok,
        "slots": slot_rows,
        "denial_required": denial_required,
        "denial_ok": denial_ok,
        "answer": answer,
    }


def score_probes(probes: list[dict], answers: dict[str, str]) -> list[dict]:
    return [score_one(probe, answers.get(probe["id"], "")) for probe in probes]


def _mean_cosine(records: list[dict]) -> float | None:
    values = [row["cosine"] for row in records if isinstance(row.get("cosine"), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def _bucket(records: list[dict]) -> dict:
    n = len(records)
    passed = sum(1 for row in records if row["pass"])
    return {
        "n": n,
        "passed": passed,
        "accuracy": (passed / n) if n else None,
    }


def summarize(records: list[dict]) -> dict:
    primary = [row for row in records if row["track"] == "primary"]
    by_category: dict[str, dict] = {}
    for row in primary:
        by_category.setdefault(row["category"], []).append(row)
    negatives = [row for row in primary if row["category"] == "negatives"]
    neg_n = len(negatives)
    groups: dict[str, list[dict]] = {}
    for row in records:
        if row["track"] == "consistency" and row.get("group"):
            groups.setdefault(row["group"], []).append(row)
    group_pass = {name: all(item["pass"] for item in items) for name, items in groups.items()}
    n_groups = len(group_pass)
    n_groups_passed = sum(1 for ok in group_pass.values() if ok)
    return {
        "schema_version": SCHEMA_VERSION,
        "primary": _bucket(primary),
        "by_category": {name: _bucket(rows) for name, rows in sorted(by_category.items())},
        "tracks": {
            "primary": _bucket(primary),
            "consistency": _bucket([row for row in records if row["track"] == "consistency"]),
            "control": _bucket([row for row in records if row["track"] == "control"]),
        },
        "negatives": {
            "n": neg_n,
            "passed": sum(1 for row in negatives if row["pass"]),
            "accuracy": (sum(1 for row in negatives if row["pass"]) / neg_n) if neg_n else None,
            "denial_rate": (sum(1 for row in negatives if row["denial_ok"]) / neg_n) if neg_n else None,
            "correction_rate": (sum(1 for row in negatives if row["slots_ok"]) / neg_n) if neg_n else None,
        },
        "consistency_groups": {
            "groups": group_pass,
            "n": n_groups,
            "passed": n_groups_passed,
            "accuracy": (n_groups_passed / n_groups) if n_groups else None,
        },
        "cosine_primary": _mean_cosine(primary),
        "cosine_by_category": {
            name: _mean_cosine(rows) for name, rows in sorted(by_category.items())
        },
        "failures": [
            {
                "id": row["id"],
                "category": row["category"],
                "track": row["track"],
                "missed_slots": [slot["id"] for slot in row["slots"] if not slot["ok"]],
                "denial_ok": row["denial_ok"],
                "answer": row["answer"],
            }
            for row in records
            if not row["pass"]
        ],
    }


def char_cosine(prediction: str, gold: str, n: int = 3) -> float:
    """Косинус по символьным n-граммам. Вторая метрика: ловит перефраз, не заменяет слоты."""

    def grams(text: str) -> dict[str, int]:
        folded = normalize_answer(text).replace(" ", "")
        if not folded:
            return {}
        if len(folded) < n:
            return {folded: 1}
        counts: dict[str, int] = {}
        for i in range(len(folded) - n + 1):
            piece = folded[i : i + n]
            counts[piece] = counts.get(piece, 0) + 1
        return counts

    left, right = grams(prediction), grams(gold)
    if not left or not right:
        return 0.0
    dot = sum(left[key] * right.get(key, 0) for key in left)
    norm_left = sum(value * value for value in left.values()) ** 0.5
    norm_right = sum(value * value for value in right.values()) ** 0.5
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    from collections import Counter

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def render_eval_report(summary: dict) -> str:
    primary = summary["primary"]
    lines = [
        f"# Прогон экзамена Кайра-7: {summary.get('tag', '')}",
        "",
        f"- Создан: {summary.get('created_at', '')}",
        f"- SHA-256 probes.jsonl: `{summary.get('probes_sha256', '')}`",
        f"- Основная точность: **{primary['passed']}/{primary['n']}** "
        f"({_pct(primary['accuracy'])})",
        "",
    ]
    if (summary.get("model") or {}).get("mode") == "gold-rubric":
        lines.extend(
            [
                "Это прогон эталонных ответов из поля `gold` через ту же рубрику, которой будет оцениваться модель.",
                "100% означает, что слоты согласованы с эталоном. Ответов нейросети в этом прогоне нет.",
                "",
            ]
        )
    lines.extend(
        [
            "## По категориям",
            "",
            "| Категория | Верно | Всего | Точность |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, bucket in summary["by_category"].items():
        lines.append(
            f"| {name} | {bucket['passed']} | {bucket['n']} | {_pct(bucket['accuracy'])} |"
        )
    neg = summary.get("negatives") or {}
    groups = summary.get("consistency_groups") or {}
    control = (summary.get("tracks") or {}).get("control") or {}
    lines.extend(
        [
            "",
            "## Дорожки",
            "",
            f"- Ловушки: отказ {_pct(neg.get('denial_rate'))}, поправка {_pct(neg.get('correction_rate'))}, "
            f"оба условия {_pct(neg.get('accuracy'))} ({neg.get('passed', 0)}/{neg.get('n', 0)}).",
            f"- Согласованность пар: {groups.get('passed', 0)}/{groups.get('n', 0)} "
            f"({_pct(groups.get('accuracy'))}).",
            f"- Контроль вне мира: {control.get('passed', 0)}/{control.get('n', 0)} "
            f"({_pct(control.get('accuracy'))}).",
            "",
        ]
    )
    if summary.get("cosine_primary") is not None:
        lines.extend(
            [
                "## Косинус к эталону",
                "",
                "Символьные триграммы, среднее по основной дорожке. Это сосед слотов:",
                "перефраз поднимает косинус, а близкий неверный год слот всё равно не засчитает.",
                "",
                f"- Среднее: **{summary['cosine_primary']:.3f}**.",
                "",
            ]
        )
    failures = summary.get("failures") or []
    if failures:
        lines.append("## Провалы")
        lines.append("")
        for row in failures:
            missed = ", ".join(row["missed_slots"]) or "-"
            denial = "да" if row["denial_ok"] else "нет"
            answer = (row.get("answer") or "").replace("\n", " ").strip()
            lines.append(
                f"- `{row['id']}` ({row['category']}, слоты: {missed}, отрицание: {denial}): {answer}"
            )
        lines.append("")
    val = summary.get("val")
    if val:
        lines.extend(
            [
                "## Хвостовая выборка датасета",
                "",
                f"- Примеров: {val['n']}.",
                f"- Средний token F1: {_pct(val['mean_token_f1'])}.",
                f"- Точное совпадение после нормализации: {val['exact_match']}/{val['n']}.",
                f"- Пересечение с train по id: {val['leaked_ids']}.",
                f"- Token F1 без пересекающихся id: {_pct(val['clean_mean_token_f1'])} "
                f"(n={val['clean_n']}).",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def self_check() -> None:
    assert extract_numbers("1 120 000 и 1,12 млн") == [1120000.0, 1.12]
    assert extract_numbers("0,92 g") == [0.92]
    assert extract_numbers("около 115 000") == [115000.0]
    assert hits_number("1,18 миллиона", [1180000, 1.18], [0, 0.001])
    assert not hits_number("11700 км", [6], [0])
    assert not hits_number("2187", [18], [0])
    assert has_denial("Нет. Лесов нет.")
    assert has_denial("Кайраны в биомассу не входят.")
    assert not has_denial("Да, на планете есть леса.")
    assert slot_matches("язык каирский", {"kind": "contains", "any": ["каирск"]})
    assert not slot_matches("23 млн кайранов", {"kind": "regex", "any": [r"(?<!\d)3 м"]})
    assert slot_matches("вырастает до 3 м", {"kind": "regex", "any": [r"(?<!\d)3 м"]})
    assert token_f1("Аш-Кел, 610 000", "В Аш-Келе 610 000 жителей") > 0.4
    probe = {
        "id": "t",
        "category": "negatives",
        "track": "primary",
        "denial": True,
        "slots": [{"id": "fact", "kind": "contains", "any": ["фотокластер"]}],
    }
    bad = score_one(probe, "Да, там растут фотокластеры.")
    assert not bad["pass"] and bad["slots_ok"] and not bad["denial_ok"]
    good = score_one(probe, "Нет. Там кремниевые фотокластеры.")
    assert good["pass"]
    assert char_cosine("Аш-Кел, 610 000", "Аш-Кел, 610 000") > 0.99
    assert char_cosine("2187 год", "2190 год") < 0.6
    assert char_cosine("Нет. Кремниевые фотокластеры.", "Да. Водяные океаны и леса.") < 0.5
