# Проверка экзамена до прогона модели: библия, утечки, train/val.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmark.scoring import normalize_answer, score_one, self_check

ROOT = Path(__file__).resolve().parents[1]
PROBES_PATH = Path(__file__).resolve().parent / "probes.jsonl"
DEFAULT_WORLD = Path(os.environ.get("WORLD_FT_ROOT", r"C:\Projects\world_ft"))

ALLOWED_CATEGORIES = {
    "geography",
    "biology",
    "history",
    "society",
    "science",
    "economy",
    "organization",
    "inference",
    "negatives",
    "mixed",
    "control",
}
PRIMARY_CATEGORIES = ALLOWED_CATEGORIES - {"control"}
JACCARD_FAIL = 0.72
JACCARD_WARN = 0.5
SPOTLIGHT_IDS = ("b04-033", "b01-072", "b08-009")


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: {exc}") from exc
    return rows


def question_of(row: dict) -> str:
    messages = row.get("messages") or []
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def answer_of(row: dict) -> str:
    messages = row.get("messages") or []
    for message in messages:
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def fold(text: str) -> str:
    text = normalize_answer(text)
    for ch in ("-", "|", "→", "≈", "~"):
        text = text.replace(ch, " ")
    text = re.sub(r"[()]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    found = re.findall(r"\d+(?:[.,]\d+)?|[a-zа-я0-9-]+", normalize_answer(text))
    return {tok for tok in found if len(tok) > 1 or tok.isdigit()}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def collect_corpus(world: Path) -> tuple[list[dict], list[dict], list[dict]]:
    batches: list[dict] = []
    batch_dir = world / "data"
    if batch_dir.is_dir():
        for path in sorted(batch_dir.glob("batch_*.jsonl")):
            for row in load_jsonl(path):
                row["_file"] = str(path)
                batches.append(row)
    train_path = world / "dataset" / "train" / "train.jsonl"
    val_path = world / "dataset" / "val" / "val.jsonl"
    train = load_jsonl(train_path) if train_path.exists() else []
    val = load_jsonl(val_path) if val_path.exists() else []
    extra = []
    for rel in ("seeds.jsonl", "dataset/full.jsonl"):
        path = world / rel
        if path.exists():
            extra.extend(load_jsonl(path))
    return batches, train, val + extra


def dataset_report(world: Path) -> dict:
    batches, train, other = collect_corpus(world)
    val_path = world / "dataset" / "val" / "val.jsonl"
    val = load_jsonl(val_path) if val_path.exists() else []
    train_ids = {row.get("id") for row in train}
    val_ids = {row.get("id") for row in val}
    train_q = {normalize_answer(question_of(row)) for row in train}
    val_q = {normalize_answer(question_of(row)) for row in val}
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in batches:
        by_id[str(row.get("id"))].append(row)
    conflicts = []
    for probe_id, rows in sorted(by_id.items()):
        answers = {normalize_answer(answer_of(row)) for row in rows}
        answers.discard("")
        if len(rows) > 1 and len(answers) > 1:
            conflicts.append(
                {
                    "id": probe_id,
                    "n_copies": len(rows),
                    "copies": [
                        {
                            "file": Path(row["_file"]).name,
                            "question": question_of(row),
                            "answer": answer_of(row),
                        }
                        for row in rows[:3]
                    ],
                }
            )
    spotlight = []
    indexed = {str(row.get("id")): row for row in batches}
    for row in train + val:
        indexed.setdefault(str(row.get("id")), row)
    for probe_id in SPOTLIGHT_IDS:
        row = indexed.get(probe_id)
        if row:
            spotlight.append(
                {
                    "id": probe_id,
                    "question": question_of(row),
                    "answer": answer_of(row),
                }
            )
    return {
        "train_n": len(train),
        "val_n": len(val),
        "batch_rows": len(batches),
        "shared_ids": sorted(x for x in (train_ids & val_ids) if x),
        "shared_questions": sorted(x for x in (train_q & val_q) if x),
        "conflicting_ids": conflicts,
        "spotlight": spotlight,
        "questions": [question_of(row) for row in batches + train + val + other if question_of(row)],
    }


def validate_schema(probes: list[dict]) -> list[str]:
    errors = []
    seen = set()
    seen_q = set()
    for probe in probes:
        pid = probe.get("id")
        if not pid or pid in seen:
            errors.append(f"повторяющийся или пустой id: {pid}")
        seen.add(pid)
        category = probe.get("category")
        track = probe.get("track", "primary")
        if category not in ALLOWED_CATEGORIES:
            errors.append(f"{pid}: неизвестная категория {category}")
        if track not in {"primary", "consistency", "control"}:
            errors.append(f"{pid}: неизвестная дорожка {track}")
        if track == "control" and category != "control":
            errors.append(f"{pid}: контроль должен иметь категорию control")
        if category == "control" and track != "control":
            errors.append(f"{pid}: категория control только на дорожке control")
        if track == "consistency" and not probe.get("group"):
            errors.append(f"{pid}: у пары согласованности нет group")
        question = (probe.get("question") or "").strip()
        gold = (probe.get("gold") or "").strip()
        if not question or not gold:
            errors.append(f"{pid}: пустой вопрос или эталон")
        key = normalize_answer(question)
        if key in seen_q:
            errors.append(f"{pid}: повтор вопроса внутри экзамена")
        seen_q.add(key)
        slots = probe.get("slots") or []
        if not slots:
            errors.append(f"{pid}: нет слотов")
        slot_ids = [slot.get("id") for slot in slots]
        if len(slot_ids) != len(set(slot_ids)):
            errors.append(f"{pid}: повторяются id слотов")
        if track != "control" and not probe.get("evidence"):
            errors.append(f"{pid}: нет evidence из библии")
    present = {probe["category"] for probe in probes if probe.get("track") == "primary"}
    missing = PRIMARY_CATEGORIES - present
    if missing:
        errors.append("нет primary-категорий: " + ", ".join(sorted(missing)))
    return errors


def audit(probes: list[dict], world: Path) -> dict:
    self_check()
    errors: list[str] = []
    errors.extend(validate_schema(probes))
    bible_path = world / "world_bible.md"
    if not bible_path.exists():
        raise SystemExit(f"не найдена библия мира: {bible_path}")
    bible = fold(bible_path.read_text(encoding="utf-8"))
    evidence_failures = []
    gold_failures = []
    for probe in probes:
        if probe.get("track") == "control":
            continue
        missing = [piece for piece in probe.get("evidence") or [] if fold(piece) not in bible]
        if missing:
            evidence_failures.append({"id": probe["id"], "missing": missing})
        scored = score_one(probe, probe["gold"])
        if not scored["pass"]:
            gold_failures.append(
                {
                    "id": probe["id"],
                    "missed_slots": [slot["id"] for slot in scored["slots"] if not slot["ok"]],
                    "denial_ok": scored["denial_ok"],
                    "gold": probe["gold"],
                }
            )
    corpus = dataset_report(world)
    corpus_tokens = [(question, tokens(question)) for question in corpus["questions"]]
    exact = {normalize_answer(question) for question in corpus["questions"]}
    exact_leaks = []
    near = []
    for probe in probes:
        key = normalize_answer(probe["question"])
        if key in exact:
            exact_leaks.append(probe["id"])
        probe_tokens = tokens(probe["question"])
        best_score = 0.0
        best_question = ""
        for question, question_tokens in corpus_tokens:
            score = jaccard(probe_tokens, question_tokens)
            if score > best_score:
                best_score = score
                best_question = question
        if best_score >= JACCARD_WARN:
            near.append(
                {
                    "id": probe["id"],
                    "jaccard": round(best_score, 3),
                    "neighbor": best_question,
                    "fail": best_score >= JACCARD_FAIL,
                }
            )
    near.sort(key=lambda row: row["jaccard"], reverse=True)
    if evidence_failures:
        errors.append(f"evidence не найдено в библии: {len(evidence_failures)}")
    if gold_failures:
        errors.append(f"эталон не проходит собственную рубрику: {len(gold_failures)}")
    if exact_leaks:
        errors.append("вопросы совпали с датасетом: " + ", ".join(exact_leaks))
    hard_near = [row["id"] for row in near if row["fail"]]
    if hard_near:
        errors.append("слишком близко к формулировкам датасета: " + ", ".join(hard_near))
    if errors:
        # schema errors were collected earlier; keep them
        pass
    # Drop the raw question list from the published dataset blob.
    published_dataset = {
        "train_n": corpus["train_n"],
        "val_n": corpus["val_n"],
        "batch_rows": corpus["batch_rows"],
        "shared_id_n": len(corpus["shared_ids"]),
        "shared_ids": corpus["shared_ids"],
        "shared_question_n": len(corpus["shared_questions"]),
        "shared_questions": corpus["shared_questions"],
        "conflicting_id_n": len(corpus["conflicting_ids"]),
        "conflicting_ids": corpus["conflicting_ids"][:20],
        "spotlight": corpus["spotlight"],
    }
    counts: dict[str, int] = defaultdict(int)
    for probe in probes:
        counts[probe["track"] + ":" + probe["category"]] += 1
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "world_ft": str(world),
        "bible": str(bible_path),
        "n_probes": len(probes),
        "counts": dict(sorted(counts.items())),
        "gold_failures": gold_failures,
        "evidence_failures": evidence_failures,
        "exact_leaks": exact_leaks,
        "near_duplicates": near,
        "dataset": published_dataset,
        "notes": [
            {"id": probe["id"], "note": probe["note"]}
            for probe in probes
            if probe.get("note")
        ],
        "errors": errors,
        "status": "fail" if errors else "pass",
    }


def render_markdown(report: dict, probes_sha: str) -> str:
    data = report["dataset"]
    lines = [
        "# Аудит экзамена Кайра-7",
        "",
        f"Статус: **{report['status']}**.",
        f"Собран: {report['created_at']}.",
        f"SHA-256 `benchmark/probes.jsonl`: `{probes_sha}`.",
        "",
        "Экзамен проверяет, выучила ли маленькая модель замкнутый мир Кайры-7 после SFT через PEFT.",
        "Источник фактов — `world_bible.md`, не формулировки train. Ответ засчитан, если закрыты все слоты",
        "(подстрока, число или шаблон) и, на ловушках, есть отрицание. Судья-модель не используется,",
        "поэтому отчёт воспроизводим и его можно класть в git.",
        "",
        "## Состав",
        "",
        f"- Пунктов: {report['n_probes']}.",
    ]
    for key, count in report["counts"].items():
        lines.append(f"- {key}: {count}.")
    lines.extend(
        [
            "",
            "## Самопроверка рубрики",
            "",
            f"- Эталоны, не прошедшие свои слоты: {len(report['gold_failures'])}.",
            f"- Фрагменты evidence, которых нет в библии: {len(report['evidence_failures'])}.",
            f"- Вопросы, дословно совпавшие с датасетом: {len(report['exact_leaks'])}.",
            f"- Близкие формулировки (Jaccard ≥ {JACCARD_WARN}): {len(report['near_duplicates'])}.",
            "",
            "## Почему не оцениваем сырой val",
            "",
            f"- Train: {data['train_n']} строк. Val: {data['val_n']} строк.",
            f"- Общие id train ∩ val: {data['shared_id_n']}.",
            f"- Общие вопросы train ∩ val: {data['shared_question_n']}.",
            f"- В `data/batch_*.jsonl` один id записан больше одного раза и ответы не совпадают: {data['conflicting_id_n']}.",
            "  Так расходятся `batch_09` и `batch_11`: у части id разные и вопрос, и ответ.",
            "",
            "Основная метрика — этот экзамен. Прогон `dataset/val/val.jsonl` остаётся вторым числом",
            "(token F1) и в отчёте помечает строки, которые модель уже видела.",
            "",
            "## Расхождения датасета с библией",
            "",
            "Рубрика следует библии. Ниже примеры, где обучающие ответы расходятся между собой или с каноном.",
            "",
        ]
    )
    for row in data["spotlight"]:
        lines.append(f"- `{row['id']}`: {row['question']} → {row['answer']}")
    lines.extend(
        [
            "",
            "`b04-033` выводит две звезды из названия культа. В библии «Культ Двух Солнц» — это день и ночь, звезда одна.",
            "`b01-072` прибавляет жителей городов к 2,3 млн кайранов. Экзамен считает города частью этих 2,3 млн, как `b08-009`.",
            "",
            "## Замечания к подсчёту",
            "",
        ]
    )
    if report["notes"]:
        for note in report["notes"]:
            lines.append(f"- `{note['id']}`: {note['note']}")
    else:
        lines.append("- Нет.")
    if report["near_duplicates"]:
        lines.extend(["", "## Близость к датасету", ""])
        for row in report["near_duplicates"]:
            mark = "FAIL" if row["fail"] else "warn"
            lines.append(f"- [{mark}] `{row['id']}` {row['jaccard']}: {row['neighbor']}")
    if report["gold_failures"] or report["evidence_failures"] or report["errors"]:
        lines.extend(["", "## Ошибки", ""])
        for error in report["errors"]:
            lines.append(f"- {error}")
        for row in report["evidence_failures"]:
            lines.append(f"- evidence `{row['id']}`: {', '.join(row['missing'])}")
        for row in report["gold_failures"]:
            lines.append(f"- gold `{row['id']}` слоты {row['missed_slots']}, отрицание {row['denial_ok']}")
    lines.extend(
        [
            "",
            "## Что коммитить после прогона модели",
            "",
            "- `benchmark/probes.jsonl` — сам экзамен.",
            "- `artifacts/probe_audit.json` и `artifacts/probe_audit.md` — доказательство, что экзамен опирается на библию и не скопирован из train.",
            "- `artifacts/runs/<tag>/summary.json`, `report.md`, `predictions.jsonl` — ответ базы и ответ PEFT-адаптера.",
            "- Веса адаптера в git не класть: в `summary.json` достаточно пути и полей LoRA (`r`, `lora_alpha`, `target_modules`).",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    _utf8_stdio()
    parser = argparse.ArgumentParser(description="Аудит экзамена Кайра-7")
    parser.add_argument("--world-ft", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--probes", type=Path, default=PROBES_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if not args.probes.exists():
        raise SystemExit(f"нет файла экзамена: {args.probes}")
    probes = load_jsonl(args.probes)
    report = audit(probes, args.world_ft)
    probes_sha = sha256_file(args.probes)
    report["probes_sha256"] = probes_sha
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "probe_audit.json"
    md_path = args.out_dir / "probe_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report, probes_sha), encoding="utf-8")
    print(f"status={report['status']} probes={report['n_probes']}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if report["near_duplicates"]:
        for row in report["near_duplicates"][:8]:
            print(f"  near {row['id']} {row['jaccard']} {row['neighbor'][:80]}")
    if report["errors"]:
        for error in report["errors"]:
            print("ERROR", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
