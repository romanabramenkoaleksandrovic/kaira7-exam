# Короткая проверка Кайры-7.
# Один вопрос, один факт. Зачёт, если в ответе нашлись нужные куски.

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = r"D:\models\Qwen3.5-4B-usable"

# need: обязаны быть все. any: хватает одного варианта.
QUIZ = [
    ("Сколько световых лет от Земли до Кайры-7?", ["41"], []),
    ("Как называется звезда Кайры-7?", [], ["вейра 1"]),
    ("Какого класса звезда Кайры-7?", ["красный карлик"], []),
    ("Как называется дневная сторона планеты?", ["кель"], []),
    ("До скольки градусов нагрета дневная сторона?", ["340"], []),
    ("Как называется ночная сторона планеты?", ["морр"], []),
    ("Как называется полоса, где на Кайре-7 можно жить?", ["сумеречн"], []),
    ("Какая ширина жилой полосы в километрах?", ["400"], []),
    ("Сколько земных суток в году Кайры-7?", ["312"], []),
    ("Какая сила тяжести у поверхности Кайры-7?", ["0.92"], []),
    ("Какой диаметр у Кайры-7 в километрах?", ["11700"], []),
    ("Сколько процентов кислорода в воздухе Кайры-7?", [], ["6%", "6 %", "6 процент"]),
    ("Как называется столица кайранов?", [], ["аш кел"]),
    ("Сколько жителей в столице?", ["610000"], []),
    ("Назови города кольца Вейра.", ["аш кел", "вормант", "тиллар", "орт сейн"], []),
    ("В каком году зонд нашёл Кайру-7?", ["2169"], []),
    ("В каком году люди впервые встретили кайранов?", ["2187"], []),
    ("Как называется договор первой встречи?", ["аш кельск"], []),
    ("Как называется язык кайранов?", ["каирск"], []),
    ("Сколько примерно слов в языке кайранов?", ["4200"], []),
    ("Как называется религия кайранов?", ["культ двух солнц"], []),
    ("Сколько кайранов живёт на планете?", [], ["2.3", "2300000"]),
    ("Сколько килограммов урана заменяет один килограмм люмита?", ["400"], []),
    ("Как называется земной консорциум на Кайре-7?", ["аврора"], []),
    ("На какой планете штаб этого консорциума?", ["марс"], []),
    ("У системы Кайры-7 две звезды?", ["вейра"], ["нет", "нельзя", "одна"]),
    ("Какой город является столицей Франции?", ["париж"], []),
    ("Сколько будет 17 умножить на 2?", ["34"], []),
]


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def norm(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("-", " ")
    text = re.sub(r"(?<=\d) (?=\d)", "", text)
    text = text.replace(",", ".")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ok(answer: str, need: list, any_of: list) -> bool:
    text = norm(answer)
    if any(norm(piece) not in text for piece in need):
        return False
    if any_of and not any(norm(piece) in text for piece in any_of):
        return False
    return True


def visible(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S)
    if "<think>" in text.lower() and "</think>" not in text.lower():
        return ""
    return text.strip()


def load_tokenizer(base: str):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None) or processor
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(base: str, adapter: str | None, gpu: int):
    import torch
    import transformers
    from transformers import BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    kwargs = {
        "quantization_config": quant,
        "device_map": {"": 0},
        "trust_remote_code": True,
    }
    model = None
    last_error = None
    for name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(base, **kwargs)
            print(f"модель: {name}", flush=True)
            break
        except Exception as exc:
            last_error = exc
            print(f"{name} не подошёл: {exc}", flush=True)
    if model is None:
        raise RuntimeError(f"не удалось загрузить {base}") from last_error
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"адаптер: {adapter}", flush=True)
    model.eval()
    return model


def ask(model, tokenizer, question: str) -> str:
    import torch

    messages = [{"role": "user", "content": question}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if isinstance(prompt, list):
        prompt = prompt[0]
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return visible(tokenizer.decode(new_tokens, skip_special_tokens=True))


def main() -> None:
    _utf8()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gpu", type=int, default=0)
    pre_args, _ = pre.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pre_args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(description="Короткая проверка фактов Кайры-7")
    parser.add_argument("--gpu", type=int, default=pre_args.gpu)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.base)
    model = load_model(args.base, args.adapter, args.gpu)

    rows = []
    hits = 0
    for index, (question, need, any_of) in enumerate(QUIZ, start=1):
        answer = ask(model, tokenizer, question)
        good = ok(answer, need, any_of)
        hits += int(good)
        mark = "ок" if good else "нет"
        print(f"[{index}/{len(QUIZ)}] {mark} | {question}", flush=True)
        print(f"    {answer.replace(chr(10), ' ')[:300]}", flush=True)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "pass": good,
                "need": need,
                "any": any_of,
            }
        )

    out_dir = ROOT / "artifacts" / "simple"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": args.tag,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "adapter": args.adapter,
        "n": len(QUIZ),
        "passed": hits,
        "rows": rows,
    }
    path = out_dir / f"{args.tag}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"итог {hits}/{len(QUIZ)}", flush=True)
    print(f"файл {path}", flush=True)


if __name__ == "__main__":
    main()
