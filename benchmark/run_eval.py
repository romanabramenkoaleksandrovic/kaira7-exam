# Прогон экзамена. --gold гоняет эталоны, без модели.

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark.audit import answer_of, load_jsonl, question_of
from benchmark.scoring import (
    char_cosine,
    normalize_answer,
    render_eval_report,
    score_probes,
    summarize,
    token_f1,
)

ROOT = Path(__file__).resolve().parents[1]
PROBES_PATH = Path(__file__).resolve().parent / "probes.jsonl"
_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def visible_answer(text: str) -> str:
    # если think не закрыт, в ответе остаётся мусор, такой ответ выкидываю
    text = _THINK_CLOSED.sub("", text or "")
    text = re.sub(r"<\|think\|>.*?<\|/think\|>", "", text, flags=re.DOTALL)
    if re.search(r"<think>", text, re.IGNORECASE) and not re.search(r"</think>", text, re.IGNORECASE):
        return ""
    return text.strip()


def render_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


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


def quantize_linears_nf4(model, gpu: int):
    import gc

    import bitsandbytes as bnb
    import torch
    import torch.nn as nn

    device = torch.device(f"cuda:{gpu}")
    pairs = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if type(child) is nn.Linear
    ]
    for parent, name, child in pairs:
        layer = bnb.nn.LinearNF4(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
        )
        layer.weight = bnb.nn.Params4bit(
            child.weight.detach().to(dtype=torch.bfloat16).contiguous(),
            requires_grad=False,
            compress_statistics=True,
            quant_type="nf4",
            module=layer,
        )
        if child.bias is not None:
            layer.bias = nn.Parameter(child.bias.detach().to(dtype=torch.bfloat16), requires_grad=False)
        setattr(parent, name, layer.to(device))
    leftover = sum(type(module) is nn.Linear for module in model.modules())
    if leftover:
        raise RuntimeError(f"после квантования осталось Linear: {leftover}")
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            continue
        for param in module.parameters(recurse=False):
            if param.device.type != "cuda":
                dtype = torch.bfloat16 if param.is_floating_point() else param.dtype
                param.data = param.data.to(device=device, dtype=dtype)
        for buf in module.buffers(recurse=False):
            if buf.device.type != "cuda":
                buf.data = buf.data.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"NF4 слоёв: {len(pairs)}", flush=True)
    return model


def load_model(base: str, adapter: str | None, load_4bit: bool, gpu: int, merge: bool = False):
    import torch
    import transformers
    from transformers import BitsAndBytesConfig

    merge_bf16_first = bool(merge and load_4bit and adapter)
    quant = None
    if load_4bit and not merge_bf16_first:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    kwargs = {"trust_remote_code": True}
    if merge_bf16_first:
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "cpu"
        kwargs["low_cpu_mem_usage"] = True
    else:
        kwargs["device_map"] = f"cuda:{gpu}"
        if quant is not None:
            kwargs["quantization_config"] = quant
        else:
            kwargs["torch_dtype"] = torch.bfloat16
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
            print(f"{name} не подошёл: {type(exc).__name__}: {exc}", flush=True)
    if model is None:
        raise RuntimeError(f"не удалось загрузить {base}") from last_error
    lora = None
    if adapter:
        adapter_path = Path(adapter)
        config_path = adapter_path / "adapter_config.json"
        if not config_path.exists():
            raise SystemExit(f"в {adapter} нет adapter_config.json — это не папка PEFT")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        lora = json.loads(config_path.read_text(encoding="utf-8"))
        print("PEFT-адаптер подключён", flush=True)
        if merge_bf16_first:
            model = model.merge_and_unload()
            lora["merged_in_memory"] = True
            lora["merge_order"] = "bf16_then_nf4"
            print("адаптер влит в bf16, квантую NF4", flush=True)
            model = quantize_linears_nf4(model, gpu)
        elif merge:
            model = model.merge_and_unload()
            lora["merged_in_memory"] = True
            lora["merge_order"] = "weights_as_loaded"
            torch.cuda.empty_cache()
            print("адаптер влит в веса в памяти, на диск не пишу", flush=True)
    model.eval()
    return model, lora


def generate_one(model, tokenizer, question: str, max_new_tokens: int) -> str:
    import torch

    prompt = render_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return visible_answer(tokenizer.decode(new_tokens, skip_special_tokens=True))


def score_val(world: Path, answers: dict[str, str]) -> dict:
    train_path = world / "dataset" / "train" / "train.jsonl"
    val_path = world / "dataset" / "val" / "val.jsonl"
    train_ids = set()
    if train_path.exists():
        train_ids = {row.get("id") for row in load_jsonl(train_path)}
    rows = load_jsonl(val_path)
    f1s = []
    clean = []
    exact = 0
    leaked = 0
    for index, row in enumerate(rows):
        key = row.get("id") or f"val-{index}"
        prediction = answers.get(key, "")
        gold = answer_of(row)
        score = token_f1(prediction, gold)
        f1s.append(score)
        if normalize_answer(prediction) == normalize_answer(gold) and gold.strip():
            exact += 1
        if row.get("id") in train_ids:
            leaked += 1
        else:
            clean.append(score)
    n = len(f1s)
    return {
        "n": n,
        "mean_token_f1": (sum(f1s) / n) if n else None,
        "exact_match": exact,
        "leaked_ids": leaked,
        "clean_n": len(clean),
        "clean_mean_token_f1": (sum(clean) / len(clean)) if clean else None,
    }


def build_summary(tag: str, probes_path: Path, records_summary: dict, model_meta: dict, val_summary: dict | None) -> dict:
    payload = {
        "tag": tag,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "probes_sha256": hashlib.sha256(probes_path.read_bytes()).hexdigest() if probes_path.exists() else None,
        "model": model_meta,
    }
    payload.update(records_summary)
    if val_summary is not None:
        payload["val"] = val_summary
    return payload


def write_run(out_dir: Path, summary: dict, predictions: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(render_eval_report(summary), encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_dir}", flush=True)


def prediction_row(probe: dict, answer: str, scored: dict) -> dict:
    return {
        "id": probe["id"],
        "category": probe["category"],
        "track": probe["track"],
        "question": probe["question"],
        "gold": probe.get("gold"),
        "answer": answer,
        "pass": scored["pass"],
        "missed_slots": [slot["id"] for slot in scored["slots"] if not slot["ok"]],
        "denial_ok": scored["denial_ok"],
        "cosine": scored.get("cosine"),
    }


def lora_public(config: dict | None) -> dict | None:
    if not config:
        return None
    keys = (
        "peft_type",
        "task_type",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "base_model_name_or_path",
        "bias",
        "use_dora",
        "merged_in_memory",
        "merge_order",
    )
    return {key: config.get(key) for key in keys if key in config}


def main() -> None:
    _utf8_stdio()
    parser = argparse.ArgumentParser(description="Экзамен Кайра-7: база или PEFT")
    parser.add_argument("--base", default=r"D:\models\Qwen3.5-4B-usable")
    parser.add_argument("--adapter", default=None, help="Папка PEFT с adapter_config.json")
    parser.add_argument("--world-ft", type=Path, default=Path(r"C:\Projects\world_ft"))
    parser.add_argument("--probes", type=Path, default=PROBES_PATH)
    parser.add_argument("--split", choices=("probes", "val", "both"), default="probes")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Влить адаптер в bf16, затем при --load-4bit сжать NF4",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int, default=0, help="Сколько пунктов взять, 0 = все")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--gold", action="store_true", help="Прогнать эталоны через рубрику, без модели")
    parser.add_argument("--predictions", type=Path, default=None, help="Пересчитать готовый predictions.jsonl")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    tag = args.tag or ("gold-rubric" if args.gold else ("peft" if args.adapter else "base"))
    out_dir = args.out or (ROOT / "artifacts" / "runs" / tag)
    probes = load_jsonl(args.probes)
    if args.limit:
        probes = probes[: args.limit]

    answers: dict[str, str] = {}
    model_meta: dict = {
        "mode": "gold-rubric" if args.gold else "generate",
        "base": None if args.gold else args.base,
        "adapter": args.adapter,
        "load_4bit": bool(args.load_4bit),
        "merged_in_memory": bool(args.merge),
        "merge_order": "bf16_then_nf4" if args.merge and args.load_4bit and args.adapter else None,
        "gpu": args.gpu,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": False,
    }

    if args.gold:
        answers = {probe["id"]: probe["gold"] for probe in probes}
    elif args.predictions:
        for row in load_jsonl(args.predictions):
            text = row.get("answer", row.get("prediction", row.get("output", "")))
            answers[row["id"]] = text
        model_meta["mode"] = "rescore"
        model_meta["predictions"] = str(args.predictions)
    else:
        tokenizer = load_tokenizer(args.base)
        model, lora = load_model(args.base, args.adapter, args.load_4bit, args.gpu, merge=args.merge)
        model_meta["lora"] = lora_public(lora)
        try:
            import peft
            import torch
            import transformers

            model_meta["versions"] = {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "cuda": torch.cuda.get_device_name(args.gpu),
            }
        except Exception as exc:
            model_meta["versions_error"] = str(exc)
        if args.split in {"probes", "both"}:
            for index, probe in enumerate(probes, 1):
                print(f"[{index}/{len(probes)}] {probe['id']} ...", flush=True)
                answers[probe["id"]] = generate_one(model, tokenizer, probe["question"], args.max_new_tokens)
                print(f"[{index}/{len(probes)}] {probe['id']} готово", flush=True)
        if args.split in {"val", "both"}:
            val_rows = load_jsonl(args.world_ft / "dataset" / "val" / "val.jsonl")
            if args.limit:
                val_rows = val_rows[: args.limit]
            for index, row in enumerate(val_rows, 1):
                key = row.get("id") or f"val-{index}"
                print(f"[val {index}/{len(val_rows)}] {key} ...", flush=True)
                answers[key] = generate_one(model, tokenizer, question_of(row), args.max_new_tokens)
                print(f"[val {index}/{len(val_rows)}] {key} готово", flush=True)

    records = score_probes(probes, answers) if args.split in {"probes", "both"} or args.gold or args.predictions else []
    for record, probe in zip(records, probes):
        record["cosine"] = char_cosine(record["answer"], probe.get("gold") or "")
    summary_body = summarize(records) if records else {
        "schema_version": 1,
        "primary": {"n": 0, "passed": 0, "accuracy": None},
        "by_category": {},
        "tracks": {},
        "negatives": {},
        "consistency_groups": {"groups": {}, "n": 0, "passed": 0, "accuracy": None},
        "failures": [],
    }
    val_summary = None
    if args.split in {"val", "both"} and not args.gold:
        val_summary = score_val(args.world_ft, answers)
    summary = build_summary(tag, args.probes, summary_body, model_meta, val_summary)
    by_id = {row["id"]: row for row in records}
    predictions = [prediction_row(probe, answers.get(probe["id"], ""), by_id[probe["id"]]) for probe in probes if probe["id"] in by_id]
    write_run(out_dir, summary, predictions)
    primary = summary["primary"]
    if primary["n"]:
        print(f"primary {primary['passed']}/{primary['n']} = {primary['accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
