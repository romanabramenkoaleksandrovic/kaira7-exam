# Обучение Кайры-7: qlora или qdora, одна карта.
# Карту надо выбрать до импорта torch, иначе он уже занял обе.

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = r"D:\models\Qwen3.5-4B-usable"
DEFAULT_TRAIN = r"C:\Projects\world_ft\dataset\train\train.jsonl"
TARGET_RE = (
    r"model\.language_model\..*\.(?:"
    r"q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|"
    r"in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$"
)


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def apply_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(text, list):
        text = text[0]
    return text


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            if len(messages) < 2:
                continue
            if messages[-1].get("role") != "assistant":
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"в {path} нет обучающих пар")
    return rows


def encode(tokenizer, row: dict, max_length: int) -> dict:
    messages = row["messages"]
    full_text = apply_template(tokenizer, messages, add_generation_prompt=False)
    prompt_text = apply_template(tokenizer, messages[:-1], add_generation_prompt=True)
    full = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    plen = len(prompt)
    if len(full) > max_length:
        overflow = len(full) - max_length
        full = full[-max_length:]
        plen = max(0, plen - overflow)
    labels = list(full)
    # вопрос в лосс не кладу, учится только ответ
    labels[:plen] = [-100] * min(plen, len(labels))
    return {
        "input_ids": list(full),
        "attention_mask": [1] * len(full),
        "labels": labels,
    }


class EncodedDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]):
        import torch

        width = max(len(item["input_ids"]) for item in features)
        input_ids, attention, labels = [], [], []
        for item in features:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_id] * pad)
            attention.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_tokenizer(base: str):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None) or processor
    except Exception:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, processor


def load_model(base: str):
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
    last_error = None
    for name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(base, **kwargs)
            print(f"модель: {name}", flush=True)
            return model
        except Exception as exc:
            last_error = exc
            print(f"{name} не подошёл: {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError(f"не удалось загрузить {base}") from last_error


def park_vision(model) -> None:
    # картиночная часть на 8 гб вместе с обучением не влезает
    for name, module in model.named_modules():
        if name.split(".")[-1] != "visual":
            continue
        try:
            module.to("cpu")
            for param in module.parameters():
                param.requires_grad_(False)
            print(f"vision на CPU: {name}", flush=True)
        except Exception as exc:
            print(f"vision остаётся на GPU ({name}): {type(exc).__name__}: {exc}", flush=True)
        return


def silence_cache(model) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return
    config.use_cache = False
    text = getattr(config, "text_config", None)
    if text is not None:
        text.use_cache = False


def trainable_counts(model) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def main() -> None:
    _utf8_stdio()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gpu", type=int, default=0)
    pre_args, _ = pre.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pre_args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(description="QLoRA / QDoRA на Кайре-7")
    parser.add_argument("--method", choices=("qlora", "qdora"), required=True)
    parser.add_argument("--gpu", type=int, default=pre_args.gpu)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--train", type=Path, default=Path(DEFAULT_TRAIN))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    out = args.out or (ROOT / "adapters" / args.method)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"method={args.method} gpu={args.gpu} device={torch.cuda.get_device_name(0)} out={out}",
        flush=True,
    )

    tokenizer, processor = load_tokenizer(args.base)
    rows = load_rows(args.train, args.limit_samples)
    encoded = [encode(tokenizer, row, args.max_length) for row in rows]
    lengths = sorted(len(item["input_ids"]) for item in encoded)
    supervised = sum(sum(1 for token in item["labels"] if token != -100) for item in encoded)
    print(
        f"примеров={len(encoded)} длина p50={lengths[len(lengths)//2]} max={lengths[-1]} "
        f"supervised_tokens={supervised}",
        flush=True,
    )

    model = load_model(args.base)
    park_vision(model)
    silence_cache(model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=TARGET_RE,
        use_dora=args.method == "qdora",
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    trainable, total = trainable_counts(model)

    train_kwargs = dict(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        bf16=True,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=42,
        data_seed=42,
        disable_tqdm=False,
    )
    if args.max_steps:
        train_kwargs["max_steps"] = args.max_steps
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**train_kwargs),
        train_dataset=EncodedDataset(encoded),
        data_collator=PadCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    print("старт обучения", flush=True)
    result = trainer.train()
    model.save_pretrained(out)
    if processor is not None:
        processor.save_pretrained(out)
    else:
        tokenizer.save_pretrained(out)

    meta = {
        "method": args.method,
        "use_dora": args.method == "qdora",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base": args.base,
        "train": str(args.train),
        "examples": len(encoded),
        "epochs": args.epochs,
        "max_steps": args.max_steps or None,
        "max_length": args.max_length,
        "learning_rate": args.lr,
        "lr_scheduler": "cosine",
        "rank": args.rank,
        "lora_alpha": args.rank * 2,
        "target_modules": TARGET_RE,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_index": args.gpu,
        "trainable_params": trainable,
        "total_params": total,
        "train_loss": result.training_loss,
        "global_steps": result.global_step,
        "log_history": trainer.state.log_history,
        "adapter": str(out),
    }
    report_dir = ROOT / "artifacts" / "train"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.method}.json"
    report_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"loss={result.training_loss:.4f} steps={result.global_step}", flush=True)
    print(f"adapter={out}", flush=True)
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
