"""FinnAI SLM — QLoRA SFT of Qwen3-1.7B. SageMaker (SM_CHANNEL_TRAIN) or local GPU."""

from __future__ import annotations

import argparse
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml

MODEL_NAME = "FinnAI-SLM"
THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def _dec(value) -> Decimal | None:
    if value is None or value == "":
        return None
    s = re.sub(r"(Rs\.?|INR|₹)", "", str(value), flags=re.I)
    s = s.replace(",", "").replace(" ", "").strip()
    if s.lower() in {"null", "none"}:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _parse_json(text: str) -> dict | None:
    text = FENCE_RE.sub("", THINK_RE.sub("", text or "")).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _merchant(value) -> str:
    s = "" if value is None else str(value).strip().casefold()
    s = re.sub(r"^upi/", "", s)
    return re.sub(r"\s+", " ", s)


def strict_rem(pred_text: str, gold: dict | None) -> bool:
    obj = _parse_json(pred_text)
    empty_gold = not gold or (gold.get("amount") in (None, "") and gold.get("type") in (None, ""))
    if obj is None:
        return False
    if empty_gold:
        return not obj or (obj.get("amount") in (None, "") and obj.get("type") in (None, ""))
    gold = gold or {}
    if _dec(obj.get("amount")) != _dec(gold.get("amount")):
        return False
    if str(obj.get("type") or "").upper() != str(gold.get("type") or "").upper():
        return False
    if _merchant(obj.get("merchant")) != _merchant(gold.get("merchant")):
        return False
    pred_acc = re.sub(r"\D", "", str(obj.get("account") or ""))[-4:] or None
    gold_acc = re.sub(r"\D", "", str(gold.get("account") or ""))[-4:] or None
    if pred_acc != gold_acc:
        return False
    return _dec(obj.get("balance")) == _dec(gold.get("balance"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_sms_rem(model, tok, rows: list[dict], max_items: int, device) -> float:
    import torch

    subset = [r for r in rows if r.get("task") == "sms"][:max_items]
    if not subset:
        return 0.0
    model.eval()
    hits = 0
    with torch.no_grad():
        for row in subset:
            messages = [m for m in row["messages"] if m["role"] != "assistant"]
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tok(text, return_tensors="pt").to(device)
            gen = model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
            )
            new_tokens = gen[0][inputs["input_ids"].shape[1] :]
            pred = tok.decode(new_tokens, skip_special_tokens=True)
            if strict_rem(pred, row.get("gold")):
                hits += 1
    model.train()
    return hits / len(subset)


def main() -> None:
    p = argparse.ArgumentParser(description=f"Train {MODEL_NAME} (QLoRA SFT)")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("train_config.yaml"))
    p.add_argument("--train-file", type=Path, default=None)
    p.add_argument("--val-file", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_dir = _env_path("SM_CHANNEL_TRAIN", str(Path("data/out")))
    out_dir = _env_path("SM_MODEL_DIR", str(args.output_dir or Path("train/out")))
    train_file = args.train_file or train_dir / "train.jsonl"
    val_file = args.val_file or train_dir / "val.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        TrainingArguments,
    )

    model_id = cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_4bit = bool(cfg.get("quant", {}).get("load_in_4bit", True)) and torch.cuda.is_available()
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": str(train_file), "val": str(val_file)})

    def to_text(ex):
        kwargs = dict(tokenize=False, add_generation_prompt=False)
        try:
            text = tok.apply_chat_template(ex["messages"], enable_thinking=False, **kwargs)
        except TypeError:
            text = tok.apply_chat_template(ex["messages"], **kwargs)
        return {"text": text}

    keep = {"messages"}
    ds = ds.map(to_text, remove_columns=[c for c in ds["train"].column_names if c not in keep])

    t = cfg["train"]
    common_args = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=float(t["lr"]),
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        eval_strategy="epoch",
        bf16=bool(torch.cuda.is_available()),
        fp16=False,
        lr_scheduler_type="cosine",
        report_to=[],
        seed=cfg["seed"],
        gradient_checkpointing=True,
        save_total_limit=3,
        load_best_model_at_end=False,
        run_name=cfg.get("model_name", MODEL_NAME),
    )

    val_rows = load_jsonl(val_file)
    rem_limit = int(t.get("val_rem_max_items", 256))
    rem_log: list[dict] = []

    class ValRemCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            unwrapped = kwargs.get("model")
            if unwrapped is None:
                return
            device = next(unwrapped.parameters()).device
            rem = evaluate_sms_rem(unwrapped, tok, val_rows, rem_limit, device)
            rec = {"epoch": state.epoch, "val_sms_rem": rem, "step": state.global_step}
            rem_log.append(rec)
            print(f"[{MODEL_NAME}] epoch={state.epoch} val_sms_R-EM={rem:.4f}", flush=True)
            (out_dir / "val_rem.json").write_text(json.dumps(rem_log, indent=2) + "\n")

    from trl import SFTTrainer

    try:
        from trl import SFTConfig

        sft_args = SFTConfig(
            **common_args,
            max_length=t["max_seq_len"],
            packing=t["packing"],
            dataset_text_field="text",
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=ds["train"],
            eval_dataset=ds["val"],
            processing_class=tok,
            callbacks=[ValRemCallback()],
        )
    except TypeError:
        training_args = TrainingArguments(**common_args)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["val"],
            processing_class=tok,
            max_seq_length=t["max_seq_len"],
            packing=t["packing"],
            dataset_text_field="text",
            callbacks=[ValRemCallback()],
        )

    trainer.train()
    adapter = out_dir / "adapter"
    trainer.save_model(str(adapter))
    tok.save_pretrained(str(adapter))
    meta = {
        "model_name": cfg.get("model_name", MODEL_NAME),
        "base_model": model_id,
        "method": "QLoRA-SFT" if use_4bit else "LoRA-SFT",
        "config": cfg,
        "train_file": str(train_file),
        "val_sms_rem": rem_log,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(adapter)


if __name__ == "__main__":
    main()
