"""DPO / preference stage for FinnAI v3 (groundedness).

Expects preference jsonl from data.generate_dpo_prefs with fields:
  prompt: [{role, content}, ...]
  chosen: str
  rejected: str

Loads a LoRA adapter from SFT (optional) on top of the base, then runs TRL DPO.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def main() -> None:
    p = argparse.ArgumentParser(description="FinnAI v3 DPO training")
    p.add_argument("--config", type=Path, required=True, help="Same family yaml as SFT (base + lora)")
    p.add_argument("--prefs", type=Path, required=True)
    p.add_argument("--adapter", type=Path, default=None, help="SFT adapter dir to continue from")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = args.output_dir or _env_path("SM_MODEL_DIR", "train/output_v3_dpo")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    model_id = cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base)

    if args.adapter and Path(args.adapter).exists():
        model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=True)
    else:
        lora = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(base, lora)

    rows = []
    with args.prefs.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    def fmt_prompt(messages: list[dict]) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tok.apply_chat_template(messages, **kwargs)

    ds_rows = []
    for r in rows:
        ds_rows.append(
            {
                "prompt": fmt_prompt(r["prompt"]),
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
        )
    ds = Dataset.from_list(ds_rows)

    dpo_args = DPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.lr,
        beta=args.beta,
        logging_steps=10,
        save_strategy="epoch",
        bf16=bool(torch.cuda.is_available()),
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    adapter = out_dir / "adapter"
    trainer.save_model(str(adapter))
    tok.save_pretrained(str(adapter))
    meta = {
        "base_model": model_id,
        "sft_adapter": str(args.adapter) if args.adapter else None,
        "prefs": str(args.prefs),
        "n": len(ds_rows),
        "beta": args.beta,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    (out_dir / "dpo_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(adapter)


if __name__ == "__main__":
    main()
