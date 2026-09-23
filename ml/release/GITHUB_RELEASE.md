# FinnAI SLM v2 — Open Source Release

**Apache 2.0** · Based on [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

On-device Indian bank SMS → JSON + finance coaching for [FinnDot](https://github.com/devaka207/Finndot).

## Results (held-out test)

| Metric | FinnAI v2 | Qwen3-1.7B base | Qwen2.5-1.5B |
|---|---|---|---|
| SMS R-EM % | **97.97** | 28.16 | 42.43 |
| Amount EM % | **99.53** | 82.84 | 89.78 |
| Chat groundedness % | **68.0** | 26.0 | 68.0 |
| False-parse % | **0.54** | 100.0 | 47.83 |

**SHIP: YES** — eval report in this release.

## What's in this release

| Asset | Description | Size |
|---|---|---|
| `adapter/` | QLoRA adapter (rank 16) — apply on top of Qwen3-1.7B | ~67 MB |
| `dataset/` | Synthetic train/val/test jsonl (no user SMS) | ~30 MB |
| `docs/` | Model card, finetune guide, dataset card, eval report | — |
| `*.litertlm` | INT4 on-device file (when conversion finishes) | ~0.9 GB |

> Full merged bf16 (~3.2 GB) is not on GitHub (file size limit). Merge locally from the adapter, or wait for Hugging Face once an account exists.

## Quick start — use the adapter

```bash
pip install transformers peft torch

python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "./adapter")
model = model.merge_and_unload()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

messages = [
    {"role": "system", "content": "Extract transaction details from this SMS as JSON: {amount, merchant, type, account, balance, category}. If not a transaction, return {}."},
    {"role": "user", "content": "HDFC Bank: Rs.499.00 debited from A/c XX1234 to SWIGGY via UPI. Avl Bal Rs.15000.50"},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
PY
```

## Reproduce training

See `docs/FINETUNE_GUIDE.md` and the `ml/` folder in this repo.

```bash
cd ml
python -m data.build_splits --repo-root .. --out data/out --train-size 16000
python train/sft_qlora.py --config train/train_config.yaml
```

## License

Apache 2.0. Keep Qwen attribution. No user SMS in training data.
