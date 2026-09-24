# FinnAI SLM

**On-device Indian bank SMS → JSON + personal finance coaching**

FinnAI SLM is a QLoRA fine-tune of [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) for the [FinnDot](https://github.com/devaka207/Finndot) expense tracker. It runs fully on-device (privacy-first) and is released under **Apache 2.0**.

## Artifacts (ready to use)

| What | Link |
| --- | --- |
| **Model** (merged bf16 + LoRA adapter) | https://huggingface.co/finndot/finnai-slm-v2 |
| **Dataset** (synthetic train/val/test) | https://huggingface.co/datasets/finndot/finnai-slm-data |
| App | [FinnDot on GitHub](https://github.com/devaka207/Finndot) |

## Results (v2 held-out eval)

**Non-empty transaction rows only** (1,098 SMS with a real spend/credit):

| Metric | FinnAI v2 | Qwen3-1.7B | Qwen2.5-1.5B |
| --- | ---: | ---: | ---: |
| Amount EM % | **99.54** | **96.72** | 96.08 |
| Merchant exact % | **98.00** | 71.58 | 76.32 |
| Strict R-EM % | **97.72** | 32.88 | 40.80 |

Bases already get the rupee amount right on real transactions. FinnAI wins on full-field match.

**Empty rows separately** (184 OTP / promo / failed UPI): false-parse FinnAI **0.54%**, Qwen3 **100%**, Qwen2.5 **47.83%** — untuned models invent spends when nothing is there; fine-tuning fixes that.

Full report: [`docs/eval-v2-report.md`](docs/eval-v2-report.md)

## What's in this repo

Ready-made code to **reproduce**, **fine-tune further**, or **adapt** to similar domains (invoices, receipts, medical notes, etc.):

```text
ml/
  data/     # synthetic SMS + coach generators, split builder
  train/    # QLoRA SFT (sft_qlora.py) + merge + LiteRT export
  eval/     # R-EM metrics, ship gates, report writer
  aws/      # optional EC2 / SageMaker launch helpers
  oss/      # Hugging Face publish helper
docs/
  FINETUNE_GUIDE.md   # why / when / how (start here)
  MODEL_CARD.md
  DATASET_CARD.md
  llm-eval-protocol.md
  finnai-slm-finetune.md
```

## Quick start

### 1. Use the published model

```bash
pip install transformers peft torch accelerate

python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "finndot/finnai-slm-v2"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

messages = [
    {
        "role": "system",
        "content": (
            "Extract transaction details from this SMS as JSON: "
            "{amount, merchant, type, account, balance, category}. "
            "type is one of: INCOME, EXPENSE, CREDIT, TRANSFER, INVESTMENT. "
            "If this is not a transaction (OTP, promo, KYC), return {}."
        ),
    },
    {
        "role": "user",
        "content": "HDFC Bank: Rs.499.00 debited from A/c XX1234 to SWIGGY via UPI. Avl Bal Rs.15000.50",
    },
]
text = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
PY
```

Or load only the LoRA adapter on top of the base:

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", ...)
model = PeftModel.from_pretrained(base, "finndot/finnai-slm-v2", subfolder="adapter")
```

### 2. Rebuild the dataset locally

```bash
cd ml
python -m pip install -r requirements.txt
# Synthetic-only (no FinnDot monorepo needed):
python -c "from data.generate_synthetic_sms import generate as g; print(len(g()))"
python -m data.build_splits --repo-root . --out data/out --train-size 12000
```

> Parser-gold extraction (`extract_parser_gold.py`) needs the FinnDot `parser-core` tests. Clone [devaka207/Finndot](https://github.com/devaka207/Finndot) and pass `--repo-root /path/to/Finndot` for the full mix.

### 3. Train (1× A10G / T4 GPU)

```bash
cd ml
export SM_CHANNEL_TRAIN=$PWD/data/out
export SM_MODEL_DIR=$PWD/train/output
python train/sft_qlora.py --config train/train_config.yaml
python train/merge_lora.py --adapter train/output/adapter --out train/output/merged
```

### 4. Evaluate

```bash
python -m eval.run_eval \
  --backend hf \
  --sms data/out/test_sms.jsonl \
  --chat eval/fixtures/chat_eval.jsonl \
  --model-id train/output/merged \
  --baseline-id Qwen/Qwen3-1.7B \
  --tag my-run
```

## Documentation

| Doc | Purpose |
| --- | --- |
| [`docs/FINETUNE_GUIDE.md`](docs/FINETUNE_GUIDE.md) | **Start here** — problem, when to use this approach, data recipe, training, adapting to other domains |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Model card |
| [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) | Dataset card |
| [`docs/llm-eval-protocol.md`](docs/llm-eval-protocol.md) | Pre-registered ship gates |
| [`docs/finnai-slm-finetune.md`](docs/finnai-slm-finetune.md) | Lab notebook / ops |
| [`docs/FinnAI_SLM_CXO_FineTuning_Brief.docx`](docs/FinnAI_SLM_CXO_FineTuning_Brief.docx) | **CXO briefing** — flowcharts + plain English + full technical depth |

## Privacy

- **No user SMS** in training data
- Dataset is **100% synthetic** (+ parser-core JUnit gold with synthetic values)
- Nova Pro Indic expansions are **not** in the public dataset dump (regenerate locally if needed)

## License

Apache 2.0 — same as Qwen3-1.7B. Keep Qwen attribution.

## Citation

```bibtex
@misc{finnai-slm-2026,
  title={FinnAI SLM: On-device Indian bank SMS parsing with QLoRA fine-tuning},
  author={FinnDot Team},
  year={2026},
  url={https://huggingface.co/finndot/finnai-slm-v2}
}
```

## Community

- Issues / PRs welcome on this repo
- Product app: [FinnDot](https://github.com/devaka207/Finndot)
- Model discussion: [HF model page](https://huggingface.co/finndot/finnai-slm-v2)
