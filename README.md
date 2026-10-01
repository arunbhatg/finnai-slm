# FinnAI SLM

**On-device Indian bank SMS → JSON + personal finance coaching**

FinnAI SLM is a QLoRA fine-tune of [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) for on-device expense tracking. It runs fully on-device (privacy-first) and is released under **Apache 2.0**.

## Artifacts (ready to use)

| What | Link |
| --- | --- |
| **Model** (merged bf16 + LoRA adapter) | https://huggingface.co/finndot/finnai-slm-v2 |
| **On-device model** (INT4 `.litertlm` for Android) | [Download (974 MB)](https://dgdzwh27431n8.cloudfront.net/models/qwen3-1.7b-finndot/latest/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm) |
| **Dataset** (synthetic train/val/test) | https://huggingface.co/datasets/finndot/finnai-slm-data |
| **Android App** | [Download on Play Store](https://play.google.com/store/apps/details?id=com.anomapro.finndot) |

## Results (v2 held-out eval)

**Non-empty transaction rows only** (1,098 SMS with a real spend/credit):

| Metric | FinnAI v2 | Qwen3-1.7B (untuned) |
| --- | ---: | ---: |
| Amount EM % | **99.54** | **96.72** |
| Merchant exact % | **98.00** | 71.58 |
| Strict R-EM % | **97.72** | 32.88 |

Untuned Qwen3 already gets the rupee amount right on real transactions. FinnAI wins on full-field match.

**Empty rows separately** (184 OTP / promo / failed UPI): false-parse FinnAI **0.54%** vs Qwen3 **100%** — the base invents spends when nothing is there; fine-tuning fixes that.

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

## For App Users

The Android app (available on Play Store) automatically downloads this model on first launch. No manual steps needed!

**App Download:** [Get it on Play Store](https://play.google.com/store/apps/details?id=com.anomapro.finndot)

**Model Performance:**
- 97.97% accuracy on Indian bank SMS parsing
- Supports 35+ Indian banks
- Languages: EN, HI, Hinglish, TA, TE, MR, BN
- 100% on-device (no cloud, no data upload)

### Manual Model Download

If you need the model file directly (974 MB):

**Using curl:**
```bash
curl -L -O "https://dgdzwh27431n8.cloudfront.net/models/qwen3-1.7b-finndot/latest/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm"
```

**Using wget:**
```bash
wget "https://dgdzwh27431n8.cloudfront.net/models/qwen3-1.7b-finndot/latest/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm"
```

**Using Python:**
```python
import urllib.request

url = "https://dgdzwh27431n8.cloudfront.net/models/qwen3-1.7b-finndot/latest/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm"
filename = "finnai_model.litertlm"

print("Downloading model (974 MB)...")
urllib.request.urlretrieve(url, filename)
print(f"Downloaded to: {filename}")
```

**Using Python with progress bar:**
```python
import requests
from tqdm import tqdm

url = "https://dgdzwh27431n8.cloudfront.net/models/qwen3-1.7b-finndot/latest/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm"
filename = "finnai_model.litertlm"

response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(filename, 'wb') as file, tqdm(
    desc=filename,
    total=total_size,
    unit='iB',
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for data in response.iter_content(chunk_size=1024):
        size = file.write(data)
        bar.update(size)

print(f"✓ Downloaded successfully: {filename}")
```

**File Details:**
- Size: 973,979,088 bytes (974 MB)
- Format: LiteRT-LM INT4 `.litertlm`
- Compatible with: LiteRT inference engine (Android, Linux, macOS)

---

## For Developers

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
# Synthetic SMS generation:
python -c "from data.generate_synthetic_sms import generate as g; print(len(g()))"
python -m data.build_splits --repo-root . --out data/out --train-size 12000
```

### 3. Train (1├ù A10G / T4 GPU)

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
| [`docs/FINETUNE_GUIDE.md`](docs/FINETUNE_GUIDE.md) | **Start here** ΓÇö problem, when to use this approach, data recipe, training, adapting to other domains |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Model card |
| [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) | Dataset card |
| [`docs/llm-eval-protocol.md`](docs/llm-eval-protocol.md) | Pre-registered ship gates |
| [`docs/finnai-slm-finetune.md`](docs/finnai-slm-finetune.md) | Lab notebook / ops |
| [`docs/FinnAI_SLM_CXO_FineTuning_Brief.docx`](docs/FinnAI_SLM_CXO_FineTuning_Brief.docx) | **CXO briefing** ΓÇö flowcharts + plain English + full technical depth |

## Privacy

- **No user SMS** in training data
- Dataset is **100% synthetic** (+ parser-core JUnit gold with synthetic values)
- Nova Pro Indic expansions are **not** in the public dataset dump (regenerate locally if needed)

## License

Apache 2.0 ΓÇö same as Qwen3-1.7B. Keep Qwen attribution.

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
- Model discussion: [HF model page](https://huggingface.co/finndot/finnai-slm-v2)
- Android App: [Play Store](https://play.google.com/store/apps/details?id=com.anomapro.finndot)
