---
language:
- en
- hi
- ta
- te
- mr
- bn
license: apache-2.0
library_name: transformers
tags:
- finndot
- finance
- sms-parsing
- on-device
- qlora
- qwen3
- expense-tracker
- indian-banking
base_model: Qwen/Qwen3-1.7B
datasets:
- finndot/finnai-slm-data
model-index:
- name: FinnAI-SLM-v2
  results:
  - task:
      type: text-generation
      name: Indian Bank SMS → JSON extraction
    dataset:
      name: FinnAI SMS test (synthetic)
      type: finndot/finnai-slm-data
    metrics:
    - name: Strict Record Exact-Match (R-EM)
      type: exact_match
      value: 97.97
      verified: true
    - name: Amount Exact-Match
      type: exact_match
      value: 99.53
      verified: true
    - name: JSON Validity
      type: accuracy
      value: 100.0
      verified: true
---

# FinnAI SLM v2

**A 1.7B on-device language model for Indian bank SMS parsing and personal finance coaching.**

FinnAI SLM is a [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) derivative fine-tuned with QLoRA SFT for the [FinnDot](https://github.com/AnomaProAI/Finndot) expense tracker app. It runs entirely on-device — no cloud calls, no user data leaves the phone.

## What it does

| Task | Description |
|---|---|
| **SMS → JSON** | Extracts `{amount, merchant, type, account, balance, category}` from Indian bank SMS (UPI, NEFT, IMPS, credit card, salary, ATM) |
| **Finance coach** | Answers personal finance questions grounded in the user's ledger (spending breakdown, saving tips) |
| **General finance Q&A** | Short answers about SIP, EMI, tax, budgeting |

The model handles **English, Hindi, Hinglish, Tamil, Telugu, Marathi, and Bengali** bank SMS.

## Key results (v2 held-out eval)

Test set: **1,098 non-empty transaction SMS** + 184 empty/non-transaction SMS (OTP, promo, failed UPI, …). Ship gates still use the full mix; the extraction comparison below is **non-empty rows only**.

### Non-empty rows only (1,098 real transactions)

| Metric | FinnAI v2 | Qwen3-1.7B (untuned) | Qwen2.5-1.5B-Instruct (untuned) |
|---|---|---|---|
| Strict R-EM % | **97.72** | 32.88 | 40.80 |
| Amount EM % | **99.54** | **96.72** | 96.08 |
| Merchant exact % | **98.00** | 71.58 | 76.32 |

On real spends, untuned bases already read **amounts** well (~96%). FinnAI’s gain here is mostly **full-field** match (merchant / account / type together), not “learning what a rupee is.”

### Empty / non-transaction rows (184) — separate product check

| Metric | FinnAI v2 | Qwen3-1.7B | Qwen2.5-1.5B-Instruct |
|---|---|---|---|
| False-parse % (invented a spend instead of `{}`) | **0.54** | **100.0** | 47.83 |

Newer untuned chat models over-help: they emit JSON even when nothing is there. That got **worse** from Qwen2.5 → Qwen3. Fine-tuning teaches refusal. (Overall R-EM on all 1,282 rows mixes these two slices: FinnAI **97.97%**, Qwen3 **28.16%**, Qwen2.5 **42.43%**.)

Chat groundedness (50-item set): FinnAI **68%**, Qwen3 **26%**, Qwen2.5 **68%**.

**SHIP: YES.** Gates 1–4 pass. McNemar p = 7.6e-270. Qwen2.5 / Qwen3 columns are untuned checkpoints; FinnAI is the only fine-tune. Non-empty metrics derived from published aggregates + the 1,098 / 184 split.

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "finndot/finnai-slm-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# SMS parsing
messages = [
    {"role": "system", "content": "Extract transaction details from this SMS as JSON: {amount, merchant, type, account, balance, category}. type is one of: INCOME, EXPENSE, CREDIT, TRANSFER, INVESTMENT. If this is not a transaction (OTP, promo, KYC), return {}."},
    {"role": "user", "content": "HDFC Bank: Rs.499.00 debited from A/c XX1234 to SWIGGY via UPI. Avl Bal Rs.15000.50"}
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(result)
# {"amount": 499.0, "merchant": "SWIGGY", "type": "EXPENSE", "account": "1234", "balance": 15000.5, "category": "Food"}
```

## Training details

| Parameter | Value |
|---|---|
| Base model | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |
| Method | QLoRA SFT (4-bit NF4, double quant, bf16 compute) |
| LoRA rank / alpha | 16 / 32 |
| LoRA targets | q, k, v, o, gate, up, down_proj |
| Dropout | 0.05 |
| Epochs | 3 |
| Learning rate | 2e-4 cosine, 3% warmup |
| Effective batch | 16 (2 × 8 grad accum) |
| Max sequence | 1536 |
| Training data | 16,000 examples (60% SMS, 25% coach, 15% general) |
| Checkpoint selection | Best validation SMS strict R-EM |
| Thinking | **Disabled** (`enable_thinking=false`) |
| Hardware | 1× NVIDIA A10G (g5.2xlarge), ~5 hours |
| Seed | 42 |

### Data recipe (privacy-safe)

**No user SMS. No real financial data. Fully synthetic.**

| Source | Description | Count |
|---|---|---|
| Parser gold | JUnit test cases from `parser-core` (35+ Indian banks) | ~3,000 |
| Synthetic SMS | Template-generated UPI/NEFT/IMPS/CC/salary/ATM in English + Hindi/Hinglish | ~6,000 |
| Indic SMS | Hindi, Hinglish, Tamil, Telugu, Marathi, Bengali bank SMS paraphrases | ~300 |
| Finance coach | Grounded Q&A with synthetic ledgers (English + Indic) | ~4,000 |
| General instruct | Short finance Q&A (SIP, EMI, tax, budgeting) | ~2,400 |
| Negatives | OTP, KYC, promo, failed UPI → `{}` | ~300 |

Split hygiene: assignment unit is `(bank, template_id)`, not individual SMS. All variants of a template stay in one split. See `ml/data/manifest/SHA256SUMS`.

## Evaluation protocol

Pre-registered before looking at test scores: [`docs/llm-eval-protocol.md`](https://github.com/AnomaProAI/Finndot/blob/main/docs/llm-eval-protocol.md)

### Ship gates

1. SMS R-EM lift vs Qwen3-base ≥ +3 pp, CI excludes 0
2. Amount EM drop vs base ≤ 1 pp
3. Chat groundedness drop vs base ≤ 5 pp
4. JSON validity ≥ 95%
5. On-device TTFT / peak RSS ≤ 1.3× old Qwen2.5

## Limitations

- **Indian bank SMS only** — trained on Indian banking formats (UPI, NEFT, IMPS, etc.). May not generalize to US/EU bank SMS.
- **Merchant extraction** is the hardest field; regex parsers in `parser-core` still win on supported banks.
- **Chat groundedness** is improved in v2 (~68%) but still not perfect; the model can paraphrase numbers.
- **Thinking is disabled** — do not expect chain-of-thought reasoning.
- **1.7B parameter model** — smaller than general-purpose assistants; optimized for the SMS+coach task mix.
- **Untuned bases over-parse** — raw Qwen3 almost always invents a transaction on OTP/promo SMS; do not use the base model for this task without fine-tuning.

## On-device deployment

The production format is **LiteRT-LM INT4** (`.litertlm`, ~0.9 GB). The bf16 safetensors here are the merge before quantization. For the quantized on-device file, see the FinnDot app's model download.

Conversion: `litert-torch export_hf`, `dynamic_int4_block32`, KV 1280, `nothink` prefix.

## Reproduce

```bash
git clone https://github.com/AnomaProAI/Finndot
cd Finndot/ml

# Build dataset
python -m data.build_splits --repo-root .. --out data/out --train-size 16000

# Train (needs 1× A10G / T4 GPU)
python train/sft_qlora.py --config train/train_config.yaml

# Merge adapter
python train/merge_lora.py --adapter <output>/adapter --out <output>/merged

# Evaluate
python -m eval.run_eval \
  --backend hf \
  --sms data/out/test_sms.jsonl \
  --chat eval/fixtures/chat_eval.jsonl \
  --model-id <output>/merged \
  --baseline-id Qwen/Qwen3-1.7B \
  --tag my-eval
```

## Supported banks (35+)

HDFC, SBI, ICICI, Axis, Kotak, PNB, Bank of Baroda, Union Bank, IDFC First, Canara, Federal, Indian Bank, IOB, Central Bank, Karnataka, South Indian, City Union, DBS, HSBC, IDBI, J&K Bank, AMEX, OneCard, Slice, LazyPay, Juspay, Airtel Payments, Jio Payments, IPPB, Jupiter, Utkarsh, Mashreq, and more.

## Citation

```bibtex
@misc{finnai-slm-2026,
  title={FinnAI SLM: On-device Indian bank SMS parsing with QLoRA fine-tuning},
  author={FinnDot Team},
  year={2026},
  url={https://huggingface.co/finndot/finnai-slm-v2},
  note={Apache 2.0. Based on Qwen3-1.7B.}
}
```

## License

Apache 2.0 — same as the [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) base model.
