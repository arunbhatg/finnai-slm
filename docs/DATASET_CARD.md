---
language:
- en
- hi
- ta
- te
- mr
- bn
license: apache-2.0
task_categories:
- text-generation
- token-classification
tags:
- finance
- sms-parsing
- indian-banking
- synthetic
- expense-tracker
- finndot
size_categories:
- 10K<n<100K
---

# FinnAI SLM Training Data

Synthetic training data for fine-tuning on-device models to parse Indian bank SMS into structured JSON. Created for the [FinnDot](https://github.com/AnomaProAI/Finndot) expense tracker.

## Key facts

- **100% synthetic** — no real user SMS, no real financial data
- **Privacy-safe** — can be freely shared, no PII
- **Apache 2.0** — use for any purpose including commercial
- **Multi-language** — English, Hindi, Hinglish + seed templates for Tamil, Telugu, Marathi, Bengali
- **Multi-task** — SMS extraction (60%), finance coaching (25%), general finance Q&A (15%)

## Dataset structure

| Split | Examples | SMS | Coach | General |
|---|---|---|---|---|
| train | 16,000 | 9,600 | 4,000 | 2,400 |
| val | 1,561 | 1,254 | 307 | 0 |
| test | 1,592 | 1,282 | 160 | 150 |

### SMS example

```json
{
  "id": "a1b2c3d4e5f6g7h8",
  "task": "sms",
  "bank": "HDFC",
  "template_id": "upi_debit/HDFC/SWIGGY/0",
  "split_key": "HDFC|upi_debit/HDFC/SWIGGY/0",
  "messages": [
    {"role": "system", "content": "Extract transaction details..."},
    {"role": "user", "content": "HDFC Bank: Rs.499.00 debited from A/c XX1234 to SWIGGY via UPI. Avl Bal Rs.15000.50"},
    {"role": "assistant", "content": "{\"amount\": 499.0, \"merchant\": \"SWIGGY\", \"type\": \"EXPENSE\", \"account\": \"1234\", \"balance\": 15000.5, \"category\": \"Food\"}"}
  ],
  "gold": {"amount": 499.0, "merchant": "SWIGGY", "type": "EXPENSE", "account": "1234", "balance": 15000.5, "category": "Food"}
}
```

### Coach example

```json
{
  "id": "b2c3d4e5f6g7h8i9",
  "task": "chat",
  "bank": "COACH",
  "template_id": "coach/leak/led_001",
  "messages": [
    {"role": "system", "content": "You are a personal finance coach. The user's spending this month: Total: ₹18,240..."},
    {"role": "user", "content": "Where am I spending the most?"},
    {"role": "assistant", "content": "You spent ₹7200.00 on Food — that is 39% of ₹18240.00 this month..."}
  ],
  "must_ground": ["18240.00", "7200.00", "65000.00"]
}
```

### Negative example (non-transaction)

```json
{
  "id": "c3d4e5f6g7h8i9j0",
  "task": "sms",
  "bank": "HDFC",
  "template_id": "neg/otp/HDFC",
  "messages": [
    {"role": "system", "content": "Extract transaction details..."},
    {"role": "user", "content": "123456 is your OTP for HDFC Bank NetBanking. Valid for 5 min. Do not share."},
    {"role": "assistant", "content": "{}"}
  ],
  "gold": {}
}
```

## Data sources

| Source | Generator script | Description |
|---|---|---|
| Parser gold | `ml/data/extract_parser_gold.py` | JUnit test cases from 35+ bank parsers |
| Synthetic SMS | `ml/data/generate_synthetic_sms.py` | Template-based UPI/NEFT/IMPS/CC/salary/ATM |
| Finance coach | `ml/data/generate_finance_chat.py` | Grounded Q&A with synthetic ledgers |
| General instruct | `ml/data/generate_general_instruct.py` | Short finance Q&A (SIP, EMI, tax) |

> **Note**: The published dataset excludes Nova Pro Indic expansions (per AWS service terms). To regenerate Indic data locally, run `python -m data.generate_nova_indic` with your own Bedrock access.

## Split methodology

- Assignment unit: `(bank, template_id)` — all variants of a template stay in one split
- Hash: `SHA-256(seed=42 + split_key)`
- Ratios: ~80% train / 10% val / 10% test
- Train mix: 60% SMS, 25% coach, 15% general (with upsampling)
- Chat eval: Frozen 50-item set, never in training

## Supported banks

HDFC, SBI, ICICI, Axis, Kotak, PNB, Bank of Baroda, Union Bank, IDFC First, Canara, Federal, Indian Bank, IOB, Central Bank, Karnataka, South Indian, City Union, DBS, HSBC, IDBI, J&K Bank, AMEX, OneCard, Slice, LazyPay, Juspay, Airtel Payments, Jio Payments, IPPB, Jupiter, Utkarsh, Mashreq, and more.

## SMS types covered

| Type | Examples |
|---|---|
| UPI debit/credit | `Rs.499 debited to SWIGGY via UPI` |
| NEFT/IMPS | `Rs.15000 credited via NEFT from SALARY` |
| Credit card | `CC XX1234 used for Rs.2999 at AMAZON` |
| Salary | `Salary Rs.65000 credited to A/c` |
| ATM withdrawal | `Rs.5000 withdrawn from ATM` |
| Balance enquiry | `Avl Bal in A/c XX1234: Rs.25000` |
| OTP (negative) | `123456 is your OTP` → `{}` |
| Promo (negative) | `Exclusive offer! Get 10% cashback` → `{}` |
| KYC (negative) | `Complete your KYC by visiting...` → `{}` |
| Failed UPI (negative) | `UPI txn failed for Rs.500` → `{}` |

## How to use

```python
from datasets import load_dataset

ds = load_dataset("finndot/finnai-slm-data")

# Training
for row in ds["train"]:
    messages = row["messages"]
    # Use with any chat-template trainer (TRL SFTTrainer, axolotl, etc.)
```

## Reproduce from source

```bash
git clone https://github.com/AnomaProAI/Finndot
cd Finndot/ml
python -m data.build_splits --repo-root .. --out data/out --train-size 16000
# Verify: sha256sum data/out/*.jsonl  → compare with ml/data/manifest/SHA256SUMS
```

## Integrity

SHA-256 hashes of all split files are in `ml/data/manifest/SHA256SUMS` in the FinnDot repo. Verify before training.

## Citation

```bibtex
@misc{finnai-slm-data-2026,
  title={FinnAI SLM Training Data: Synthetic Indian Bank SMS for On-Device Extraction},
  author={FinnDot Team},
  year={2026},
  url={https://huggingface.co/datasets/finndot/finnai-slm-data}
}
```

## License

Apache 2.0
