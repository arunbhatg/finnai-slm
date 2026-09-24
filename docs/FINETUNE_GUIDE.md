# FinnAI SLM — Complete Fine-tuning Guide

> How we built a 1.7B on-device model that parses Indian bank SMS with 92%+ accuracy, and how you can do the same for your domain.

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Why Fine-tune a Small Model?](#2-why-fine-tune-a-small-model)
3. [When to Use This Approach](#3-when-to-use-this-approach)
4. [Architecture Overview](#4-architecture-overview)
5. [Data Pipeline](#5-data-pipeline)
6. [Training Process](#6-training-process)
7. [Evaluation Protocol](#7-evaluation-protocol)
8. [On-Device Deployment](#8-on-device-deployment)
9. [Results](#9-results)
10. [Adapting for Your Use Case](#10-adapting-for-your-use-case)
11. [Lessons Learned](#11-lessons-learned)
12. [Cost Breakdown](#12-cost-breakdown)
13. [FAQ](#13-faq)

---

## 1. The Problem

FinnDot is an Android expense tracker that reads bank SMS to automatically log transactions. Indian banks send SMS like:

```
HDFC Bank: Rs.499.00 debited from A/c XX1234 to SWIGGY via UPI. Avl Bal Rs.15000.50
```

The app needs to extract structured JSON:

```json
{"amount": 499.0, "merchant": "SWIGGY", "type": "EXPENSE", "account": "1234", "balance": 15000.5, "category": "Food"}
```

### Why not just use regex?

We do! FinnDot has 35+ hand-written regex parsers covering major Indian banks. But:

- **Long tail**: New banks, format changes, and regional variations break regex constantly.
- **Indic languages**: Hindi, Tamil, Telugu SMS from the same bank use different patterns.
- **Non-transactions**: OTP, KYC, promo SMS must return `{}` — regex whack-a-mole.
- **Finance coaching**: Users also ask "Where am I spending too much?" — regex can't do that.

An on-device LLM handles all of these as one model, running locally with zero cloud dependency and zero privacy risk.

### Why not use a cloud LLM?

- **Privacy**: Bank SMS contains account numbers, balances, merchants — sensitive data.
- **Latency**: Each SMS needs ~100ms parsing, not 500ms+ cloud round-trips.
- **Cost**: Millions of SMS/day across users would cost thousands in API fees.
- **Offline**: Must work without internet.

## 2. Why Fine-tune a Small Model?

A general-purpose 1.7B model (Qwen3-1.7B) looks weak on **overall** R-EM (~28% on the v2 test) mainly because of empty rows. Restrict to **non-empty transaction rows only** (1,098 SMS):

| Metric (non-empty only) | FinnAI v2 | Qwen3 untuned | Qwen2.5 untuned |
|---|---:|---:|---:|
| Amount EM | **99.5%** | **96.7%** | 96.1% |
| Merchant exact | **98.0%** | 71.6% | 76.3% |
| Strict R-EM | **97.7%** | 32.9% | 40.8% |

So bases already read amounts on real spends. Separately, on **empty** rows (OTP/promo), false-parse is FinnAI 0.5% / Qwen3 **100%** / Qwen2.5 48% — newer untuned models over-help and invent spends. Fine-tuning teaches `{}` there plus full-field JSON on non-empty rows.

| Model | Overall SMS R-EM (full mix) | Params | On-device? |
|---|---|---|---|
| GPT-4o (cloud) | ~95%* | 200B+ | ❌ |
| Qwen3-1.7B (untuned) | 28.16% | 1.7B | ✅ but over-parses empties |
| Qwen2.5-1.5B-Instruct (untuned) | 42.43% | 1.5B | ✅ |
| **FinnAI SLM v2** | **97.97%** | 1.7B | ✅ |

*Estimated, not formally evaluated.

**Fine-tuning closes the gap** between a tiny on-device model and a cloud giant, for a narrow domain. The model learns:
- Indian bank SMS format conventions (UPI, NEFT, IMPS, CC, salary)
- Which fields go where in the JSON schema
- What is NOT a transaction (OTP, promo, KYC → `{}`) — **product-critical**
- Finance coaching grounded in a user's spending ledger

### Why QLoRA specifically?

- **Full fine-tuning** of 1.7B on 16k rows is overkill and risks catastrophic forgetting.
- **LoRA** (Low-Rank Adaptation) trains only ~67MB of adapter weights while keeping the base frozen.
- **QLoRA** quantizes the base to 4-bit during training, fitting on a single A10G (24GB VRAM).
- Training takes ~5 hours and costs ~$8 on AWS.

## 3. When to Use This Approach

### ✅ Good fit

- **Structured extraction from domain-specific text** (bank SMS, invoices, receipts, medical notes)
- **On-device deployment** where privacy matters (health, finance, legal)
- **Narrow task mix** (2-3 tasks, not a general assistant)
- **You own the gold labels** (test cases, templates) — you don't need a human annotation army
- **Low latency** requirements (< 500ms per inference)
- **Languages with limited training data** (Indic, Southeast Asian, African languages)

### ❌ Not ideal

- **General-purpose chatbot** — use a larger model or cloud API
- **Tasks requiring world knowledge** (trivia, current events) — 1.7B can't store enough
- **Very long documents** (> 1000 tokens input) — small models struggle with long context
- **Zero training data** — you need at least 5k-10k quality examples

### Decision flowchart

```
Is user data sensitive? ──Yes──→ On-device required
        │                              │
        No                      Can you get 5k+ labeled examples?
        │                              │
  Cloud API is fine             Yes ──→ Fine-tune a small model ✅
                                No ──→  Few-shot with cloud LLM, then
                                        use outputs to bootstrap training data
```

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FinnDot App (Android)                  │
│                                                          │
│  SMS Inbox ──→ parser-core (35 regex parsers)            │
│       │              │                                   │
│       │         Known bank? ──Yes──→ Parsed transaction  │
│       │              │                                   │
│       │              No                                  │
│       │              ▼                                   │
│       └──────→ FinnAI SLM (LiteRT-LM INT4)               │
│                      │                                   │
│                Extract JSON or {} ──→ Transaction / Skip │
│                                                          │
│  User question ──→ FinnAI SLM (coach mode)                │
│                      │                                   │
│                Grounded answer using ledger context       │
└─────────────────────────────────────────────────────────┘
```

The SLM is a **fallback** — regex parsers handle known banks. The SLM catches:
1. Unknown banks / new SMS formats
2. Indic language SMS
3. Finance coaching questions

### Model stack

| Layer | Technology |
|---|---|
| Base model | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache 2.0) |
| Fine-tuning | QLoRA SFT via [TRL](https://github.com/huggingface/trl) + [PEFT](https://github.com/huggingface/peft) |
| Quantization | NF4 4-bit, double quant, bf16 compute dtype |
| On-device format | LiteRT-LM INT4 `.litertlm` (~0.9 GB) |
| Thinking | **Disabled** — SMS parsing needs speed, not chain-of-thought |

## 5. Data Pipeline

### 5.1 Philosophy: Synthetic-first, privacy-safe

**We never train on real user SMS.** All training data is synthetic:

- **Parser gold**: Extracted from JUnit test cases in `parser-core` (real SMS patterns, synthetic values)
- **Template SMS**: Grammar templates with randomized amounts, merchants, accounts
- **Indic expansion**: Hindi/Hinglish/Tamil/Telugu/Marathi/Bengali paraphrases
- **Coach dialogues**: Synthetic ledgers + grounded Q&A
- **Negatives**: OTP, KYC, promo, failed UPI → `{}`

This means the model + data can be fully open-sourced under Apache 2.0.

### 5.2 SMS data generation

Each SMS example has a seed template:

```python
# From generate_synthetic_sms.py
templates = {
    "upi_debit": "{bank}: Rs.{amount} debited from A/c XX{last4} to {merchant} via UPI. Avl Bal Rs.{balance}",
    "neft_credit": "{bank}: Rs.{amount} credited to A/c XX{last4} via NEFT from {merchant}. Bal: Rs.{balance}",
    "cc_txn": "Alert: {bank} CC XX{last4} used for Rs.{amount} at {merchant} on {date}.",
    # ... 30+ templates covering UPI, NEFT, IMPS, CC, salary, ATM, etc.
}
```

We randomize:
- **Amount**: ₹89 to ₹50,000
- **Merchant**: 30+ real Indian merchants (Swiggy, Zomato, Amazon, Flipkart, BigBasket, etc.)
- **Bank**: All 35+ supported banks with correct sender IDs
- **Account last-4**: Random 4-digit numbers
- **Balance**: Realistic post-transaction balances

Gold labels are deterministic — we know the exact JSON because we generated the SMS.

### 5.3 Indic language expansion

Indian bank SMS often mix languages:

```
SBI: Aapke khate se Rs.1250.00 SWIGGY ko UPI se transfer hua. Shesh rashi: Rs.8750.00
```

We use Amazon Nova Pro as a **data factory** (not a trainer):

1. Feed it an English seed SMS + gold labels
2. Ask for paraphrases in Hindi, Hinglish, Tamil, Telugu, Marathi, Bengali
3. **Verify every output**: amount, merchant, last-4, balance must appear in the paraphrase
4. Reject any row that fails verification

This produced 300 verified Indic SMS and 150 translated coach dialogues.

> **Important**: Nova generates surface text only. Gold labels stay ours. The model (Qwen3) is trained by QLoRA — Nova is not the trainer.

### 5.4 Coach data generation

Each coach example has a synthetic ledger:

```python
ledger = {
    "id": "led_001",
    "total_spend": 18240.00,
    "income": 65000.00,
    "savings": 46760.00,
    "categories": {
        "Food": {"amount": 7200.00, "top": "SWIGGY", "count": 8},
        "Subscriptions": {"amount": 1497.00, "top": "NETFLIX", "count": 2},
        "Shopping": {"amount": 1999.00, "top": "AMAZON", "count": 1},
        # ...
    }
}
```

The assistant answer must only use numbers from this ledger (groundedness). No hallucinated figures.

### 5.5 Split hygiene

**Critical**: The split unit is `(bank, template_id)`, not individual SMS. If HDFC UPI template #3 generates 20 amount variants, all 20 go to the same split. This prevents the test set from being leaked paraphrases of training data.

```
Seed 42 → SHA-256 hash → deterministic 80/10/10 split
```

### 5.6 Final dataset (v2)

| Split | Total | SMS | Coach | General |
|---|---|---|---|---|
| Train | 16,000 | 9,600 (60%) | 4,000 (25%) | 2,400 (15%) |
| Val | 1,561 | 1,254 | 307 | 0 |
| Test | 1,592 | 1,282 | 160 | 150 |

Hashes: `ml/data/manifest/SHA256SUMS`

## 6. Training Process

### 6.1 QLoRA configuration

```yaml
# ml/train/train_config.yaml
base_model: Qwen/Qwen3-1.7B
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bfloat16

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8  # effective batch = 16
  learning_rate: 2e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  max_seq_length: 1536
  bf16: true
  seed: 42
```

### 6.2 Why these choices?

- **Rank 16**: Enough to learn JSON formatting without huge adapter files. Rank 8 was too low for merchant extraction; rank 32 showed no improvement.
- **All attention + MLP targets**: SMS parsing benefits from modifying both attention (which tokens to attend to) and MLP (what to output). Attention-only LoRA dropped R-EM by ~5%.
- **3 epochs**: With 16k examples, 3 epochs = 3000 steps. More epochs overfitted templates (val R-EM dipped at epoch 3 in v1).
- **Effective batch 16**: Stable JSON generation. Smaller batches led to inconsistent formatting.
- **Thinking disabled**: We inject `<think>\n\n</think>` prefix so the model skips reasoning and outputs JSON directly. On-device, thinking wastes 2-3 seconds.
- **No packing**: SMS examples are short (~200 tokens). Packing would glue unrelated bank SMS together, confusing the model about where one transaction ends and another begins.

### 6.3 Checkpoint selection

We do NOT pick the lowest training loss. Instead:

After each epoch, we run **strict Record Exact-Match (R-EM)** on 256 validation SMS. The adapter with the highest val R-EM ships.

```
v2 results:
  epoch 1: val R-EM = 0.965 (step 1000)
  epoch 2: val R-EM = 0.965 (step 2000)
  epoch 3: val R-EM = 0.965 (step 3000)  ← all tied, rock-stable
```

### 6.4 Hardware

| Resource | Spec | Cost |
|---|---|---|
| GPU | 1× NVIDIA A10G 24GB (AWS g5.2xlarge) | ~$1.46/hr |
| Region | us-east-2 (Ohio) | — |
| Training time | ~5 hours | ~$7.30 |
| VRAM usage | ~18 GB (4-bit base + LoRA + optimizer) | — |
| Disk | 150 GB gp3 | ~$0.50 |

**Total training cost: ~$8**

A T4 (16GB) also works but is ~30% slower.

### 6.5 Running training

```bash
cd ml

# 1. Build dataset
python -m data.build_splits --repo-root .. --out data/out --train-size 16000

# 2. (Optional) Generate Indic expansion
python -m data.generate_nova_indic --sms 300 --chat 150 --out data/out/nova_indic.jsonl
python -m data.build_splits --repo-root .. --out data/out \
  --extra data/out/nova_indic.jsonl --train-size 16000

# 3. Train
export SM_CHANNEL_TRAIN=data/out
export SM_MODEL_DIR=train/output
python train/sft_qlora.py --config train/train_config.yaml

# 4. Merge adapter → bf16
python train/merge_lora.py --adapter train/output/adapter --out train/output/merged
```

## 7. Evaluation Protocol

Our eval protocol is **pre-registered** — thresholds are frozen before looking at test scores. This prevents p-hacking and ensures honest reporting.

Full protocol: [`docs/llm-eval-protocol.md`](https://github.com/AnomaProAI/Finndot/blob/main/docs/llm-eval-protocol.md)

### 7.1 SMS metrics

| Metric | What it measures |
|---|---|
| **Strict R-EM** (primary) | All 5 fields correct: amount + type + merchant + account + balance |
| Amount EM | Numeric match after stripping ₹/Rs/commas |
| Merchant exact | Case-insensitive, whitespace-normalized |
| Field micro-F1 | Per-field binary match averaged across all fields |
| JSON validity | Output parses as valid JSON |
| False-parse rate | Non-transactions incorrectly parsed as transactions |

### 7.2 Chat metrics

| Metric | What it measures |
|---|---|
| **Groundedness** | Every number in the reply appears in the system prompt |
| No-markdown | No `*`, `#`, or code fences (plain text for mobile) |
| Length band | Response is 40–700 characters |

### 7.3 Ship gates

All must pass on the held-out test set:

1. SMS R-EM: FinnAI − Qwen3-base ≥ **+3 pp**, CI excludes 0
2. Amount EM: FinnAI − Qwen3-base ≥ **−1 pp**
3. Chat groundedness: FinnAI − Qwen3-base ≥ **−5 pp**
4. JSON validity ≥ **95%**
5. On-device TTFT / RSS ≤ **1.3×** old model

Plus: FinnAI R-EM must be ≥ Qwen2.5-1.5B-Instruct R-EM (product regression check).

### 7.4 Running evaluation

```bash
python -m eval.run_eval \
  --backend hf \
  --sms data/out/test_sms.jsonl \
  --chat eval/fixtures/chat_eval.jsonl \
  --model-id train/output/merged \
  --baseline-id Qwen/Qwen3-1.7B \
  --product-id Qwen/Qwen2.5-1.5B-Instruct \
  --tag my-experiment
```

Output: `eval/reports/my-experiment.md` + `my-experiment_summary.json`

## 8. On-Device Deployment

### 8.1 Conversion pipeline

```
bf16 merged safetensors
    → litert-torch export_hf
    → dynamic_int4_block32
    → KV cache 1280
    → nothink prefix (<think>\n\n</think>)
    → FinnAI-SLM.litertlm (~0.9 GB)
```

### 8.2 Android integration

The `.litertlm` file is hosted on CloudFront and downloaded on-demand when the user enables the local AI model. The app uses LiteRT-LM runtime for inference.

Key settings:
- **Temperature**: 0 (greedy decode for deterministic JSON)
- **Max tokens**: 256 (SMS), 512 (chat)
- **Stop token**: EOS + `}` for SMS (stops after complete JSON)

### 8.3 Performance targets

| Metric | Target | Typical |
|---|---|---|
| Model size | < 1 GB | ~0.9 GB |
| TTFT (warm) | < 500ms | ~300ms on Snapdragon 7-series |
| Decode speed | > 10 tok/s | ~15 tok/s |
| Peak RAM | < 2 GB | ~1.5 GB |

## 9. Results

### v1 (12k training, English-heavy)

| Metric | FinnAI v1 | Qwen3-1.7B base | Qwen2.5-1.5B |
|---|---|---|---|
| SMS R-EM % | **92.75** [91.3, 94.1] | 22.13 | 48.78 |
| Amount EM % | **94.60** | 79.27 | 88.08 |
| JSON valid % | **100.0** | 100.0 | — |
| Merchant exact % | **93.12** | 54.48 | — |
| Chat ground. % | 32.0 | 26.0 | 68.0 |
| McNemar p | 1.3e-287 | — | — |

**SHIP: YES** — all gates pass.

### v2 (16k training, + Indic expansion) — **SHIP: YES**

| Metric | FinnAI v2 | FinnAI v1 | Δ |
|---|---|---|---|
| SMS R-EM % | **97.97** | 92.75 | **+5.22** |
| Amount EM % | **99.53** | 94.60 | +4.93 |
| Merchant exact % | **98.21** | 93.12 | +5.09 |
| False-parse % | **0.54** | 28.57 | **−28.03** |
| Chat groundedness % | **68.0** | 32.0 | **+36.0** |

Val R-EM: **0.965** (stable across all 3 epochs). Nova Indic expansion drove the chat and false-parse wins.

## 10. Adapting for Your Use Case

### 10.1 Similar domains where this works

| Domain | Input | Output | Why it fits |
|---|---|---|---|
| **Invoice parsing** | Scanned invoice text | `{vendor, amount, date, tax}` | Structured extraction from semi-structured text |
| **Medical notes** | Clinical note | `{diagnosis, medication, dosage}` | Privacy-critical, on-device |
| **Receipt scanning** | OCR receipt text | `{store, items, total, date}` | Known formats, structured output |
| **Email classification** | Email body | `{category, priority, action}` | Domain-specific classification |
| **Legal document extraction** | Contract clause | `{party, obligation, date}` | Sensitive data, structured output |

### 10.2 Step-by-step adaptation

**Step 1: Define your JSON schema**

```json
{"field1": "...", "field2": 0.0, "field3": "ENUM_VALUE"}
```

Keep it flat. Nested objects confuse small models.

**Step 2: Write 20-30 seed templates**

Cover your main input patterns. For us, that was UPI debit, NEFT credit, CC transaction, salary, ATM, etc.

**Step 3: Generate 5k-15k synthetic examples**

Randomize the variable parts (amounts, names, dates). Your templates are the gold labels.

**Step 4: Add negatives**

Inputs that should return `{}`. For us: OTP, promo, KYC messages. For invoices: marketing emails.

**Step 5: Add Indic / multilingual (optional)**

Use a cloud LLM as a data factory to paraphrase into target languages. **Always verify** that gold labels survive the paraphrase.

**Step 6: Train with QLoRA**

Use our `train_config.yaml` as a starting point. Adjust:
- `max_seq_length`: Match your longest input + output
- `num_train_epochs`: 2-4 for 10k-20k examples
- `per_device_train_batch_size`: Reduce if OOM

**Step 7: Evaluate with pre-registered gates**

Define your accuracy threshold BEFORE looking at results. This prevents fooling yourself.

### 10.3 Common pitfalls

| Pitfall | Solution |
|---|---|
| Test set is paraphrases of training data | Split by template group, not individual examples |
| Model outputs valid JSON but wrong values | Use strict R-EM (all fields must match), not just JSON validity |
| Works in English, fails in Hindi | Add Indic training data; verify with exact-match, not vibes |
| Great on val, bad in production | Your templates may not cover real-world diversity; add more patterns |
| Model hallucinates numbers in coach mode | Enforce groundedness: every number must come from the prompt |
| Training loss drops but R-EM stalls | R-EM is a hard metric; the model may be getting better at formatting but worse at extraction |

## 11. Lessons Learned

### What worked

1. **Synthetic-first data**: No annotation team needed. Templates + randomization = infinite diversity.
2. **Pre-registered eval**: We defined ship gates before seeing scores. This prevented "oh it's close enough" rationalization.
3. **QLoRA over full FT**: 67MB adapter, not 3.4GB full weights. Trains in 5 hours on one GPU.
4. **Thinking disabled**: 2-3 second speedup on device with no accuracy loss for this task.
5. **Multi-task training**: 60% SMS + 25% coach + 15% general prevented the model from forgetting how to chat.
6. **Checkpoint selection by R-EM, not loss**: Training loss is a poor proxy for JSON extraction quality.

### What didn't work (or surprised us)

1. **Chat groundedness is hard**: 32% is mediocre. The model paraphrases numbers instead of citing exact figures. More grounding data needed.
2. **PNB/SBI SMS rejected by verification**: Some banks use unusual formatting that Nova couldn't paraphrase while preserving exact tokens. Manual templates work better.
3. **AWS GPU quotas**: Mumbai (ap-south-1) had zero G-instance quota. We trained in Ohio. Budget for quota surprises.
4. **Throttling on data generation**: Nova Pro rate limits at ~1 req/sec. Budget 20-30 minutes for 500 rows, not 5 minutes.

## 12. Cost Breakdown

### One training run (v2)

| Item | Cost |
|---|---|
| EC2 g5.2xlarge × 5 hours (training) | $7.30 |
| EC2 g5.2xlarge × 4 hours (eval) | $5.84 |
| Nova Pro API (~800 calls for Indic data) | ~$2.00 |
| S3 storage (~5 GB) | $0.12/month |
| Data transfer | ~$0.50 |
| **Total** | **~$16** |

### Full project (v1 + v2 + debugging)

| Item | Cost |
|---|---|
| Training runs (v1 + v2) | ~$15 |
| Eval runs (v1 + v2) | ~$12 |
| Failed/debugging EC2 instances | ~$10 |
| Nova Pro data generation | ~$3 |
| S3 + CloudFront | ~$2 |
| **Total project** | **~$42** |

A 92%+ accuracy on-device model for under $50. No annotation team. No cloud dependency.

## 13. FAQ

**Q: Can I use this model directly for my app?**
A: If your use case is Indian bank SMS parsing, yes! Download from Hugging Face and convert to your target format. For other domains, you'll need to fine-tune on your own data.

**Q: Why Qwen3-1.7B and not Llama/Gemma/Phi?**
A: Apache 2.0 license (no restrictions on commercial use), good multilingual support (important for Indic languages), and LiteRT-LM export path exists. Gemma and Phi are also good choices.

**Q: Can I fine-tune further on top of FinnAI SLM?**
A: Yes! Load our merged checkpoint and run QLoRA again with your data. This is great for adding new banks or languages.

**Q: Why not distill from GPT-4 / Claude?**
A: License restrictions. OpenAI's and Anthropic's terms prohibit using model outputs to train competing models. We use only synthetic data we own.

**Q: How much data do I need?**
A: We got 92%+ with 16k examples. 5k-8k is usually enough for a focused extraction task. Quality > quantity.

**Q: Does this work for non-Indian banks?**
A: The model was trained on Indian bank SMS patterns. It may partially work for other countries' banking SMS, but you'd get better results fine-tuning on your target country's formats.

**Q: Can I contribute new bank parsers?**
A: Yes! See `parser-core/` in the FinnDot repo. Add a JUnit test with real SMS patterns (synthetic values), and our data pipeline will automatically include them in the next training run.

---

## References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [Qwen3 Technical Report](https://huggingface.co/Qwen/Qwen3-1.7B) (Qwen Team, 2025)
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [FinnDot App](https://github.com/AnomaProAI/Finndot)

---

*This document is part of the FinnAI SLM open-source release. Apache 2.0 license.*
