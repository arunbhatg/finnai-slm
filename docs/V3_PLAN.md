# FinnAI SLM v3 — India usefulness plan

**Status:** implementation in progress (data + eval + train config landed; GPU train/export pending)  
**Ship target:** stay on `Qwen/Qwen3-1.7B` + QLoRA → LiteRT INT4 (same on-device envelope as v2)  
**Not in scope for ship:** larger base model (4B+) — optional offline ablation only

## How to build v3 data (this repo)

```bash
cd ml
bash scripts/build_v3_data.sh
# optional Bedrock Indic expansion:
# bash scripts/build_v3_data.sh --with-nova
```

Outputs land in `ml/data/out_v3/` plus eval fixtures:
- `ml/eval/fixtures/india_flows_eval.jsonl`
- `ml/eval/fixtures/indic_sms_eval.jsonl`

Train (1× A10G / T4):

```bash
cd ml
export SM_CHANNEL_TRAIN=$PWD/data/out_v3
export SM_MODEL_DIR=$PWD/train/output_v3
python train/sft_qlora.py --config train/train_config.yaml
python train/merge_lora.py --adapter train/output_v3/adapter --out train/output_v3/merged
```

Eval with v3 gates (absolute chat groundedness ≥ 85%):

```bash
python -m eval.run_eval \
  --backend hf \
  --version v3 \
  --sms data/out_v3/test_sms.jsonl \
  --chat eval/fixtures/chat_eval.jsonl \
  --sms-slice eval/fixtures/indic_sms_eval.jsonl \
  --sms-slice eval/fixtures/india_flows_eval.jsonl \
  --model-id train/output_v3/merged \
  --baseline-id Qwen/Qwen3-1.7B \
  --tag v3-run
```

## Code landed

| Path | Role |
| --- | --- |
| `ml/data/generate_india_flows.py` | BBPS, EMI, wallets, MF SIP, refunds, CC payment/statement |
| `ml/data/generate_indic_sms_manual.py` | HI/Hinglish/TA/TE/MR/BN templates without Bedrock |
| `ml/data/generate_finance_chat.py` | More ledgers, Hinglish asks, hard grounding refusals |
| `ml/data/build_splits.py` | `--version v3`, 28k train, 55/30/15 mix |
| `ml/eval/run_eval.py` | `--version v3` gate3b + `--sms-slice` |
| `ml/train/train_config.yaml` | `FinnAI-SLM-v3`, larger val R-EM sample |

## Why not a bigger model first?

v2 already hits **~98% SMS R-EM** and **~99.5% amount EM** on held-out synthetic transactions. Untuned Qwen3 already reads rupee amounts well; fine-tuning mainly taught full-field JSON + `{}` refusal. The product gaps that hurt Indian users are elsewhere:

| Gap (v2) | Why it matters in India | Leverage |
| --- | --- | --- |
| Chat groundedness **68%** | Ask Finn can paraphrase / invent ₹ figures | More ledger-grounded coach data (EN + Hinglish + HI) |
| Indic SMS ~**300** Nova rows | TA/TE/MR/BN bank SMS still thin vs EN/HI | Scale verified Indic paraphrases + manual templates where Nova fails (PNB/SBI) |
| Template taxonomy English-heavy | Real inboxes have BBPS, EMI, wallets, MF, refunds, CC statements | New synthetic generators |
| Gate 5 pending | Mid-range phones must stay ≤1.3× prior TTFT/RSS | Finish on-device bench before calling v3 “shipped” |

**Verdict:** better **data + eval gates**, not a bigger base. A 4B model would grow the ~974 MB download and hurt mid-range Android TTFT with little SMS gain.

## v3 goals (measurable)

Keep v2 SMS quality; raise India UX.

| Gate | v2 | v3 target |
| --- | ---: | ---: |
| SMS strict R-EM (full mix) | 97.97% | ≥ **97.5%** (non-regress) |
| False-parse (empty rows) | 0.54% | ≤ **1.0%** |
| Chat groundedness (frozen set) | 68% | ≥ **85%** |
| Indic SMS R-EM (new held-out slice, ≥200 rows TA/TE/MR/BN/HI) | n/a | ≥ **95%** |
| New-flow SMS R-EM (BBPS/EMI/wallet/MF/refund slice) | n/a | ≥ **94%** |
| On-device gate 5 (TTFT/RSS ≤1.3×) | pending | **pass** |
| INT4 vs bf16 amount EM drop | not gated | ≤ **1 pp** |

## Data recipe

Grow train from **16k → ~28k** (same mix philosophy, richer India content).

| Bucket | v2 | v3 | Notes |
| --- | ---: | ---: | --- |
| SMS (EN templates + parser gold) | ~9.6k | ~12k | Keep; add BBPS, EMI mandate, UPI refund, wallet (PhonePe/Amazon Pay/Paytm), MF SIP, CC statement/payment |
| Indic SMS (verified Nova + manual) | ~300 | **~2.5k** | Cap per `(bank, lang)`; reject if amount/merchant/last-4 drift |
| Coach (ledger-grounded) | ~4k | **~9k** | Hard negatives: empty ledger, wrong-month asks, Hinglish “kitna kharch”; every assistant ₹ must be in `must_ground` |
| General finance India | ~2.4k | ~3.5k | SIP, Section 80C, UPI limits, NPCI rails — short answers only |
| Negatives → `{}` | ~300 | ~800 | e-mandate OTP, KYC, promo, failed UPI, balance enquiry |

**Still no real user SMS** in training (privacy / Apache 2.0). Gold stays deterministic; Nova (or local paraphraser) only rewrites *surface* text.

### Mix ratios (train)

- 55% SMS (incl. Indic + new flows)
- 30% coach (up from 25% — groundedness is the main UX gap)
- 15% general

Split unit remains `(bank, template_id)` so paraphrases never leak across train/val/test.

## Code changes (this repo)

1. **`ml/data/generate_india_flows.py`** — BBPS / EMI / wallet / MF SIP / refund / CC templates with Indian merchants and ₹ Indian-comma amounts.
2. **`ml/data/generate_nova_indic.py`** — raise defaults (`--sms 2000 --chat 800`); stronger verify; manual seed list for PNB/SBI.
3. **`ml/data/generate_finance_chat.py`** — more Hinglish/HI asks; adversarial groundedness (refuse numbers not in ledger).
4. **`ml/data/build_splits.py`** — `--train-size 28000`, new mix weights.
5. **`ml/eval/`** — add `fixtures/indic_sms_eval.jsonl`, `fixtures/india_flows_eval.jsonl`; raise chat groundedness ship gate to ≥85% vs untuned base; report Indic / new-flow slices in `eval-v3-report.md`.
6. **`ml/train/train_config.yaml`** — keep LoRA r=16; bump `val_rem_max_items` if needed; same export (`dynamic_int4_block32`, `nothink`).
7. **Publish** — `finndot/finnai-slm-v3` on HF + CloudFront `.litertlm` when gates pass.

## Optional ablation (offline only)

Run one QLoRA on **Qwen3-4B** with the same v3 data; compare coach groundedness and on-device TTFT on a Pixel/Snapdragon mid-range. **Ship stays 1.7B** unless 4B wins groundedness by ≥8 pp *and* gate 5 still passes (unlikely on mid-range).

## What we will not do in v3

- Distill from GPT/Claude (license / OSS story)
- Train on raw user SMS
- Replace `parser-core` regex for known banks (hybrid stays: regex first, SLM long-tail + coach)
- Enable Qwen “thinking” on-device (latency)

## Execution order

1. Generators + expanded Nova Indic (data)
2. Build 28k splits; train QLoRA on 1× A10G
3. Eval gates + Indic/new-flow slices; iterate data if groundedness &lt; 85%
4. LiteRT export + gate 5 on-device bench
5. HF + CloudFront publish; point app `MODEL_URL` at v3

## Success for Indian users

- Ask Finn answers stay on the user’s ₹ numbers (Hinglish OK)
- Hindi/Tamil/Telugu/Marathi/Bengali bank SMS parse without inventing spends
- Common India rails beyond plain UPI debit (BBPS, EMI, wallets, SIP) work in the long-tail path
- Phone download size and speed stay in the v2 ballpark
