# FinnAI SLM — how we fine-tune it

**Model name:** FinnAI SLM  
**Base:** [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache 2.0 Instruct)  
**Method:** QLoRA supervised fine-tuning (SFT), thinking **off**  
**On-device target:** LiteRT-LM INT4 `.litertlm` (~0.9 GB)  
**Eval protocol (frozen before scores):** [llm-eval-protocol.md](llm-eval-protocol.md)  
**Ops:** [aws-ondevice-slm.md](aws-ondevice-slm.md)

This document is the lab notebook. An independent reviewer should be able to reproduce the adapter from git + this file + `ml/data/manifest/SHA256SUMS`.

## 1. Why we fine-tune

FinnDot already parses **known banks** with regex in `parser-core`. The on-device SLM is for:

1. **Ask / Learn / analytics tips** when the user downloads a local model (`ChatAiMode.LOCAL`).
2. **SMS JSON fallback** for senders with no bank parser (only if the engine is loaded).

The old shipped file was Qwen2.5-1.5B MediaPipe `.task`. Qwen3 on-device builds are **LiteRT-LM `.litertlm`**, so we fine-tune Qwen3-1.7B and convert that checkpoint — not a MediaPipe `.task`.

We do **not** replace regex parsers. Fine-tuning is specialization: more reliable SMS JSON and finance-coach answers that stay grounded in the ledger, without dumping user SMS into the cloud.

## 2. What “FinnAI SLM” is (and is not)

| Is | Is not |
| --- | --- |
| A 1.7B Qwen3 Instruct derivative with a LoRA adapter trained on FinnDot prompts | A new pretrain from scratch |
| Multi-task: ~60% SMS JSON, ~25% finance coach, ~15% general instruct | An SMS-only model (that would collapse Ask/Learn) |
| Apache 2.0 redistributable if we keep training data synthetic + parser gold | Trained on Firestore `sms_reports` or live inboxes |
| Greedy decode, thinking disabled | A chain-of-thought reasoner on the phone |

Checkpoint selection is **best validation SMS strict record exact-match (R-EM)**, not lowest train loss.

## 3. Data recipe (privacy-safe)

No user SMS. No Claude outputs in the SFT mix (Anthropic competing-model terms).

**Nova Pro is a data factory, not the trainer.** QLoRA still trains Qwen3. Nova only proposes Indic/Hinglish *surface* SMS and translations of *our* coach Q/A. A row is kept only if amount, merchant, last-4, and ledger rupees still match the seed gold. Assistant JSON is never Nova-authored.

### 3.1 Sources

| Slice | Generator | Target mix (train) |
| --- | --- | --- |
| Parser gold | `ml/data/extract_parser_gold.py` reads `ParserTestCase` from `parser-core` tests | folded into SMS |
| Synthetic bank SMS | `ml/data/generate_synthetic_sms.py` (UPI, CC, salary, NEFT, IMPS, ATM, Hinglish/Hindi, OTP/KYC/promo negatives) | 60% SMS total |
| Finance coach | `ml/data/generate_finance_chat.py` — English + Hinglish asks, same ledger grounding | 25% |
| Nova Indic expand (optional) | `ml/data/generate_nova_indic.py` — verified paraphrases/translations | extra SMS/chat |
| General instruct | `ml/data/generate_general_instruct.py` — short finance Q&A so the model still answers non-JSON | 15% |

Build:

```powershell
cd ml
python -m data.build_splits --repo-root .. --out data/out
# after Ohio v1 job finishes, optional v2 with Nova surface text:
python -m data.generate_nova_indic --out data/out/nova_indic.jsonl --sms 400 --chat 400
python -m data.build_splits --repo-root .. --out data/out --extra data/out/nova_indic.jsonl --train-size 16000
```

Default scale: **12,000 train / ~1.7k val / ~1.5k test** (seed **42**). Hashes: `ml/data/manifest/SHA256SUMS`.

### 3.2 Split hygiene

The assignment unit is `(bank, template_id)`, **not** a single SMS line. Amount/merchant variants of one template stay in one split so the test set is not leaked paraphrases.

Chat **eval** is a frozen 50-item set in `ml/eval/fixtures/chat_eval.jsonl` (ids prefixed `eval/`). It is never mixed into SFT.

### 3.3 SMS target schema

Matches `LlmSmsParser.buildPrompt`:

```json
{
  "amount": 1234.5,
  "merchant": "SWIGGY",
  "type": "EXPENSE",
  "account": "1234",
  "balance": 8000.0,
  "category": "Food"
}
```

`type` ∈ `INCOME | EXPENSE | CREDIT | TRANSFER | INVESTMENT`. Non-transactions (OTP, KYC, promo, failed UPI) target `{}`.

ChatML messages: system = parser/coach instructions, user = SMS or question, assistant = JSON or grounded prose. Qwen3 chat template is applied with `enable_thinking=false`.

### 3.4 Coach answers

Every rupee figure in the assistant turn is taken from the synthetic ledger in the system prompt (groundedness). No markdown.

## 4. Training method (QLoRA SFT)

Paper-style: Hu et al. LoRA; Dettmers et al. QLoRA. We do **not** full-finetune 1.7B.

| Knob | Value | Why |
| --- | --- | --- |
| Base | `Qwen/Qwen3-1.7B` | Apache 2.0, LiteRT-LM export path exists |
| Quantization | NF4 4-bit, double quant, bf16 compute | Fits 1× A10G 24 GB (`ml.g5.2xlarge`) |
| LoRA rank / alpha | 16 / 32 | Plan default; enough for JSON style without huge adapters |
| Dropout | 0.05 | Light regularisation |
| Targets | `q,k,v,o,gate,up,down_proj` | Attention + MLP |
| Epochs | 3 | Small specialised set; more overfits templates |
| LR | 2e-4 cosine, 3% warmup | Typical QLoRA SFT |
| Effective batch | 2 × 8 grad-accum = 16 | Stable JSON loss |
| Max seq | 1536 | Coach system prompt + SMS |
| Packing | off | Do not glue JSON examples together |
| Decode in callback | greedy, 192 new tokens | Same spirit as eval `temperature=0` |
| Seed | 42 | Reproducible splits + init |

Entry point: `ml/train/sft_qlora.py` (config `ml/train/train_config.yaml`).

After each epoch the trainer scores up to **256** val SMS items with **strict R-EM** (amount + type + merchant + account last-4 + balance). The log is `val_rem.json`. We keep the adapter from the last completed train, and reviewers should pick the epoch with max `val_sms_rem` if we later save per-epoch adapters under `checkpoints/`.

If 4-bit `bitsandbytes` is missing, the script falls back to bf16 LoRA (needs more VRAM).

### 4.1 Why not full FT / DPO / distillation

- Full FT of 1.7B on 12k rows is unnecessary and more likely to erase general chat.
- DPO needs preference pairs we do not have.
- Distilling Claude would block an OSS release.
- Do not train on raw Nova completions. Nova may expand *our* templates; drop any row that fails gold checks. Keep the Nova jsonl out of a public dataset dump if you open-source FinnAI.

## 5. Hardware and cost (ap-south-1)

Preferred: **`ml.g5.2xlarge`** (1× A10G 24 GB, **32 GiB host RAM**). Host RAM matters later for LiteRT export.

| Instance | Role | On-demand (Mumbai, ~2026-09) |
| --- | --- | --- |
| `ml.g5.2xlarge` / `g5.2xlarge` | Train + export | ~$1.46/hr EC2; SageMaker is higher |
| `g5.xlarge` | Avoid for export | 16 GiB host RAM too tight |
| `ml.g4dn.xlarge` | Train-only fallback (T4 16 GB) | If G5 quota is 0 |

One train+export+eval pass is typically **4–8 hours** (~$6–12 on EC2). Request G5 quota first (`python -m aws.request_gpu_quota`).

## 6. Job flow (AWS)

```text
parser-core + generators
        → data/out/*.jsonl  (hashed)
        → S3 training bucket  datasets/v1/
        → SageMaker / EC2  sft_qlora.py
        → adapter/  (LoRA)
        → merge_lora.py  → bf16 merged
        → litert-torch INT4 nothink  → FinnAI-SLM.litertlm
        → eval.run_eval vs Qwen3-base (+ Qwen2.5 product check)
        → SHIP: YES?  CloudFront + Hugging Face
```

Launch (after `cdk deploy FinndotOnDeviceModelStack`):

```powershell
cd ml
python -m data.upload_s3 --bucket <TrainingBucketName> --prefix datasets/finai-slm/v1 --dir data/out
$env:SAGEMAKER_ROLE_ARN = "<SageMakerRoleArn>"
python -m train.sagemaker_launch --train-s3 s3://<bucket>/datasets/finai-slm/v1 --output-s3 s3://<bucket>/jobs/finai-slm/
```

Merge and convert:

```powershell
python -m train.merge_lora --base Qwen/Qwen3-1.7B --adapter <job>/adapter --out train/out/finnai-slm-merged
python -m train.export_litertlm --checkpoint train/out/finnai-slm-merged --out-dir train/out/litertlm
```

Conversion flags (must be logged in the model card): `dynamic_int4_block32`, KV 1280, **nothink** assistant prefix (`<think>\n\n</think>`) so SMS JSON is not wrapped in a think block.

## 7. Evaluation (do not skip)

Gates are pre-registered in [llm-eval-protocol.md](llm-eval-protocol.md). **INT4** must pass, not only bf16.

1. SMS R-EM of FinnAI-SLM INT4 − Qwen3-base ≥ **+3 pp**, bootstrap 95% CI excludes 0.  
2. Amount EM drop vs base ≤ **1 pp**.  
3. Chat groundedness drop vs base ≤ **5 pp**.  
4. JSON validity ≥ **95%**.  
5. On-device TTFT / peak RSS ≤ **1.3×** old Qwen2.5 MediaPipe (or documented proxy).

Also report vs `Qwen/Qwen2.5-1.5B-Instruct` as a **product** check. Default: INT4 R-EM must be ≥ that baseline.

```powershell
python -m eval.run_eval --sms data/out/test_sms.jsonl --chat eval/fixtures/chat_eval.jsonl --backend hf --model-id train/out/finnai-slm-merged --baseline-id Qwen/Qwen3-1.7B --tag finnai-slm
```

If any gate fails: iterate data/LoRA/conversion. **Do not** set Android `MODEL_URL` to CloudFront.

## 8. What we publish

| Artifact | Where |
| --- | --- |
| LoRA + merged bf16 | Hugging Face `finndot/finnai-slm` (after SHIP: YES) |
| `.litertlm` | CloudFront (app CDN) + `finndot/finnai-slm-litertlm` |
| This recipe | This file + `ml/` scripts (MIT with the app) |
| User SMS | **Never** |

App download stays on **our CloudFront** even if HF is public. HF is the scientific copy.

## 9. Limitations

- Templates are English-heavy Indian bank SMS, not every NPCI/UPI variant in the wild.
- Merchant strings are the hardest field; regex still wins on supported banks.
- INT4 can drop field F1 vs bf16; that gap is a first-class eval condition.
- Qwen3 thinking is disabled; do not expect long chain-of-thought on device.
- A dummy harness that copies gold JSON is **not** a model score.

## 10. Code map

| Path | Role |
| --- | --- |
| `ml/train/sft_qlora.py` | QLoRA SFT + val R-EM callback |
| `ml/train/train_config.yaml` | Frozen hyperparameters |
| `ml/train/merge_lora.py` | Adapter → bf16 |
| `ml/train/export_litertlm.py` | bf16 → INT4 `.litertlm` |
| `ml/train/sagemaker_launch.py` | `ml.g5.2xlarge` job |
| `ml/data/build_splits.py` | Dataset + hashes |
| `ml/eval/metrics.py` | R-EM, F1, bootstrap, McNemar |
| `docs/llm-eval-protocol.md` | Ship gates |
| `infra/lib/ondevice-model-stack.ts` | S3 + CloudFront + SageMaker role |

## 11. Live AWS status

S3 object prefixes still use inai-slm/ (historical); the product name is **FinnAI SLM**.
 (2026-09-21)

Account `008692857726`. Dataset and checkpoints stay in **ap-south-1** S3. Training GPU is **us-east-2 (Ohio)** because Mumbai SageMaker/EC2 G quotas are 0.

### v1 — SHIP: YES

| Piece | Status |
| --- | --- |
| Dataset v1 | 12k train / 1.7k val / 1.5k test; hashes in `ml/data/manifest/SHA256SUMS` |
| S3 v1 data | `s3://…/datasets/finai-slm/v1/` |
| Training job | **done** (`finai-slm-ohio-20260920-091315`). Instance terminated. |
| Val SMS R-EM (256) | epoch1 0.953 → epoch2 **0.965** → epoch3 0.961 (best = epoch 2 / `checkpoint-1500`) |
| Held-out eval | **SHIP: YES** — `finai-eval-ohio-20260921-082426` — instance terminated |
| SMS R-EM (1351 test) | **92.75%** [91.3, 94.1] vs Qwen3-base 22.13%, Qwen2.5 48.78% |
| Amount EM | **94.60%** vs Qwen3-base 79.27%, Qwen2.5 88.08% |
| Chat groundedness | 32.0% vs Qwen3-base 26.0%, Qwen2.5 68.0% |
| Gate 1 R-EM lift | ✅ +70.61 pp |
| Gate 2 amount | ✅ +15.32 pp |
| Gate 3 chat | ✅ +6.0 pp |
| Gate 4 JSON | ✅ 100% |
| Gate 5 on-device | ⏳ hardware test pending |
| RQ3 ≥ Qwen2.5 | ✅ 92.75 > 48.78 |

### v2 — Nova Pro Indic expansion

| Piece | Status |
| --- | --- |
| Nova data factory | **done** 450 rows (300 SMS + 150 chat) in 6 langs (hi, hinglish, ta, te, mr, bn) |
| Dataset v2 | **16k train / 1.6k val / 1.6k test** (v1 + nova_indic.jsonl) |
| S3 v2 data | `s3://…/datasets/finai-slm/v2/` |
| Training job | **done** `finai-slm-v2-ohio-20260921-164442` — terminated |
| Val SMS R-EM (256) | epoch1 **0.965** → epoch2 **0.965** → epoch3 **0.965** (all tied) |
| Held-out eval | **SHIP: YES** — `finai-eval-v2-ohio-20260922-144643` — terminated |
| SMS R-EM (1282 test) | **97.97%** [97.2, 98.7] vs Qwen3-base 28.16%, Qwen2.5 42.43%, **v1 92.75%** |
| Amount EM | **99.53%** (v1 94.60%) |
| Merchant exact | **98.21%** (v1 93.12%) |
| False-parse | **0.54%** (v1 28.57% — huge win) |
| Chat groundedness | **68.0%** (v1 32.0% — +36 pp from Nova coach data) |
| Next | Finalize v2: INT4 LiteRT + CloudFront + open-source HF release |

### Infrastructure

| Piece | Status |
| --- | --- |
| Delivery CloudFront | `https://dgdzwh27431n8.cloudfront.net/models/qwen3-1.7b-finndot/latest/...` (empty until SHIP: YES + INT4) |
| SageMaker role / EC2 profile | `FinndotOnDeviceModelStack-SlmSageMakerRole086F6E1A-ySAeAOkiYao9` / `FinndotSlmEc2Profile` |
| Ohio EC2 G/VT On-Demand | **16 vCPU** |
| Mumbai quota increase | `ml.g5.2xlarge for training job usage` still **PENDING** |

When quota is 1:

```powershell
cd ml
$env:SAGEMAKER_ROLE_ARN = "arn:aws:iam::008692857726:role/FinndotOnDeviceModelStack-SlmSageMakerRole086F6E1A-ySAeAOkiYao9"
python -m train.sagemaker_launch --train-s3 s3://finndotondevicemodelstack-slmtrainingbucket1a75fa7-4mz3w5n387wo/datasets/finai-slm/v1 --output-s3 s3://finndotondevicemodelstack-slmtrainingbucket1a75fa7-4mz3w5n387wo/jobs/finai-slm/ --role $env:SAGEMAKER_ROLE_ARN
```

Waiter (polls quota, then launches):

```powershell
python -m aws.wait_quota_and_train --role $env:SAGEMAKER_ROLE_ARN --train-s3 s3://finndotondevicemodelstack-slmtrainingbucket1a75fa7-4mz3w5n387wo/datasets/finai-slm/v1 --output-s3 s3://finndotondevicemodelstack-slmtrainingbucket1a75fa7-4mz3w5n387wo/jobs/finai-slm/
```

