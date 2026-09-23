# FinnDot on-device SLM evaluation protocol

**Status:** pre-registered (frozen before test scores are inspected)  
**Date frozen:** 2026-09-20  
**Task:** replace Qwen2.5-1.5B-Instruct (MediaPipe `.task`) with a QLoRA fine-tune of Qwen3-1.7B, exported to LiteRT-LM INT4 (`.litertlm`).  
**This document is the ship gate.** Do not change thresholds after looking at held-out scores. If the protocol must change, bump the date, describe the delta, and re-run every model under the new protocol.

An independent reviewer should be able to reproduce the tables from:

1. This file
2. Dataset split hashes in `ml/data/out/SHA256SUMS`
3. Scripts under `ml/eval/` and `ml/train/`
4. The exact Hugging Face / S3 artifact URIs logged in the report

## 1. Research questions

**RQ1 (primary, scientific).** Does FinnDot QLoRA SFT of `Qwen/Qwen3-1.7B` improve Indian bank SMS → JSON extraction versus the same Qwen3-1.7B Instruct checkpoint with thinking disabled, under identical prompts and decoding?

**RQ2 (quantization).** Does INT4 LiteRT-LM conversion preserve RQ1 gains within the registered non-inferiority margins versus the merged bf16 fine-tune?

**RQ3 (product regression).** Is the INT4 fine-tune no worse than today’s shipped Qwen2.5-1.5B-Instruct on SMS extraction and finance-coach groundedness? Architecture and runtime differ, so RQ3 is a product check, not a controlled ablation.

**RQ4 (on-device).** On a mid-range arm64 phone, is INT4 Qwen3 TTFT and peak RSS no worse than 1.3× the current MediaPipe Qwen2.5 path?

## 2. Models

| ID | Artifact | Notes |
| --- | --- | --- |
| `qwen25_hf` | `Qwen/Qwen2.5-1.5B-Instruct` (Transformers, bf16 or 4-bit) | Product baseline. Not MediaPipe. |
| `qwen3_base` | `Qwen/Qwen3-1.7B` Instruct, `enable_thinking=false` | Scientific baseline. |
| `qwen3_ft_bf16` | Merged LoRA → bf16 safetensors | Candidate before conversion. |
| `qwen3_ft_int4` | LiteRT-LM INT4 `nothink` `.litertlm` | **Only this ID may ship.** |

Optional diagnostic (not a gate): `qwen3_base` in Transformers vs the public `litert-community/Qwen3-1.7B` INT4 file, to isolate conversion loss on the base model.

## 3. Decoding (frozen)

All text-generation evals use:

| Setting | Value |
| --- | --- |
| Temperature | `0` |
| Top-p | `1.0` |
| Top-k | disabled / full |
| Max new tokens | `256` (SMS), `512` (chat) |
| Stop | model EOS + `}` extra stop for SMS after a complete JSON object |
| Seed | `42` |
| Thinking | off (`enable_thinking=false` or empty `<think></think>` prefix) |
| Batch size | 1 (on-device); ≤8 on GPU if numerically identical to batch-1 on a 32-item smoke set |

SMS prompt is exactly `ml/data/sms_prompt.py` / `LlmSmsParser.buildPrompt`. Do not add few-shot examples at eval time.

Post-process (frozen, applied to every model):

1. Strip a single markdown fence if present (` ```json ` … ` ``` `).
2. Strip `<think>…</think>` if the model emitted one.
3. Parse JSON with a lenient decoder (`true`/`false`/`null`, trailing commas not allowed).

## 4. Data

### 4.1 Sources (allowed)

- Parser-core JUnit gold (`ParserTestCase` / `ExpectedTransaction`) extracted by `ml/data/extract_parser_gold.py`.
- Synthetic Indian bank SMS from `ml/data/generate_synthetic_sms.py` (grammar templates, not user inboxes).
- Synthetic finance-coach dialogues from `ml/data/generate_finance_chat.py` matching `LlmRepository.buildSystemPrompt`.
- Synthetic general instruct from `ml/data/generate_general_instruct.py` (Apache-friendly, original text).

### 4.2 Sources (forbidden)

- Firestore `sms_reports` bodies, Room unrecognized SMS, analytics S3 pickles, Claude/Bedrock distillations, any live user SMS.

### 4.3 Splits

Unit of assignment is `(bank, template_id)`, **not** a single SMS line. Every amount/merchant variant of a template stays in one split.

- Hash `sha256(f"{seed}:{bank}:{template_id}")` with `seed=42`.
- Sort groups by hash; assign whole groups until approximately 80% train / 10% val / 10% test by example count.
- Mix on **train only** (target 60% SMS / 25% finance chat / 15% general). Val and test keep natural SMS+chat coverage; general instruct is train-only.
- Chat eval set is 50 frozen items in `ml/eval/fixtures/chat_eval.jsonl` (never used in SFT).

Publish `ml/data/out/SHA256SUMS` (jsonl files + example-id lists). Reviewers verify the hash, not a screenshot of counts.

### 4.4 Target scale

8k–20k train, ~1k val, ~1k test SMS-capable items. Quality and split hygiene beat raw size.

## 5. SMS metrics

Gold fields: `amount`, `merchant`, `type` ∈ {`INCOME`,`EXPENSE`,`CREDIT`,`TRANSFER`,`INVESTMENT`}, `account` (last 4 or null), `balance` (or null), `category` (optional, not in strict R-EM).

Non-transaction gold is `{}`. Predicting a non-empty object is a false parse.

| Metric | Definition |
| --- | --- |
| JSON validity | Object parsed after frozen post-process. Empty `{}` is valid. |
| Amount EM | `Decimal` compare after stripping `Rs`/`INR`/`₹`, commas, and spaces. Scale-insensitive (`5000` == `5000.00`). |
| Type EM | Exact enum string after uppercasing. |
| Account EM | Last 4 digits or both null. |
| Balance EM | Same numeric rule as amount, or both null. |
| Merchant exact | Casefold, collapse whitespace, strip `UPI/` prefix. |
| Merchant token F1 | Whitespace tokens of the normalized merchant; empty vs empty = 1.0. |
| Field micro-F1 | Binary presence+value match per field over {amount, type, merchant, account, balance} (nulls counted). |
| Strict R-EM | Transaction items: amount EM ∧ type EM ∧ merchant exact ∧ account EM ∧ balance EM. Non-transaction items: predicted `{}`. |
| False-parse rate | Share of non-transaction items with a parseable non-empty object. |

Primary SMS endpoint: **strict R-EM**. Field F1 and amount EM are secondary.

## 6. Chat metrics

Frozen 50-item set. Each item has a synthetic ledger in the production system-prompt shape, a user turn, and a list of **must-ground numbers** (every figure that appears in the prompt).

| Metric | Definition |
| --- | --- |
| Groundedness | Every `₹` / `Rs` / ASCII number in the reply appears in the prompt (after comma-strip). Fail = hallucination. |
| No-markdown | No `*`, `#` headings, or fenced code. |
| Length band | 40–700 characters (inclusive). |
| Rubric 1–5 | Actionability, tone (non-judgmental), uses rupee amounts from context. Two human annotators on 100% of v1; report Cohen’s κ. |

**Ship gate uses automatic groundedness only.** Rubric and LLM-as-judge are reported but not gated until κ ≥ 0.6.

## 7. Statistics

- Paired item-level scores for every model pair.
- **10,000** bootstrap resamples of items; report 95% percentile CI for R-EM, amount EM, field micro-F1, and chat groundedness.
- **McNemar** exact test on strict R-EM vs `qwen3_base` (two-sided α = 0.05). Treat as supporting evidence; the CI on the difference is the registered gate.
- Seed for bootstrap: `42`.

## 8. On-device protocol

Devices: one mid-range arm64 (e.g. Snapdragon 7-series class, 6–8 GB RAM) and one flagship. Same 100 SMS + 20 chat prompts for every runtime.

Record: time-to-first-token (warm), decode tokens/s, peak RSS / private footprint, JSON validity, thermal throttle notes.

Community LiteRT-LM microbench numbers are **not** a substitute.

Compare `qwen3_ft_int4` via LiteRT-LM to the currently shipped MediaPipe Qwen2.5 path. If Qwen2.5 `.task` is no longer loadable after the runtime swap, compare against a recorded MediaPipe baseline in `ml/eval/reports/mediapipe_qwen25_baseline.json` captured before the swap, or against Transformers `qwen25_hf` latency on the same device as a labeled proxy (must be stated).

## 9. Pre-registered ship gates

Ship `qwen3_ft_int4` (update `Constants.ModelDownload.MODEL_URL` to the CloudFront fine-tune) **only if all** hold on the frozen test set:

1. **SMS R-EM lift:** `qwen3_ft_int4` − `qwen3_base` ≥ **+3.0 percentage points**, and the bootstrap 95% CI on that difference does **not** include 0.
2. **Amount EM non-inferiority:** `qwen3_ft_int4` − `qwen3_base` ≥ **−1.0 percentage point**.
3. **Chat groundedness non-inferiority:** `qwen3_ft_int4` − `qwen3_base` ≥ **−5.0 percentage points**.
4. **JSON validity:** `qwen3_ft_int4` ≥ **95%** on SMS test items.
5. **On-device cost:** mid-range TTFT and peak RSS ≤ **1.3×** the Qwen2.5 MediaPipe (or documented proxy) measurement.

If any gate fails: do not change `MODEL_URL` to the fine-tune. Iterate data/LoRA/conversion and re-run the full suite. Partial “looks better on val” is not a ship.

RQ3 (vs Qwen2.5) is reported in the same tables. A drop vs Qwen2.5 on R-EM should block ship even if RQ1 passes, unless a documented product decision accepts it. Default: **INT4 R-EM must be ≥ `qwen25_hf` R-EM**.

## 10. Training (for reviewers)

- Base: `Qwen/Qwen3-1.7B`
- Method: QLoRA 4-bit, rank 16, alpha 32, 3 epochs, lr `2e-4`, no packing
- Checkpoint selection: **best validation SMS R-EM**, not train loss
- Conversion: `litert-torch export_hf`, `dynamic_int4_block32`, `nothink` prefixes, KV cache 1280

Exact flags live in `ml/train/train_config.yaml` and the conversion script. The report must paste the SageMaker / EC2 job id and hyperparameter JSON.

## 11. Report artifact

`ml/eval/reports/vYYYYMMDD.md` plus `summary.json` must include:

- Git commit of this protocol and of `ml/`
- Dataset SHA-256
- Model URIs
- Tables with point estimates and 95% CIs
- McNemar p-value
- 20 SMS fail cases (template_id only, full SMS text is synthetic and may be inlined)
- Explicit `SHIP: YES` or `SHIP: NO` with failed gate ids

## 12. References (methodology)

Field-level F1 plus strict record exact-match is the standard pair for LLM JSON extraction; per-entity F1 is standard for Indian SMS NER. This protocol uses both, with R-EM as the primary endpoint, bootstrap CIs, and a pre-registered non-inferiority chat check so extraction SFT cannot silently destroy Ask/Learn.
