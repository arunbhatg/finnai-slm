# Eval report (v20260922-finnai-slm-v2) — FinnAI v2

**SHIP: YES** (gates 1–4 pass; gate 5 on-device pending)

| | FinnAI v2 | FinnAI v1 | Qwen3-1.7B base | Qwen2.5-1.5B |
|---|---|---|---|---|
| SMS R-EM % | **97.97** [97.2, 98.7] | 92.75 | 28.16 | 42.43 |
| Amount EM % | **99.53** | 94.60 | 82.84 | 89.78 |
| JSON valid % | **100.0** | 100.0 | 100.0 | 99.45 |
| Field micro-F1 | **0.9911** | 0.9415 | 0.7123 | 0.8027 |
| Merchant exact % | **98.21** | 93.12 | 61.31 | 72.85 |
| False-parse % | **0.54** | 28.57 | 100.0 | 47.83 |
| Chat groundedness % | **68.0** | 32.0 | 26.0 | 68.0 |

## Gates

| Gate | Pass? | Value |
|---|---|---|
| 1. R-EM lift ≥ +3pp vs base | ✅ | +69.81 pp CI [67.3, 72.2] |
| 2. Amount EM non-inf ≥ −1pp | ✅ | +16.69 pp |
| 3. Chat groundedness non-inf ≥ −5pp | ✅ | +42.0 pp |
| 4. JSON ≥ 95% | ✅ | 100% |
| 5. On-device TTFT/RSS ≤ 1.3× | ⏳ | hardware test pending |
| RQ3: R-EM ≥ Qwen2.5 | ✅ | 97.97 > 42.43 |
| McNemar p | — | 7.6e-270 |

## v2 vs v1 deltas

| Metric | Δ |
|---|---|
| SMS R-EM | **+5.22 pp** |
| Amount EM | +4.93 pp |
| Merchant exact | +5.09 pp |
| False-parse | **−28.03 pp** (0.54% vs 28.57%) |
| Chat groundedness | **+36.0 pp** (68% vs 32%) |

## Notes
- Adapter: epoch-3 merged (`finnai-slm-v2-ohio-20260921-164442`), val SMS R-EM 0.965 all epochs
- Nova Pro Indic expansion (300 SMS + 150 chat) drove the chat groundedness jump
- False-parse collapse from 28.57% → 0.54% is the biggest quality win
- **Txn-only vs junk SMS** (1,098 transactions / 184 non-txn in the same test):

| | FinnAI v2 | Qwen3-1.7B | Qwen2.5-1.5B |
|---|---:|---:|---:|
| Txn-only Amount EM % | **99.54** | **96.72** | 96.08 |
| Txn-only Merchant % | **98.00** | 71.58 | 76.32 |
| Txn-only R-EM % | **97.72** | 32.88 | 40.80 |
| Non-txn false-parse % | **0.54** | **100.0** | 47.83 |

  Bases already read amounts on real transactions. Qwen3’s headline R-EM looks terrible mainly because it **always invents a transaction on OTP/promo**. Fine-tuning teaches `{}` there — critical so junk SMS never become spends.
- Qwen2.5 / Qwen3 columns are untuned Instruct/chat checkpoints; FinnAI is the fine-tune.
- AWS jobs: train `finnai-slm-v2-ohio-20260921-164442`, eval `finnai-eval-v2-ohio-20260922-144643`
