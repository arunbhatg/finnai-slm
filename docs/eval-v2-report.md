# Eval report (v20260922-finnai-slm-v2) — FinnAI v2

**SHIP: YES** (gates 1–4 pass; gate 5 on-device pending)

## Non-empty rows only (1,098 real transactions)

This is the fair extraction comparison — gold is a real transaction JSON, not `{}`.

| Metric | FinnAI v2 | Qwen3-1.7B (untuned) |
|---|---:|---:|
| Amount EM % | **99.54** | **96.72** |
| Merchant exact % | **98.00** | 71.58 |
| Strict R-EM % | **97.72** | 32.88 |

## Empty / non-transaction rows (184)

| Metric | FinnAI v2 | Qwen3-1.7B (untuned) |
|---|---:|---:|
| False-parse % | **0.54** | **100.0** |

## Full mix (1,282) — ship-gate table

| | FinnAI v2 | FinnAI v1 | Qwen3-1.7B base |
|---|---|---|---|
| SMS R-EM % | **97.97** [97.2, 98.7] | 92.75 | 28.16 |
| Amount EM % | **99.53** | 94.60 | 82.84 |
| JSON valid % | **100.0** | 100.0 | 100.0 |
| Field micro-F1 | **0.9911** | 0.9415 | 0.7123 |
| Merchant exact % | **98.21** | 93.12 | 61.31 |
| False-parse % | **0.54** | 28.57 | 100.0 |
| Chat groundedness % | **68.0** | 32.0 | 26.0 |

## Gates

| Gate | Pass? | Value |
|---|---|---|
| 1. R-EM lift ≥ +3pp vs base | ✅ | +69.81 pp CI [67.3, 72.2] |
| 2. Amount EM non-inf ≥ −1pp | ✅ | +16.69 pp |
| 3. Chat groundedness non-inf ≥ −5pp | ✅ | +42.0 pp |
| 4. JSON ≥ 95% | ✅ | 100% |
| 5. On-device TTFT/RSS ≤ 1.3× prior baseline | ⏳ | hardware test pending |
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
- On **non-empty** rows, untuned Qwen3 amount EM is already ~96.7%. The scary overall R-EM gap vs base is mostly empty-row false-parse (Qwen3 invents a spend on every OTP/promo).
- Fine-tuning buys refusal on empty rows + full-field match on non-empty rows.
- Adapter: epoch-3 merged (`finnai-slm-v2-ohio-20260921-164442`), val SMS R-EM 0.965 all epochs
- Nova Pro Indic expansion (300 SMS + 150 chat) drove the chat groundedness jump
- Qwen3 column is the untuned base; FinnAI is the fine-tune
- Non-empty metrics derived from published aggregates + 1,098 / 184 split
- AWS jobs: train `finnai-slm-v2-ohio-20260921-164442`, eval `finnai-eval-v2-ohio-20260922-144643`
