# Eval report (v20260921-finnai-slm) — FinnAI v1

**SHIP: YES** (gates 1–4 pass; gate 5 on-device pending)

| | FinnAI v1 (merged-best) | Qwen3-1.7B base | Qwen2.5-1.5B (product) |
|---|---|---|---|
| SMS R-EM % | **92.75** [91.3, 94.1] | 22.13 | 48.78 |
| Amount EM % | **94.60** | 79.27 | 88.08 |
| JSON valid % | **100.0** | 100.0 | — |
| Field micro-F1 | **0.9415** | 0.6865 | — |
| Merchant exact % | **93.12** | 54.48 | — |
| False-parse % | 28.57 | 100.0 | — |
| Chat groundedness % | 32.0 | 26.0 | 68.0 |
| No-markdown % | 100.0 | 42.0 | — |

## Gates

| Gate | Pass? | Value |
|---|---|---|
| 1. R-EM lift ≥ +3pp vs base | ✅ | +70.61 pp CI [68.2, 73.0] |
| 2. Amount EM non-inf ≥ −1pp | ✅ | +15.32 pp |
| 3. Chat groundedness non-inf ≥ −5pp | ✅ | +6.0 pp |
| 4. JSON ≥ 95% | ✅ | 100% |
| 5. On-device TTFT/RSS ≤ 1.3× | ⏳ | hardware test pending |
| RQ3: R-EM ≥ Qwen2.5 | ✅ | 92.75 > 48.78 |
| McNemar p | — | 1.3e-287 |

## Notes
- Adapter: epoch-2 (`checkpoint-1500`), val SMS R-EM 0.965
- Chat groundedness is low for both models (32% vs 26%); v2 Indic + more coach data targets this
- AWS job: `finnai-slm-ohio-20260920-091315`, eval: `finnai-eval-ohio-20260921-082426`
