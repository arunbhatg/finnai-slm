# Eval report (v20260921-finnai-slm) — FinnAI v1

**SHIP: YES** (gates 1–4 pass; gate 5 on-device pending)

| | FinnAI v1 (merged-best) | Qwen3-1.7B base |
|---|---|---|
| SMS R-EM % | **92.75** [91.3, 94.1] | 22.13 |
| Amount EM % | **94.60** | 79.27 |
| JSON valid % | **100.0** | 100.0 |
| Field micro-F1 | **0.9415** | 0.6405 |
| Merchant exact % | **93.12** | 54.48 |
| False-parse % | **28.57** | 98.21 |
| Chat groundedness % | **32.0** | 26.0 |

## Gates

| Gate | Pass? | Value |
|---|---|---|
| 1. R-EM lift ≥ +3pp vs base | ✅ | +70.61 pp |
| 2. Amount EM non-inf ≥ −1pp | ✅ | +15.32 pp |
| 3. Chat groundedness non-inf ≥ −5pp | ✅ | +6.0 pp |
| 4. JSON ≥ 95% | ✅ | 100% |
| 5. On-device TTFT/RSS ≤ 1.3× prior baseline | ⏳ | hardware test pending |
| McNemar p | — | 1.3e-287 |
