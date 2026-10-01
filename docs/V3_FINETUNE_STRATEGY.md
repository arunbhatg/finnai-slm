# FinnAI v3 — fine-tuning strategy (the real lever)

Base model choice (4B) buys **headroom**. Fine-tuning strategy decides whether that headroom becomes a **better product**.

v1/v2 already proved: mixed QLoRA SFT + template gold → excellent *synthetic* SMS R-EM. They also proved the failure mode: **Ask Finn still invents ₹** (68% groundedness) because CE-on-clones does not teach “never emit a number absent from the prompt.”

This doc is the v3 training doctrine. Templates and India-flow generators are *fuel*. This is the *engine*.

---

## 1. What is broken in the current recipe

| Practice (v1/v2) | Why it plateaus |
| --- | --- |
| Single mixed SFT (60% SMS / 25% coach / 15% general) | One CE loss. Model learns format averages. Coach and JSON pull in opposite directions (terse `{}` vs prose). |
| Checkpoint = max val SMS R-EM only | Can ship an epoch that is great at SMS and mediocre at groundedness (or the reverse). |
| 3 epochs, flat shuffle | Early epochs underfit hard negatives; late epochs memorize templates (v1 epoch-3 dip). |
| LoRA r=16 | Enough for JSON style; thin for long grounded coach + Indic surface variety on 4B. |
| Gold-only SFT for chat | Never sees *its own* bad answers. DPO with only synthetic rejects is weak if rejects ≠ model errors. |
| No post-DPO SMS replay | Preference stage can soften JSON discipline. |
| Eval mostly in-distribution templates | High R-EM ≠ production transfer. |

**Doctrine:** treat SMS extraction and coach groundedness as **coupled products with staged optimization**, not one flat SFT soup.

---

## 2. Objective stack (optimize in this order)

1. **Safety of money numbers (coach)** — every ₹/digit in the reply ⊆ prompt ledger (`must_ground`). Absolute floor **≥ 90%**.  
2. **Refusal correctness** — OTP/promo/failed UPI/statement-only → `{}` (false-parse ≤ 1%).  
3. **SMS field exactness** — strict R-EM non-regress vs v2 on shared slices.  
4. **Indic + India-flow transfer** — held-out slices ≥ 95–96%.  
5. **On-device** — INT4 fidelity; document TTFT (accept slower than 1.7B).

If (1) and (3) conflict, **prefer (1) after a short SMS replay**, not the other way around — Ask Finn hallucinations are user-visible trust failures; regex still covers many SMS banks.

---

## 3. Staged curriculum (not one SFT)

```text
Stage 0  Data hardness     Build train with explicit hard buckets
Stage 1  SMS anchor        SFT mostly SMS + negatives (+ light general)
Stage 2  Multi-task blend  SFT full mix (SMS+coach+general)
Stage 3  Coach harden      SFT coach-heavy + hard refusals
Stage 4  On-policy prefs   Sample student mistakes → DPO/ORPO
Stage 5  SMS replay        Short CE on SMS+negatives (anti-forget)
Stage 6  INT4 reality      Merge → quantize → eval INT4 == ship condition
```

### Stage 0 — data buckets (same rows, tagged)

Every train row gets `bucket` ∈:

- `sms_core` — UPI/NEFT/IMPS/CC spend  
- `sms_flow` — BBPS/EMI/wallet/MF/refund  
- `sms_indic` — HI/Hinglish/TA/TE/MR/BN  
- `sms_neg` — must be `{}`  
- `coach_ground` — normal grounded Q&A  
- `coach_hard` — missing month, invent-refuse, empty ledger  
- `general` — short India finance FAQ  

Splits still by `(bank, template_id)`. Upsample `sms_neg` + `coach_hard` in later stages.

### Stage 1 — SMS anchor (1–2 epochs)

- Mix ≈ **80% SMS (core+flow+indic+neg) / 10% coach / 10% general**  
- Goal: lock JSON schema + refusal before coach prose dilutes it  
- Select checkpoint by **val SMS R-EM + false-parse**

### Stage 2 — multi-task blend (1–2 epochs)

- Mix ≈ **55% SMS / 30% coach / 15% general** (v3 default)  
- Continue from Stage 1 adapter (do not reset LoRA)  
- Select by **composite** (below)

### Stage 3 — coach harden (1 epoch)

- Mix ≈ **35% SMS / 55% coach (half hard) / 10% general**  
- Same LoRA; lower LR (≤ 5e-5)  
- Select by **groundedness**, subject to SMS R-EM drop ≤ 1 pp vs Stage 2 best

### Stage 4 — on-policy preference (DPO/ORPO)

**Critical upgrade vs rule-only rejects:**

1. Take Stage 3 student; generate coach answers on a held-out preference prompt set (T=0.7, n=2–4 samples).  
2. Auto-label: if `groundedness(reply)=False` → **rejected**; gold / teacher grounded reply → **chosen**.  
3. Add a minority of synthetic catastrophic rejects (invent ₹99999) for coverage.  
4. DPO β ≈ 0.1, 0.5–1 epoch, LR 5e-5, **chat prompts only** (do not DPO on SMS JSON).  

Optional: teacher 8B ranks 2 student samples when both are grounded (style/helpfulness) — still no GPT distill into weights.

### Stage 5 — SMS replay (0.25–0.5 epoch)

- Pure `sms_*` + `sms_neg` CE at low LR  
- Restores any JSON drift from DPO  
- Stop when val R-EM recovers to ≥ Stage 2 − 0.5 pp

### Stage 6 — quantization is part of training

Ship gate runs on **INT4 LiteRT** (or at least bitsandbytes/AWQ proxy), not only bf16 merge. If bf16 groundedness is 92% and INT4 is 84%, **fail** — fix with slightly higher-bit KV or more DPO, don’t ship bf16 numbers.

---

## 4. Loss and capacity choices

### LoRA

| Knob | v2 | v3 4B recommendation |
| --- | --- | --- |
| Rank | 16 | **32** (try 64 if coach still shallow) |
| Alpha | 32 | **64** (α≈2r) |
| Targets | attn+MLP | keep attn+MLP (attn-only cost ~5 pp R-EM historically) |
| Continue | fresh each version | **continue adapters across stages** |

Avoid full FT on 4B for this dataset size — forgetting + cost. If ablating full FT, only on 8B teacher with heavy regularization.

### Sequence / packing

- Keep **packing off** (SMS boundaries matter).  
- `max_seq_len` **2048** for coach ledgers; SMS stays short.  
- `enable_thinking=false` everywhere (train + serve). Thinking burns TTFT and encourages verbose inventiveness.

### Token weighting (optional Phase B)

If SMS fields still lag after Stage 1: weight CE on assistant JSON value spans (amount/merchant/account) ×2–3. Implement only if Stage 1 R-EM stalls; don’t start here.

---

## 5. Checkpoint selection (composite, not loss)

Per epoch / stage end, compute on frozen sets:

```text
score = 0.45 * chat_groundedness
      + 0.35 * sms_r_em
      + 0.10 * (1 - false_parse)
      + 0.10 * indic_r_em
```

- Save **every** stage-best adapter.  
- Ship = Stage 5 winner under INT4 eval.  
- Never pick by train CE alone (already a known trap).

---

## 6. Anti-patterns (do not do)

1. **One 3-epoch shuffle on 4B and call it v3** — that is v2 with a bigger base.  
2. **DPO on SMS** — preferences on JSON are awkward; use CE + hard negatives.  
3. **Teacher writes gold JSON** — Nova/8B may invent merchants; gold stays deterministic.  
4. **Eval only on train-like templates** — always report india_flows + indic slices.  
5. **Raise LR to “use the GPU”** — prefer more stages / on-policy DPO over LR heroics.  
6. **Ship bf16 metrics** — users download INT4.

---

## 7. Minimal experiment matrix (spend GPU wisely)

Run in this order; stop early if a gate is hopeless:

| Exp | Question | Budget |
| --- | --- | --- |
| A | Stage1→2 only on 4B vs flat 3-epoch mix | ~1× A10G day |
| B | + Stage3 coach harden | +0.5 day |
| C | + on-policy DPO vs rule-only DPO | +0.5–1 day |
| D | + Stage5 SMS replay | +0.25 day |
| E | Teacher 8B for ranking / distill coach text | +1–2 days |
| F | LoRA r=32 vs 64 after C | +0.5 day |

**Success:** C or D clears groundedness ≥90% without SMS R-EM collapse.  
**If not:** E before blaming the base model.  
**If E fails:** reconsider task split (separate tiny JSON head vs coach model) — last resort product change.

---

## 8. Wiring in this repo

| Artifact | Role |
| --- | --- |
| `ml/data/build_splits.py --version v3` | Base pool |
| `ml/data/tag_curriculum.py` | Assign `bucket` + write `stage{1,2,3,5}.jsonl` |
| `ml/data/generate_dpo_prefs.py` | Rule rejects (bootstrap) |
| `ml/data/generate_onpolicy_prefs.py` | Student-sampled rejects (Stage 4) |
| `ml/train/train_config_v3_4b.yaml` | 4B LoRA r=32 |
| `ml/train/sft_qlora.py` | SFT; continue `--adapter` across stages |
| `ml/train/train_dpo.py` | Stage 4 |
| `ml/scripts/train_v3.sh` | Orchestrates curriculum |

---

## 9. One-sentence strategy

**Anchor JSON, blend tasks, harden groundedness with on-policy DPO, replay SMS, and only then trust INT4 — a bigger base without this sequence is an expensive v2.**
