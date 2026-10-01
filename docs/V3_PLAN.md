# FinnAI SLM v3 — real model upgrade + serious fine-tuning strategy

**Status:** redesign — great model **and** great training recipe are both required  
**Read first:** [`V3_FINETUNE_STRATEGY.md`](V3_FINETUNE_STRATEGY.md) (curriculum, DPO, checkpointing)  
**Product still on-device + privacy-first** — accept ~2–2.5 GB INT4 for **Qwen3-4B**

## One-line doctrine

Bigger base without staged fine-tuning is an expensive v2.  
**Anchor JSON → blend → harden groundedness (on-policy DPO) → SMS replay → trust INT4.**

## Honest critique

Adding BBPS templates / flat 3-epoch SFT on 4B is **not** v3. v2 already saturates synthetic SMS R-EM; Ask Finn still invents ₹ because CE-on-clones never punishes the model’s own hallucinations.

## Decision: ship a stronger base

| Role | Model | Why |
| --- | --- | --- |
| **Ship student (primary)** | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | Best open ~4B multilingual + structured-output prior; LiteRT INT4 path exists (~2–2.5 GB). Real headroom for coach + Indic vs 1.7B |
| **Teacher (GPU-heavy)** | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | Same family → clean distill / preference labels without GPT/Claude license risk |
| **Lite SKU (optional later)** | Distill 4B → **Qwen3-1.7B** | Low-RAM India phones keep a smaller file *after* v3 quality exists |

**Not primary:** Gemma-4 E2B (~2.5 GB+, different license), Llama 3.2 (weaker Indic + license), Sarvam (Indic-strong but no first-class LiteRT path).

**Phone cost we accept:** ~2–2.5× download vs v2’s ~974 MB. Mid-range TTFT will be slower; gate 5 becomes “usable on Snapdragon 7-class”, not “same as 1.7B”.

## What “really better” means (gates)

Compare **FinnAI-v3-4B** vs **FinnAI-v2-1.7B** (not only vs untuned base).

| Metric | v2 today | v3 ship bar |
| --- | ---: | ---: |
| Chat groundedness (frozen 50) | 68% | ≥ **90%** |
| SMS R-EM (full synthetic mix) | 97.97% | ≥ **97.5%** (non-regress) |
| Indic SMS R-EM (held-out slice) | thin / unreported | ≥ **96%** |
| India-flows R-EM (BBPS/EMI/…) | n/a | ≥ **95%** |
| False-parse | 0.54% | ≤ **1%** |
| Hard coach set (refuse invented ₹ / missing month) | weak | ≥ **85%** correct refusal |
| On-device | pending | INT4 loads; TTFT documented; no OOM on 8 GB devices |

If 4B loses to v2 on SMS R-EM by >1 pp after full recipe → **fail ship**, iterate data/DPO — do not call template-only SFT “v3 done”.

## Training recipe (spend GPU here)

```text
Phase A  Data factory     28k+ mix (flows + Indic + hard coach) + optional Nova
Phase B  Teacher SFT      Qwen3-8B QLoRA on same data          [1× A100 40/80 or 2× A10G]
Phase C  Student SFT      Qwen3-4B QLoRA on same data          [1× A10G/A100]
Phase D  Preference       DPO/ORPO on groundedness prefs
                          chosen = ledger-faithful; rejected = invented ₹ / markdown junk
Phase E  Distill (opt)    8B regenerates coach + hard SMS rationales → 4B SFT mix
Phase F  Export           merge → LiteRT INT4 → on-device bench
Phase G  Lite (opt)       distill 4B → 1.7B for low-end SKU
```

Rough GPU (order-of-magnitude, Ohio/Mumbai):

| Phase | Hardware | Wall time | ~USD |
| --- | --- | --- | --- |
| 8B teacher SFT | 1× A100 80GB | 8–14 h | 40–80 |
| 4B student SFT | 1× A100 or g5.2xlarge | 6–10 h | 25–50 |
| DPO 4B | same | 3–6 h | 15–30 |
| Distill regen + SFT | A100 | 6–12 h | 30–60 |
| **Total comfortable budget** | | | **~$150–250** |

Worth it if groundedness clears 90% and Indic slices clear 96%. Not worth it if we only re-SFT 1.7B on more templates.

## Data (supports the model — still not the model)

Keep the generators already in-repo, but raise the bar on **hardness**:

- Coach: missing-month, wrong-merchant, empty ledger, “quote a number not in prompt” traps  
- SMS: format drift, OCR-ish spacing, bilingual lines, failed vs success near-misses  
- Prefer **verified** Indic paraphrases (Nova or 8B teacher) over endless EN clones  
- Still **no raw user SMS** in public train set; optional private redacted eval later

## Repo layout (v3 train)

| Path | Role |
| --- | --- |
| [`ml/train/train_config_v3_4b.yaml`](../ml/train/train_config_v3_4b.yaml) | **Ship** Qwen3-4B QLoRA |
| [`ml/train/train_config_teacher_8b.yaml`](../ml/train/train_config_teacher_8b.yaml) | Teacher Qwen3-8B QLoRA |
| [`ml/train/train_dpo.py`](../ml/train/train_dpo.py) | DPO on preference jsonl |
| [`ml/data/generate_dpo_prefs.py`](../ml/data/generate_dpo_prefs.py) | Chosen/rejected coach pairs |
| [`ml/scripts/train_v3.sh`](../ml/scripts/train_v3.sh) | Orchestrates SFT → DPO |
| [`ml/scripts/build_v3_data.sh`](../ml/scripts/build_v3_data.sh) | Dataset build |

Legacy 1.7B `train_config.yaml` remains for lite distill experiments only.

## Execution order

1. Build v3 data (`build_v3_data.sh`)  
2. Train **8B teacher** SFT (quality oracle)  
3. Train **4B student** SFT on same data  
4. Build DPO prefs (rule-based rejects + teacher-ranked chosen) → DPO 4B  
5. Eval vs **v2 1.7B** + untuned 4B; require groundedness ≥90%  
6. LiteRT export + device bench on India mid-range  
7. Publish `finndot/finnai-slm-v3` (4B) + CloudFront `.litertlm`  
8. Optional: distill to 1.7B lite  

## Non-goals

- Calling “more BBPS strings on 1.7B” a v3 model release  
- Distilling from GPT/Claude into the OSS weights  
- Shipping 8B on-device as default  

## Success for Indian users

Ask Finn stops inventing rupees in Hinglish. Indic SMS and BBPS/EMI/wallet long-tail work because the **4B model actually understands them**, not because we memorized three new templates. Low-end phones can get a distilled 1.7B later — quality leads, size follows.
