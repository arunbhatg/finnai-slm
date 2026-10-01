#!/usr/bin/env bash
# FinnAI v3 curriculum: SMS anchor → blend → coach harden → on-policy DPO → SMS replay.
# See docs/V3_FINETUNE_STRATEGY.md — this is the training doctrine, not flat SFT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_OUT="${DATA_OUT:-data/out_v3}"
CONFIG_4B="train/train_config_v3_4b.yaml"
BASE_OUT="${BASE_OUT:-train/output_v3_4b}"
SKIP_ONPOLICY="${SKIP_ONPOLICY:-0}"

echo "== 0) Data + curriculum tags =="
bash scripts/build_v3_data.sh
python3 -m data.tag_curriculum --train "$DATA_OUT/train.jsonl" --out-dir "$DATA_OUT"
python3 -m data.generate_dpo_prefs --out "$DATA_OUT/dpo_rule_prefs.jsonl" --n 2000

S1="$BASE_OUT/stage1"
S2="$BASE_OUT/stage2"
S3="$BASE_OUT/stage3"
S4="$BASE_OUT/stage4_dpo"
S5="$BASE_OUT/stage5_replay"

echo "== 1) SMS anchor =="
export SM_CHANNEL_TRAIN="$PWD/$DATA_OUT"
export SM_MODEL_DIR="$PWD/$S1"
python3 train/sft_qlora.py --config "$CONFIG_4B" \
  --train-file "$DATA_OUT/train_stage1.jsonl" --val-file "$DATA_OUT/val.jsonl" \
  --epochs 2 --lr 1.5e-4

echo "== 2) Multi-task blend (continue adapter) =="
export SM_MODEL_DIR="$PWD/$S2"
python3 train/sft_qlora.py --config "$CONFIG_4B" \
  --adapter "$S1/adapter" \
  --train-file "$DATA_OUT/train_stage2.jsonl" --val-file "$DATA_OUT/val.jsonl" \
  --epochs 2 --lr 1.0e-4

echo "== 3) Coach harden =="
export SM_MODEL_DIR="$PWD/$S3"
python3 train/sft_qlora.py --config "$CONFIG_4B" \
  --adapter "$S2/adapter" \
  --train-file "$DATA_OUT/train_stage3.jsonl" --val-file "$DATA_OUT/val.jsonl" \
  --epochs 1 --lr 5.0e-5

echo "== 3b) Merge stage3 for on-policy sampling =="
python3 train/merge_lora.py --base Qwen/Qwen3-4B --adapter "$S3/adapter" --out "$S3/merged"

PREF_FILE="$DATA_OUT/dpo_rule_prefs.jsonl"
if [[ "$SKIP_ONPOLICY" != "1" ]]; then
  echo "== 4a) On-policy preference mining =="
  python3 -m data.generate_onpolicy_prefs \
    --seeds "$DATA_OUT/train_tagged.jsonl" \
    --model-id "$S3/merged" \
    --out "$DATA_OUT/dpo_onpolicy.jsonl" \
    --n 2000
  # Merge rule + on-policy prefs
  cat "$DATA_OUT/dpo_rule_prefs.jsonl" "$DATA_OUT/dpo_onpolicy.jsonl" > "$DATA_OUT/dpo_prefs.jsonl"
  PREF_FILE="$DATA_OUT/dpo_prefs.jsonl"
fi

echo "== 4b) DPO =="
python3 train/train_dpo.py \
  --config "$CONFIG_4B" \
  --prefs "$PREF_FILE" \
  --adapter "$S3/adapter" \
  --output-dir "$S4" \
  --epochs 1 --lr 5e-5 --beta 0.1

echo "== 5) SMS replay =="
export SM_MODEL_DIR="$PWD/$S5"
python3 train/sft_qlora.py --config "$CONFIG_4B" \
  --adapter "$S4/adapter" \
  --train-file "$DATA_OUT/train_stage5.jsonl" --val-file "$DATA_OUT/val.jsonl" \
  --epochs 0.5 --lr 3.0e-5

echo "== 6) Final merge =="
python3 train/merge_lora.py --base Qwen/Qwen3-4B --adapter "$S5/adapter" --out "$S5/merged"

echo "== Done =="
echo "Ship candidate: $S5/merged"
echo "Next: INT4 export + eval.run_eval --version v3 (groundedness ≥90%, SMS non-regress)"
echo "Doctrine: docs/V3_FINETUNE_STRATEGY.md"
