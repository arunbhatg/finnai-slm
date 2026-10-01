#!/usr/bin/env bash
# FinnAI v3 training orchestration: data → (optional teacher) → 4B SFT → DPO.
# Requires a CUDA GPU. Great model path = Qwen3-4B ship student.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_OUT="${DATA_OUT:-data/out_v3}"
SFT_OUT="${SFT_OUT:-train/output_v3_4b}"
DPO_OUT="${DPO_OUT:-train/output_v3_4b_dpo}"
TEACHER_OUT="${TEACHER_OUT:-train/output_v3_teacher_8b}"
RUN_TEACHER="${RUN_TEACHER:-0}"
CONFIG_4B="train/train_config_v3_4b.yaml"
CONFIG_8B="train/train_config_teacher_8b.yaml"

echo "== 1) Build v3 dataset =="
bash scripts/build_v3_data.sh
# rebuild into DATA_OUT if custom
if [[ "$DATA_OUT" != "data/out_v3" ]]; then
  python3 -m data.build_splits --version v3 --repo-root .. --out "$DATA_OUT" --train-size 28000
fi

echo "== 2) DPO preference factory =="
python3 -m data.generate_dpo_prefs --out "$DATA_OUT/dpo_prefs.jsonl" --n 4000

if [[ "$RUN_TEACHER" == "1" ]]; then
  echo "== 3a) Teacher SFT Qwen3-8B (GPU heavy) =="
  export SM_CHANNEL_TRAIN="$PWD/$DATA_OUT"
  export SM_MODEL_DIR="$PWD/$TEACHER_OUT"
  python3 train/sft_qlora.py --config "$CONFIG_8B"
fi

echo "== 3b) Student SFT Qwen3-4B =="
export SM_CHANNEL_TRAIN="$PWD/$DATA_OUT"
export SM_MODEL_DIR="$PWD/$SFT_OUT"
python3 train/sft_qlora.py --config "$CONFIG_4B"

echo "== 4) DPO on 4B adapter =="
python3 train/train_dpo.py \
  --config "$CONFIG_4B" \
  --prefs "$DATA_OUT/dpo_prefs.jsonl" \
  --adapter "$SFT_OUT/adapter" \
  --output-dir "$DPO_OUT"

echo "== 5) Merge DPO adapter =="
python3 train/merge_lora.py --adapter "$DPO_OUT/adapter" --out "$DPO_OUT/merged" \
  || python3 train/merge_lora.py --adapter "$DPO_OUT/adapter" --out "$DPO_OUT/merged" --base Qwen/Qwen3-4B

echo "== Done =="
echo "Merged model: $DPO_OUT/merged"
echo "Eval next:"
echo "  python3 -m eval.run_eval --backend hf --version v3 \\"
echo "    --sms $DATA_OUT/test_sms.jsonl --chat eval/fixtures/chat_eval.jsonl \\"
echo "    --sms-slice eval/fixtures/indic_sms_eval.jsonl \\"
echo "    --sms-slice eval/fixtures/india_flows_eval.jsonl \\"
echo "    --model-id $DPO_OUT/merged --baseline-id Qwen/Qwen3-4B --tag v3-4b-dpo"
