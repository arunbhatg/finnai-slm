#!/usr/bin/env bash
# Build FinnAI SLM v3 synthetic dataset (no GPU, no Bedrock required).
# Optional: pass --with-nova if BEDROCK credentials are configured.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/ml"
OUT="${OUT:-data/out_v3}"
WITH_NOVA=0
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --with-nova) WITH_NOVA=1 ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

python3 -m pip install -q -r requirements.txt
mkdir -p "$OUT"

if [[ "$WITH_NOVA" -eq 1 ]]; then
  echo "Generating Nova Indic expansion (requires Bedrock)..."
  python3 -m data.generate_nova_indic --out "$OUT/nova_indic.jsonl" --sms 2000 --chat 800
  EXTRA_ARGS+=(--extra "$OUT/nova_indic.jsonl")
fi

echo "Building v3 splits → $OUT"
python3 -m data.build_splits \
  --version v3 \
  --repo-root .. \
  --out "$OUT" \
  --train-size 28000 \
  "${EXTRA_ARGS[@]}"

echo "Done. Train with:"
echo "  export SM_CHANNEL_TRAIN=$PWD/$OUT"
echo "  export SM_MODEL_DIR=$PWD/train/output_v3"
echo "  python3 train/sft_qlora.py --config train/train_config.yaml"
