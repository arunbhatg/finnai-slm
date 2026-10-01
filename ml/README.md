# FinnAI SLM — ML toolkit

Training, data generation, eval, and export code for the on-device Indian bank SMS model.

**v3 ship path:** `Qwen/Qwen3-4B` SFT + DPO (`bash scripts/train_v3.sh`). Optional `RUN_TEACHER=1` trains `Qwen/Qwen3-8B` first. 1.7B remains a lite/distill SKU only — see [`docs/V3_PLAN.md`](../docs/V3_PLAN.md).

## Access

| Artifact | Link |
| --- | --- |
| **This code** | https://github.com/arunbhatg/finnai-slm (clone, then `cd ml`) |
| **Model weights** (bf16 + LoRA) | https://huggingface.co/finndot/finnai-slm-v2 |
| **Dataset** | https://huggingface.co/datasets/finndot/finnai-slm-data |
| **On-device INT4** (`.litertlm`) | [CloudFront download (~974 MB)](https://dgdzwh27431n8.cloudfront.net/models/qwen3-1.7b-finndot/latest/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm) |
| **Full guide** | [docs/FINETUNE_GUIDE.md](../docs/FINETUNE_GUIDE.md) |

```bash
git clone https://github.com/arunbhatg/finnai-slm.git
cd finnai-slm/ml
python -m pip install -r requirements.txt
huggingface-cli download finndot/finnai-slm-v2 --local-dir ../models/finnai-slm-v2
```

See the root [README.md](../README.md) for inference snippets, dataset rebuild, train, and eval commands.
