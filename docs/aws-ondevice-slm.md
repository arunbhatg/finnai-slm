# On-device Qwen3 SLM (AWS)

Region: **ap-south-1**. Protocol: [llm-eval-protocol.md](llm-eval-protocol.md). Scripts: [`ml/`](../ml/). How we train: [finnai-slm-finetune.md](finnai-slm-finetune.md).

## 1. GPU quota

`g5.2xlarge` (8 vCPU, 32 GiB RAM, 1× A10G) is required for QLoRA + LiteRT export. `g5.xlarge` host RAM is too small.

```powershell
cd ml
python -m pip install boto3
python -m aws.request_gpu_quota --dry-run
python -m aws.request_gpu_quota
```

Or in the console: Service Quotas → EC2 “Running On-Demand G and VT instances” ≥ 8 vCPU, SageMaker `ml.g5.2xlarge` for training ≥ 1.

## 2. Deploy buckets + CloudFront

```powershell
cd infra
. .\scripts\load-aws-env.ps1
$env:AWS_DEFAULT_REGION = "ap-south-1"
$env:CDK_DEFAULT_REGION = "ap-south-1"
npx cdk deploy FinndotOnDeviceModelStack --require-approval never
```

Outputs:

| Output | Use |
| --- | --- |
| `TrainingBucketName` | `python -m data.upload_s3 --bucket ...` |
| `SageMakerRoleArn` | `SAGEMAKER_ROLE_ARN` |
| `OnDeviceModelUrl` | Android `MODEL_URL` **only after `SHIP: YES`** |
| `DeliveryBucketName` | `aws s3 cp model.litertlm s3://.../models/qwen3-1.7b-finndot/latest/` |

Until the fine-tune passes gates, the app downloads the public LiteRT Community Qwen3-1.7B INT4 file from Hugging Face (same runtime, not the FinnDot adapter).

## 3. Dataset → train → convert

```powershell
cd ml
python -m data.build_splits --repo-root .. --out data/out
python -m data.upload_s3 --bucket <TrainingBucketName> --prefix datasets/finai-slm/v1 --dir data/out
$env:SAGEMAKER_ROLE_ARN = "<SageMakerRoleArn>"
python -m train.sagemaker_launch --train-s3 s3://<TrainingBucketName>/datasets/finai-slm/v1 --output-s3 s3://<TrainingBucketName>/jobs/finai-slm/
python -m train.merge_lora --adapter <job>/adapter --out train/out/merged
python -m train.export_litertlm --checkpoint train/out/merged --out-dir train/out/litertlm
```

Fallback if SageMaker quota is 0: same Docker on spot `g5.2xlarge` (~$1.46/hr on-demand in Mumbai).

## 4. Eval then ship

```powershell
python -m eval.run_eval --sms data/out/test_sms.jsonl --chat eval/fixtures/chat_eval.jsonl --backend hf --model-id train/out/merged --baseline-id Qwen/Qwen3-1.7B --tag vYYYYMMDD
```

Copy `.litertlm` to the delivery prefix, invalidate CloudFront, set `Constants.ModelDownload` size + URL, only if the report says **SHIP: YES**.
