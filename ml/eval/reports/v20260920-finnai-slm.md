# FinnAI SLM training status (2026-09-20)

**SHIP: NO** — GPU fine-tune has not produced weights yet.

## Blocker

AWS account `008692857726` has **0** `ml.g5.2xlarge` (and G4dn) SageMaker training quota, and **0** EC2 On-Demand G/VT vCPU, in ap-south-1 and the US/EU regions we checked. `CreateTrainingJob` returned `ResourceLimitExceeded`.

Quota increase for SageMaker `ml.g5.2xlarge for training job usage` is **PENDING** (`e646f3a049cd4b1697ef88d24fc7db68eBxZZBO9`).

## Ready

- Recipe: `docs/finnai-slm-finetune.md`
- Dataset on S3: `s3://finndotondevicemodelstack-slmtrainingbucket1a75fa7-4mz3w5n387wo/datasets/finnai-slm/v1/`
- Code tarball: `s3://finndotondevicemodelstack-slmtrainingbucket1a75fa7-4mz3w5n387wo/code/finnai-slm/sourcedir.tar.gz`
- Stack: `FinndotOnDeviceModelStack`

Re-run `python -m train.sagemaker_launch ...` after the quota is approved. Do not publish Hugging Face weights until eval SHIP: YES.
