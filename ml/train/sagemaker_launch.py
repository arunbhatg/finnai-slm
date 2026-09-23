"""Launch FinnAI SLM SageMaker training in ap-south-1 (boto3, no sagemaker SDK)."""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

TRAIN_DIR = Path(__file__).resolve().parent
INCLUDE = {
    "sft_qlora.py",
    "train_config.yaml",
    "requirements.txt",
    "merge_lora.py",
}

# AWS Deep Learning Container (PyTorch training, GPU, ap-south-1)
DLC = (
    "763104351884.dkr.ecr.ap-south-1.amazonaws.com/"
    "pytorch-training:2.4.0-gpu-py311-cu121-ubuntu22.04-sagemaker"
)

FALLBACK_INSTANCES = ["ml.g5.2xlarge", "ml.g4dn.xlarge", "ml.g5.xlarge"]


def _tar_source() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in INCLUDE:
            path = TRAIN_DIR / name
            if path.exists():
                tar.add(path, arcname=name)
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN", ""))
    p.add_argument("--train-s3", required=True)
    p.add_argument("--output-s3", required=True)
    p.add_argument("--instance", default="ml.g5.2xlarge")
    p.add_argument("--region", default="ap-south-1")
    p.add_argument("--job-name", default="")
    p.add_argument("--bucket", default="", help="training bucket for code tarball")
    args = p.parse_args()
    if not args.role:
        raise SystemExit("Pass --role or SAGEMAKER_ROLE_ARN")

    region = args.region
    sm = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    train_s3 = args.train_s3.rstrip("/")
    # s3://bucket/prefix
    _, _, rest = train_s3.partition("s3://")
    bucket, _, prefix = rest.partition("/")
    code_key = "code/finai-slm/sourcedir.tar.gz"
    s3.put_object(Bucket=bucket, Key=code_key, Body=_tar_source(), ServerSideEncryption="AES256")
    code_s3 = f"s3://{bucket}/{code_key}"
    print("uploaded", code_s3)

    job = args.job_name or f"finnai-slm-{time.strftime('%Y%m%d-%H%M%S')}"
    hp = {
        "sagemaker_program": json.dumps("sft_qlora.py"),
        "sagemaker_submit_directory": json.dumps(code_s3),
        "sagemaker_region": json.dumps(region),
        "sagemaker_container_log_level": json.dumps(20),
        "config": json.dumps("train_config.yaml"),
    }
    instances = [args.instance] + [i for i in FALLBACK_INSTANCES if i != args.instance]
    last_err = None
    for instance in instances:
        try:
            sm.create_training_job(
                TrainingJobName=job,
                RoleArn=args.role,
                AlgorithmSpecification={
                    "TrainingImage": DLC,
                    "TrainingInputMode": "File",
                    "EnableSageMakerMetricsTimeSeries": True,
                },
                HyperParameters=hp,
                InputDataConfig=[
                    {
                        "ChannelName": "train",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": train_s3 + "/",
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                        "InputMode": "File",
                    }
                ],
                OutputDataConfig={"S3OutputPath": args.output_s3.rstrip("/")},
                ResourceConfig={
                    "InstanceType": instance,
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 80,
                },
                StoppingCondition={"MaxRuntimeInSeconds": 6 * 60 * 60},
                EnableNetworkIsolation=False,
                EnableInterContainerTrafficEncryption=False,
                EnableManagedSpotTraining=False,
                Tags=[
                    {"Key": "project", "Value": "finndot"},
                    {"Key": "model", "Value": "finnai-slm"},
                ],
            )
            print("training_job_name=", job)
            print("instance=", instance)
            print(
                "follow: aws sagemaker describe-training-job --training-job-name",
                job,
                "--region",
                region,
                "--query TrainingJobStatus",
            )
            return
        except ClientError as e:
            last_err = e
            code = e.response.get("Error", {}).get("Code", "")
            msg = e.response.get("Error", {}).get("Message", str(e))
            print(f"{instance} failed: {code} {msg}")
            if code in {"ResourceLimitExceeded", "InsufficientInstanceCapacity", "ValidationException"}:
                job = f"finnai-slm-{time.strftime('%Y%m%d-%H%M%S')}"
                continue
            raise
    raise SystemExit(f"Could not start training: {last_err}")


if __name__ == "__main__":
    main()
