"""When SageMaker G5 quota is approved, start FinnAI SLM training."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import boto3

QUOTA = "L-2D6DEB3C"  # ml.g5.2xlarge for training job usage
TRAIN_DIR = Path(__file__).resolve().parents[1]


def quota_value(region: str) -> float:
    c = boto3.client("service-quotas", region_name=region)
    q = c.get_service_quota(ServiceCode="sagemaker", QuotaCode=QUOTA)
    return float(q["Quota"]["Value"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="ap-south-1")
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--max-hours", type=float, default=48)
    p.add_argument("--role", required=True)
    p.add_argument("--train-s3", required=True)
    p.add_argument("--output-s3", required=True)
    args = p.parse_args()

    deadline = time.time() + args.max_hours * 3600
    while time.time() < deadline:
        val = quota_value(args.region)
        print(f"ml.g5.2xlarge training quota={val}", flush=True)
        if val >= 1:
            cmd = [
                sys.executable,
                "-m",
                "train.sagemaker_launch",
                "--role",
                args.role,
                "--train-s3",
                args.train_s3,
                "--output-s3",
                args.output_s3,
                "--instance",
                "ml.g5.2xlarge",
                "--region",
                args.region,
            ]
            print("launching", cmd, flush=True)
            raise SystemExit(subprocess.call(cmd, cwd=str(TRAIN_DIR)))
        time.sleep(args.poll_seconds)
    raise SystemExit("timed out waiting for GPU quota")


if __name__ == "__main__":
    main()
