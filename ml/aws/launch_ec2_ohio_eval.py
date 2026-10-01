"""Launch Ohio g5.2xlarge to merge best-val adapter and run held-out HF eval."""

from __future__ import annotations

import io
import tarfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-2"
DATA_REGION = "ap-south-1"
BUCKET = "finndotondevicemodelstack-slmtrainingbucket1a75fa7-4mz3w5n387wo"
TRAIN_JOB = "finai-slm-ohio-20260920-091315"
# Epoch 2 had best val SMS R-EM (0.9648)
BEST_CKPT = "checkpoints/checkpoint-1500"
PROFILE_NAME = "FinndotSlmEc2Profile"
SG_NAME = "finndot-slm-train-ohio"
INSTANCES = ["g5.2xlarge", "g5.xlarge", "g4dn.2xlarge"]
ML_ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    ("eval/run_eval.py", "eval/run_eval.py"),
    ("eval/metrics.py", "eval/metrics.py"),
    ("eval/report.py", "eval/report.py"),
    ("train/merge_lora.py", "train/merge_lora.py"),
    ("train/requirements.txt", "requirements.txt"),
]

USER_DATA = r"""#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/finnai-eval.log) 2>&1
export AWS_DEFAULT_REGION=us-east-2
BUCKET=finndotondevicemodelstack-slmtrainingbucket1a75fa7-4mz3w5n387wo
TRAIN_JOB=finai-slm-ohio-20260920-091315
EVAL_JOB=EVALJOB
WORKDIR=/opt/dlami/nvme/finnai-eval
mkdir -p "$WORKDIR"
(
  while true; do
    aws s3 cp /var/log/finnai-eval.log "s3://$BUCKET/jobs/finai-slm/$EVAL_JOB/finnai-eval.log" --region ap-south-1 || true
    sleep 60
  done
) &
LOG_PID=$!

if [ -x /opt/pytorch/bin/python ]; then PY=/opt/pytorch/bin/python; else PY=python3; fi
"$PY" -m pip install -U pip
"$PY" -m pip uninstall -y transformer-engine transformer_engine torchvision torchaudio || true
aws s3 sync "s3://$BUCKET/datasets/finai-slm/v1" "$WORKDIR/data" --region ap-south-1
aws s3 cp "s3://$BUCKET/code/finai-slm/evaldir.tar.gz" /tmp/eval.tgz --region ap-south-1
mkdir -p "$WORKDIR/code"
tar -xzf /tmp/eval.tgz -C "$WORKDIR/code"
"$PY" -m pip install -r "$WORKDIR/code/requirements.txt"
export HF_HOME="$WORKDIR/hf"
export PYTHONPATH="$WORKDIR/code"

# Remerge best-val adapter (checkpoint-1500)
aws s3 sync "s3://$BUCKET/jobs/finai-slm/$TRAIN_JOB/checkpoints/checkpoint-1500" "$WORKDIR/adapter-best" --region ap-south-1
"$PY" "$WORKDIR/code/train/merge_lora.py" --base Qwen/Qwen3-1.7B --adapter "$WORKDIR/adapter-best" --out "$WORKDIR/merged-best"
aws s3 sync "$WORKDIR/merged-best" "s3://$BUCKET/jobs/finai-slm/$TRAIN_JOB/merged-best/" --region ap-south-1

cd "$WORKDIR/code"
"$PY" -m eval.run_eval \
  --backend hf \
  --sms "$WORKDIR/data/test_sms.jsonl" \
  --chat "$WORKDIR/data/chat_eval.jsonl" \
  --model-id "$WORKDIR/merged-best" \
  --baseline-id Qwen/Qwen3-1.7B \
  --product-id Qwen/Qwen2.5-1.5B-Instruct \
  --out "$WORKDIR/reports" \
  --tag v20260921-finnai-slm \
  --job-id "$TRAIN_JOB"

aws s3 sync "$WORKDIR/reports" "s3://$BUCKET/jobs/finai-slm/$EVAL_JOB/reports/" --region ap-south-1
echo DONE | aws s3 cp - "s3://$BUCKET/jobs/finai-slm/$EVAL_JOB/DONE" --region ap-south-1
kill "$LOG_PID" || true
aws s3 cp /var/log/finnai-eval.log "s3://$BUCKET/jobs/finai-slm/$EVAL_JOB/finnai-eval.log" --region ap-south-1 || true
shutdown -h now
"""


def _tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for src, arc in INCLUDE:
            path = ML_ROOT / src
            tar.add(path, arcname=arc)
        # empty package markers
        for pkg in ("eval", "train"):
            init = (ML_ROOT / pkg / "__init__.py")
            if init.exists():
                tar.add(init, arcname=f"{pkg}/__init__.py")
            else:
                info = tarfile.TarInfo(name=f"{pkg}/__init__.py")
                data = b""
                info.size = 0
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def latest_dlami(ec2) -> str:
    images = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {
                "Name": "name",
                "Values": ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04)*"],
            },
            {"Name": "state", "Values": ["available"]},
        ],
    )["Images"]
    images.sort(key=lambda i: i["CreationDate"], reverse=True)
    print("AMI", images[0]["ImageId"], images[0]["Name"])
    return images[0]["ImageId"]


def pick_subnet(ec2, instance: str) -> str:
    offers = ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [instance]}],
    )["InstanceTypeOfferings"]
    azs = {o["Location"] for o in offers}
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc]}])["Subnets"]
    for s in subnets:
        if s["AvailabilityZone"] in azs and s.get("MapPublicIpOnLaunch"):
            return s["SubnetId"]
    for s in subnets:
        if s["AvailabilityZone"] in azs:
            return s["SubnetId"]
    raise SystemExit(f"no subnet for {instance}")


def ensure_sg(ec2) -> str:
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    existing = ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [SG_NAME]},
            {"Name": "vpc-id", "Values": [vpc]},
        ]
    )["SecurityGroups"]
    if existing:
        return existing[0]["GroupId"]
    return ec2.create_security_group(
        GroupName=SG_NAME,
        Description="FinnAI SLM train egress only",
        VpcId=vpc,
    )["GroupId"]


def main() -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job = f"finnai-eval-ohio-{stamp}"
    s3 = boto3.client("s3", region_name=DATA_REGION)
    ec2 = boto3.client("ec2", region_name=REGION)
    s3.put_object(
        Bucket=BUCKET,
        Key="code/finai-slm/evaldir.tar.gz",
        Body=_tar(),
        ServerSideEncryption="AES256",
    )
    print("uploaded evaldir")
    ami = latest_dlami(ec2)
    sg = ensure_sg(ec2)
    user_data = USER_DATA.replace("EVALJOB", job)
    last = None
    for instance in INSTANCES:
        try:
            subnet = pick_subnet(ec2, instance)
            resp = ec2.run_instances(
                ImageId=ami,
                InstanceType=instance,
                MinCount=1,
                MaxCount=1,
                NetworkInterfaces=[
                    {
                        "DeviceIndex": 0,
                        "SubnetId": subnet,
                        "Groups": [sg],
                        "AssociatePublicIpAddress": True,
                    }
                ],
                IamInstanceProfile={"Name": PROFILE_NAME},
                UserData=user_data,
                InstanceInitiatedShutdownBehavior="terminate",
                BlockDeviceMappings=[
                    {
                        "DeviceName": "/dev/sda1",
                        "Ebs": {
                            "VolumeSize": 200,
                            "VolumeType": "gp3",
                            "DeleteOnTermination": True,
                        },
                    }
                ],
                MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": job},
                            {"Key": "Project", "Value": "Finndot"},
                            {"Key": "Job", "Value": job},
                        ],
                    }
                ],
            )
            iid = resp["Instances"][0]["InstanceId"]
            print("launched", iid, instance, job)
            print(f"logs s3://{BUCKET}/jobs/finai-slm/{job}/finnai-eval.log")
            return
        except ClientError as e:
            last = e
            print(instance, e.response["Error"]["Code"], e.response["Error"]["Message"])
    raise SystemExit(last)


if __name__ == "__main__":
    main()
