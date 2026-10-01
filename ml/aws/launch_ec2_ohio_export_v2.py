"""Launch Ohio g5.2xlarge to convert FinnAI v2 merged → INT4 .litertlm and ship to CloudFront origin."""

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
DELIVERY_BUCKET = "finndotondevicemodelstack-slmdeliverybucket3e573e9-2o9hxw1pny2m"
TRAIN_JOB = "finai-slm-v2-ohio-20260921-164442"
MODEL_KEY = "models/qwen3-1.7b-finndot/latest/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm"
PROFILE_NAME = "FinndotSlmEc2Profile"
SG_NAME = "finndot-slm-train-ohio"
INSTANCES = ["g5.2xlarge", "g5.xlarge", "g4dn.2xlarge"]
ML_ROOT = Path(__file__).resolve().parents[1]

USER_DATA = r"""#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/finnai-export.log) 2>&1
export AWS_DEFAULT_REGION=us-east-2
BUCKET=finndotondevicemodelstack-slmtrainingbucket1a75fa7-4mz3w5n387wo
DELIVERY=finndotondevicemodelstack-slmdeliverybucket3e573e9-2o9hxw1pny2m
TRAIN_JOB=finai-slm-v2-ohio-20260921-164442
EXPORT_JOB=EXPORTJOB
MODEL_KEY=models/qwen3-1.7b-finndot/latest/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm
WORKDIR=/opt/dlami/nvme/finnai-export
mkdir -p "$WORKDIR"
(
  while true; do
    aws s3 cp /var/log/finnai-export.log "s3://$BUCKET/jobs/finai-slm/$EXPORT_JOB/finnai-export.log" --region ap-south-1 || true
    sleep 60
  done
) &
LOG_PID=$!

if [ -x /opt/pytorch/bin/python ]; then PY=/opt/pytorch/bin/python; else PY=python3; fi
"$PY" -m pip install -U pip
"$PY" -m pip uninstall -y transformer-engine transformer_engine torchvision torchaudio || true
"$PY" -m pip install 'transformers>=4.51.0,<5' 'accelerate>=0.33' safetensors sentencepiece protobuf
# LiteRT Torch + pin torchao to a build that still imports on Torch 2.7
# (latest torchao needs Torch 2.11+ ScalingType APIs)
"$PY" -m pip install 'torchao>=0.9.0,<0.12' 'litert-torch==0.9.4' 'ai-edge-litert>=2.2.0,<2.3' 'ai-edge-quantizer==0.9.*' 'litert-lm-builder>=0.17'
# Re-pin torchao in case litert-torch pulled a Torchao that needs Torch 2.11+
"$PY" -m pip install --force-reinstall 'torchao>=0.9.0,<0.12'
# Prefer CLI export_hf (Qwen3 supported); Generative API as fallback

aws s3 sync "s3://$BUCKET/jobs/finai-slm/$TRAIN_JOB/merged" "$WORKDIR/merged" --region ap-south-1
aws s3 cp "s3://$BUCKET/code/finai-slm/exportdir-v2.tar.gz" /tmp/export.tgz --region ap-south-1
mkdir -p "$WORKDIR/code"
tar -xzf /tmp/export.tgz -C "$WORKDIR/code"
export HF_HOME="$WORKDIR/hf"
export PYTHONPATH="$WORKDIR/code"
OUT="$WORKDIR/litertlm"
mkdir -p "$OUT"

# Prefer CLI export_hf (Qwen3 supported); Generative API as fallback with nothink prefixes
set +e
"$PY" -m litert_torch.cli export_hf \
  --model "$WORKDIR/merged" \
  --output_dir "$OUT" \
  --quantization_recipe dynamic_wi4b32_afp32
CLI_RC=$?
if [ "$CLI_RC" -ne 0 ]; then
  echo "CLI export_hf failed (rc=$CLI_RC); trying Generative API"
  "$PY" "$WORKDIR/code/train/export_qwen3_litertlm.py" \
    --checkpoint "$WORKDIR/merged" \
    --out-dir "$OUT" \
    --quantize dynamic_int4_block32 \
    --kv-cache-max-len 1280 \
    --name-prefix qwen3_1.7b_finndot_nothink
  GEN_RC=$?
  if [ "$GEN_RC" -ne 0 ]; then
    echo "Generative export also failed (rc=$GEN_RC)"
    "$PY" -m litert_torch export_hf \
      --model "$WORKDIR/merged" \
      --output_dir "$OUT" \
      --quantize dynamic_int4_block32 || true
  fi
fi
set -e

# Find the .litertlm artifact
LITERT=$(find "$OUT" -name '*.litertlm' | head -1)
if [ -z "$LITERT" ]; then
  echo "ERROR: no .litertlm produced" >&2
  ls -laR "$OUT" || true
  kill "$LOG_PID" || true
  aws s3 cp /var/log/finnai-export.log "s3://$BUCKET/jobs/finai-slm/$EXPORT_JOB/finnai-export.log" --region ap-south-1 || true
  echo FAIL | aws s3 cp - "s3://$BUCKET/jobs/finai-slm/$EXPORT_JOB/FAIL" --region ap-south-1 || true
  shutdown -h now
  exit 1
fi
echo "Produced $LITERT size=$(stat -c%s "$LITERT")"
cp "$LITERT" "$OUT/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm"
stat -c%s "$OUT/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm" > "$OUT/SIZE_BYTES"
sha256sum "$OUT/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm" | awk '{print $1}' > "$OUT/SHA256"

# Ship to training job folder + CloudFront delivery origin
aws s3 sync "$OUT" "s3://$BUCKET/jobs/finai-slm/$EXPORT_JOB/litertlm/" --region ap-south-1
aws s3 cp "$OUT/qwen3_1.7b_finndot_nothink_q4_ekv1280.litertlm" \
  "s3://$DELIVERY/$MODEL_KEY" --region ap-south-1 \
  --content-type application/octet-stream \
  --cache-control "public, max-age=604800"
aws s3 cp "$OUT/SIZE_BYTES" "s3://$BUCKET/jobs/finai-slm/$EXPORT_JOB/SIZE_BYTES" --region ap-south-1
aws s3 cp "$OUT/SHA256" "s3://$BUCKET/jobs/finai-slm/$EXPORT_JOB/SHA256" --region ap-south-1

# CloudFront invalidation
aws cloudfront create-invalidation --distribution-id EB7YWQ3C4YF9L --paths "/models/*" || true

echo DONE | aws s3 cp - "s3://$BUCKET/jobs/finai-slm/$EXPORT_JOB/DONE" --region ap-south-1
kill "$LOG_PID" || true
aws s3 cp /var/log/finnai-export.log "s3://$BUCKET/jobs/finai-slm/$EXPORT_JOB/finnai-export.log" --region ap-south-1 || true
shutdown -h now
"""


def _tar() -> bytes:
    buf = io.BytesIO()
    src = ML_ROOT / "train" / "export_qwen3_litertlm.py"
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(src, arcname="train/export_qwen3_litertlm.py")
        for pkg in ("train",):
            info = tarfile.TarInfo(name=f"{pkg}/__init__.py")
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))
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
    job = f"finnai-export-v2-ohio-{stamp}"
    s3 = boto3.client("s3", region_name=DATA_REGION)
    ec2 = boto3.client("ec2", region_name=REGION)
    s3.put_object(
        Bucket=BUCKET,
        Key="code/finai-slm/exportdir-v2.tar.gz",
        Body=_tar(),
        ServerSideEncryption="AES256",
    )
    print("uploaded exportdir-v2")
    ami = latest_dlami(ec2)
    sg = ensure_sg(ec2)
    user_data = USER_DATA.replace("EXPORTJOB", job)
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
            print(f"logs s3://{BUCKET}/jobs/finai-slm/{job}/finnai-export.log")
            print(f"delivery key s3://{DELIVERY_BUCKET}/{MODEL_KEY}")
            return
        except ClientError as e:
            last = e
            print(instance, e.response["Error"]["Code"], e.response["Error"]["Message"])
    raise SystemExit(last)


if __name__ == "__main__":
    main()
