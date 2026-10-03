"""Launch FinnAI SLM v3 curriculum QLoRA on GPU EC2 in us-east-2.

v3 vs v2:
- Base: Qwen3-4B (not 1.7B)
- Dataset: v3 28k + curriculum stage files
- Training: SMS anchor → blend → coach harden → DPO → SMS replay
"""

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
DATA_PREFIX = "datasets/finai-slm/v3"
CODE_KEY = "code/finai-slm-v3/sourcedir.tar.gz"
OUT_PREFIX = "jobs/finai-slm"
ROLE_NAME = "FinndotOnDeviceModelStack-SlmSageMakerRole086F6E1A-ySAeAOkiYao9"
PROFILE_NAME = "FinndotSlmEc2Profile"
SG_NAME = "finndot-slm-train-ohio"
# Prefer more VRAM for 4B + merge; fall back like v2
INSTANCES = ["g5.2xlarge", "g6.2xlarge", "g5.4xlarge", "g5.xlarge", "g4dn.2xlarge"]
ML_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = ML_DIR / "train"
DATA_DIR = ML_DIR / "data"
EVAL_DIR = ML_DIR / "eval"

# Files packed into the instance code tarball
INCLUDE_TRAIN = {
    "sft_qlora.py",
    "train_dpo.py",
    "merge_lora.py",
    "train_config_v3_4b.yaml",
    "train_config.yaml",
    "requirements.txt",
    "requirements-train.txt",
}
INCLUDE_DATA_PY = {
    "tag_curriculum.py",
    "generate_dpo_prefs.py",
    "generate_onpolicy_prefs.py",
    "generate_finance_chat.py",
    "schema.py",
    "sms_prompt.py",
    "__init__.py",
}
INCLUDE_EVAL_PY = {"metrics.py", "__init__.py"}

USER_DATA = r"""#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/finnai-train.log) 2>&1
export AWS_DEFAULT_REGION=us-east-2
BUCKET=finndotondevicemodelstack-slmtrainingbucket1a75fa7-4mz3w5n387wo
JOB=finnai-slm-v3-ohio-JOBSTAMP
WORKDIR=/opt/finnai
if [ -d /opt/dlami/nvme ]; then WORKDIR=/opt/dlami/nvme/finnai; fi
mkdir -p "$WORKDIR" /tmp/finnai-logs
(
  while true; do
    aws s3 cp /var/log/finnai-train.log "s3://$BUCKET/jobs/finai-slm/$JOB/finnai-train.log" --region ap-south-1 || true
    sleep 60
  done
) &
LOG_PID=$!

if [ -x /opt/pytorch/bin/python ]; then PY=/opt/pytorch/bin/python
elif [ -x /opt/conda/bin/python ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate pytorch 2>/dev/null || true
  PY=python
else PY=python3
fi

aws s3 sync "s3://$BUCKET/datasets/finai-slm/v3" "$WORKDIR/data" --region ap-south-1
aws s3 cp "s3://$BUCKET/code/finai-slm-v3/sourcedir.tar.gz" /tmp/src.tgz --region ap-south-1
mkdir -p "$WORKDIR/code"
tar -xzf /tmp/src.tgz -C "$WORKDIR/code"
"$PY" -m pip install -U pip
"$PY" -m pip uninstall -y transformer-engine transformer_engine transformer-engine-cu12 transformer_engine_torch torchvision torchaudio || true
"$PY" -m pip install -r "$WORKDIR/code/requirements.txt"
# DPO extras
"$PY" -m pip install "trl>=0.9.0" "peft>=0.11.0" "bitsandbytes>=0.43.0" || true
nvidia-smi || true

export HF_HOME="$WORKDIR/hf"
export TRANSFORMERS_CACHE="$WORKDIR/hf"
cd "$WORKDIR/code"

DATA="$WORKDIR/data"
BASE="$WORKDIR/out"
CONFIG=train_config_v3_4b.yaml
mkdir -p "$BASE"

# Curriculum stage files should already be on S3; rebuild tags if missing
if [ ! -f "$DATA/train_stage1.jsonl" ]; then
  PYTHONPATH="$WORKDIR/code" "$PY" -m data.tag_curriculum --train "$DATA/train.jsonl" --out-dir "$DATA"
fi
if [ ! -f "$DATA/dpo_rule_prefs.jsonl" ]; then
  PYTHONPATH="$WORKDIR/code" "$PY" -m data.generate_dpo_prefs --out "$DATA/dpo_rule_prefs.jsonl" --n 2000
fi

S1="$BASE/stage1"
S2="$BASE/stage2"
S3="$BASE/stage3"
S4="$BASE/stage4_dpo"
S5="$BASE/stage5_replay"

echo "== Stage 1 SMS anchor =="
export SM_CHANNEL_TRAIN="$DATA"
export SM_MODEL_DIR="$S1"
"$PY" sft_qlora.py --config "$CONFIG" \
  --train-file "$DATA/train_stage1.jsonl" --val-file "$DATA/val.jsonl" \
  --epochs 2 --lr 1.5e-4
aws s3 sync "$S1" "s3://$BUCKET/jobs/finai-slm/$JOB/stage1/" --region ap-south-1 || true

echo "== Stage 2 blend =="
export SM_MODEL_DIR="$S2"
"$PY" sft_qlora.py --config "$CONFIG" \
  --adapter "$S1/adapter" \
  --train-file "$DATA/train_stage2.jsonl" --val-file "$DATA/val.jsonl" \
  --epochs 2 --lr 1.0e-4
aws s3 sync "$S2" "s3://$BUCKET/jobs/finai-slm/$JOB/stage2/" --region ap-south-1 || true

echo "== Stage 3 coach harden =="
export SM_MODEL_DIR="$S3"
"$PY" sft_qlora.py --config "$CONFIG" \
  --adapter "$S2/adapter" \
  --train-file "$DATA/train_stage3.jsonl" --val-file "$DATA/val.jsonl" \
  --epochs 1 --lr 5.0e-5
aws s3 sync "$S3" "s3://$BUCKET/jobs/finai-slm/$JOB/stage3/" --region ap-south-1 || true

echo "== Merge stage3 =="
"$PY" merge_lora.py --base Qwen/Qwen3-4B --adapter "$S3/adapter" --out "$S3/merged"

echo "== On-policy prefs (best-effort) =="
PREF="$DATA/dpo_rule_prefs.jsonl"
set +e
PYTHONPATH="$WORKDIR/code" "$PY" -m data.generate_onpolicy_prefs \
  --seeds "$DATA/train_tagged.jsonl" \
  --model-id "$S3/merged" \
  --out "$DATA/dpo_onpolicy.jsonl" \
  --n 1500
ONPOLICY=$?
set -e
if [ "$ONPOLICY" -eq 0 ] && [ -f "$DATA/dpo_onpolicy.jsonl" ]; then
  cat "$DATA/dpo_rule_prefs.jsonl" "$DATA/dpo_onpolicy.jsonl" > "$DATA/dpo_prefs.jsonl"
  PREF="$DATA/dpo_prefs.jsonl"
fi

echo "== Stage 4 DPO =="
"$PY" train_dpo.py --config "$CONFIG" --prefs "$PREF" \
  --adapter "$S3/adapter" --output-dir "$S4" --epochs 1 --lr 5e-5 --beta 0.1
aws s3 sync "$S4" "s3://$BUCKET/jobs/finai-slm/$JOB/stage4_dpo/" --region ap-south-1 || true

echo "== Stage 5 SMS replay =="
export SM_MODEL_DIR="$S5"
"$PY" sft_qlora.py --config "$CONFIG" \
  --adapter "$S4/adapter" \
  --train-file "$DATA/train_stage5.jsonl" --val-file "$DATA/val.jsonl" \
  --epochs 0.5 --lr 3.0e-5

echo "== Final merge =="
"$PY" merge_lora.py --base Qwen/Qwen3-4B --adapter "$S5/adapter" --out "$S5/merged"
aws s3 sync "$BASE" "s3://$BUCKET/jobs/finai-slm/$JOB/" --region ap-south-1
echo DONE | aws s3 cp - "s3://$BUCKET/jobs/finai-slm/$JOB/DONE" --region ap-south-1
kill "$LOG_PID" || true
aws s3 cp /var/log/finnai-train.log "s3://$BUCKET/jobs/finai-slm/$JOB/finnai-train.log" --region ap-south-1 || true
shutdown -h now
"""


def _tar_source() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in INCLUDE_TRAIN:
            path = TRAIN_DIR / name
            if path.exists():
                tar.add(path, arcname=name)
        # package data/ and eval/ helpers for on-policy prefs + curriculum
        for name in INCLUDE_DATA_PY:
            path = DATA_DIR / name
            if path.exists():
                tar.add(path, arcname=f"data/{name}")
        for name in INCLUDE_EVAL_PY:
            path = EVAL_DIR / name
            if path.exists():
                tar.add(path, arcname=f"eval/{name}")
        # empty package markers
        for pkg in ("data", "eval"):
            init = ML_DIR / pkg / "__init__.py"
            if init.exists():
                tar.add(init, arcname=f"{pkg}/__init__.py")
    return buf.getvalue()


def ensure_instance_profile(iam) -> str:
    role_arn = f"arn:aws:iam::008692857726:role/{ROLE_NAME}"
    try:
        iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        iam.create_instance_profile(InstanceProfileName=PROFILE_NAME)
        iam.add_role_to_instance_profile(InstanceProfileName=PROFILE_NAME, RoleName=ROLE_NAME)
        time.sleep(12)
    attached = iam.list_attached_role_policies(RoleName=ROLE_NAME)["AttachedPolicies"]
    if not any(p["PolicyName"] == "AmazonSSMManagedInstanceCore" for p in attached):
        iam.attach_role_policy(
            RoleName=ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
        )
    return role_arn


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
    if not images:
        raise SystemExit("no Deep Learning PyTorch GPU AMI in us-east-2")
    images.sort(key=lambda i: i["CreationDate"], reverse=True)
    ami = images[0]
    print("AMI", ami["ImageId"], ami["Name"])
    return ami["ImageId"]


def pick_subnet(ec2, instance: str) -> str:
    offers = ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [instance]}],
    )["InstanceTypeOfferings"]
    azs = {o["Location"] for o in offers}
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise SystemExit("no default VPC in us-east-2")
    vpc_id = vpcs[0]["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    for s in subnets:
        if s["AvailabilityZone"] in azs and s["MapPublicIpOnLaunch"]:
            return s["SubnetId"]
    for s in subnets:
        if s["AvailabilityZone"] in azs:
            return s["SubnetId"]
    raise SystemExit(f"no subnet for {instance} in {sorted(azs)}")


def ensure_sg(ec2) -> str:
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    vpc_id = vpcs[0]["VpcId"]
    existing = ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [SG_NAME]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    )["SecurityGroups"]
    if existing:
        return existing[0]["GroupId"]
    sg = ec2.create_security_group(
        GroupName=SG_NAME,
        Description="FinnAI SLM train egress only",
        VpcId=vpc_id,
    )
    return sg["GroupId"]


def upload_dataset(s3, local: Path) -> None:
    if not local.exists():
        raise SystemExit(f"missing dataset dir {local} — run scripts/build_v3_data.sh + tag_curriculum first")
    for path in sorted(local.rglob("*")):
        if path.is_file():
            key = f"{DATA_PREFIX}/{path.relative_to(local).as_posix()}"
            print("upload", key)
            s3.upload_file(str(path), BUCKET, key, ExtraArgs={"ServerSideEncryption": "AES256"})


def main() -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job = f"finnai-slm-v3-ohio-{stamp}"
    iam = boto3.client("iam")
    s3 = boto3.client("s3", region_name=DATA_REGION)
    ec2 = boto3.client("ec2", region_name=REGION)

    data_local = ML_DIR / "data" / "out_v3"
    # Ensure curriculum files exist
    stage1 = data_local / "train_stage1.jsonl"
    if not stage1.exists():
        raise SystemExit(
            "Curriculum files missing. From ml/: "
            "python3 -m data.tag_curriculum --train data/out_v3/train.jsonl --out-dir data/out_v3"
        )

    print("uploading v3 dataset…")
    upload_dataset(s3, data_local)
    s3.put_object(Bucket=BUCKET, Key=CODE_KEY, Body=_tar_source(), ServerSideEncryption="AES256")
    print("uploaded", f"s3://{BUCKET}/{CODE_KEY}")

    ensure_instance_profile(iam)
    ami = latest_dlami(ec2)
    sg = ensure_sg(ec2)
    user_data = USER_DATA.replace("JOBSTAMP", stamp)

    last_err = None
    for instance in INSTANCES:
        try:
            subnet = pick_subnet(ec2, instance)
            print("trying", instance, "subnet", subnet)
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
                            "VolumeSize": 250,
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
                            {"Key": "Version", "Value": "v3"},
                        ],
                    }
                ],
            )
            iid = resp["Instances"][0]["InstanceId"]
            print("launched", iid, instance, "job", job)
            print(f"logs: s3://{BUCKET}/{OUT_PREFIX}/{job}/finnai-train.log")
            print(f"artifacts: s3://{BUCKET}/{OUT_PREFIX}/{job}/")
            (ML_DIR / "aws" / "last_v3_job.txt").write_text(
                f"{job}\n{iid}\n{instance}\n", encoding="utf-8"
            )
            return
        except ClientError as e:
            last_err = e
            print(instance, e.response["Error"]["Code"], e.response["Error"]["Message"])
    raise SystemExit(last_err)


if __name__ == "__main__":
    main()
