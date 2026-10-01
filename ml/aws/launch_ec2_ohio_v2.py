"""Launch FinnAI SLM v2 QLoRA on G instance in us-east-2.

v2 differences vs v1:
- Dataset includes Nova Pro Indic expansion (nova_indic.jsonl)
- train-size 16,000 (v1 was 12,000)
- Starts from Qwen3-1.7B base (fresh QLoRA, not resume from v1 adapter)
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
DATA_PREFIX = "datasets/finai-slm/v2"
CODE_KEY = "code/finai-slm-v2/sourcedir.tar.gz"
OUT_PREFIX = "jobs/finai-slm"
ROLE_NAME = "FinndotOnDeviceModelStack-SlmSageMakerRole086F6E1A-ySAeAOkiYao9"
PROFILE_NAME = "FinndotSlmEc2Profile"
SG_NAME = "finndot-slm-train-ohio"
INSTANCES = ["g5.2xlarge", "g5.xlarge", "g6.2xlarge", "g4dn.2xlarge", "g4dn.xlarge"]
TRAIN_DIR = Path(__file__).resolve().parents[1] / "train"
INCLUDE = {"sft_qlora.py", "train_config.yaml", "requirements.txt", "merge_lora.py"}

USER_DATA = r"""#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/finnai-train.log) 2>&1
export AWS_DEFAULT_REGION=us-east-2
BUCKET=finndotondevicemodelstack-slmtrainingbucket1a75fa7-4mz3w5n387wo
JOB=finnai-slm-v2-ohio-JOBSTAMP
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

aws s3 sync "s3://$BUCKET/datasets/finai-slm/v2" "$WORKDIR/data" --region ap-south-1
aws s3 cp "s3://$BUCKET/code/finai-slm-v2/sourcedir.tar.gz" /tmp/src.tgz --region ap-south-1
mkdir -p "$WORKDIR/code"
tar -xzf /tmp/src.tgz -C "$WORKDIR/code"
"$PY" -m pip install -U pip
"$PY" -m pip uninstall -y transformer-engine transformer_engine transformer-engine-cu12 transformer_engine_torch torchvision torchaudio || true
"$PY" -m pip install -r "$WORKDIR/code/requirements.txt"
nvidia-smi || true
export SM_CHANNEL_TRAIN="$WORKDIR/data"
export SM_MODEL_DIR="$WORKDIR/out"
export HF_HOME="$WORKDIR/hf"
cd "$WORKDIR/code"
"$PY" sft_qlora.py --config train_config.yaml
"$PY" merge_lora.py --adapter "$WORKDIR/out/adapter" --out "$WORKDIR/out/merged"
aws s3 sync "$WORKDIR/out" "s3://$BUCKET/jobs/finai-slm/$JOB/" --region ap-south-1
echo DONE | aws s3 cp - "s3://$BUCKET/jobs/finai-slm/$JOB/DONE" --region ap-south-1
kill "$LOG_PID" || true
aws s3 cp /var/log/finnai-train.log "s3://$BUCKET/jobs/finai-slm/$JOB/finnai-train.log" --region ap-south-1 || true
shutdown -h now
"""


def _tar_source() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in INCLUDE:
            path = TRAIN_DIR / name
            if path.exists():
                tar.add(path, arcname=name)
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


def main() -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job = f"finnai-slm-v2-ohio-{stamp}"
    iam = boto3.client("iam")
    s3 = boto3.client("s3", region_name=DATA_REGION)
    ec2 = boto3.client("ec2", region_name=REGION)

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
                            "VolumeSize": 150,
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
            print("launched", iid, instance, "job", job)
            print(f"logs: s3://{BUCKET}/{OUT_PREFIX}/{job}/finnai-train.log")
            return
        except ClientError as e:
            last_err = e
            print(instance, e.response["Error"]["Code"], e.response["Error"]["Message"])
    raise SystemExit(last_err)


if __name__ == "__main__":
    main()
