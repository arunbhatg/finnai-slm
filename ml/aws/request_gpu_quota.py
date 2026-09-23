"""Request GPU quota in ap-south-1. Run with company AWS creds."""

from __future__ import annotations

import argparse
import json

import boto3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="ap-south-1")
    p.add_argument("--desired-g-vcpu", type=int, default=8, help="g5.2xlarge is 8 vCPU")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    sq = boto3.client("service-quotas", region_name=args.region)
    print("Listing SageMaker + EC2 GPU quotas (filter G5)...")

    found = []
    for service in ("sagemaker", "ec2"):
        paginator = sq.get_paginator("list_service_quotas")
        for page in paginator.paginate(ServiceCode=service):
            for q in page.get("Quotas", []):
                name = q.get("QuotaName", "")
                if "g5.2xlarge" in name.lower() or (
                    service == "ec2" and "Running On-Demand G" in name
                ):
                    found.append(
                        {
                            "service": service,
                            "code": q["QuotaCode"],
                            "name": name,
                            "value": q.get("Value"),
                            "arn": q.get("QuotaArn"),
                        }
                    )
    print(json.dumps(found, indent=2))
    if args.dry_run:
        print("Dry run: not requesting increases")
        return
    for q in found:
        if q["service"] == "ec2" and "Running On-Demand G" in q["name"]:
            if (q["value"] or 0) >= args.desired_g_vcpu:
                print("EC2 G quota already", q["value"])
                continue
            resp = sq.request_service_quota_increase(
                ServiceCode="ec2",
                QuotaCode=q["code"],
                DesiredValue=float(args.desired_g_vcpu),
            )
            print("Requested EC2:", resp["RequestedQuota"]["Status"])
        if q["service"] == "sagemaker" and "g5.2xlarge" in q["name"].lower() and "training" in q["name"].lower():
            if (q["value"] or 0) >= 1:
                print("SageMaker training quota already", q["value"])
                continue
            resp = sq.request_service_quota_increase(
                ServiceCode="sagemaker",
                QuotaCode=q["code"],
                DesiredValue=1.0,
            )
            print("Requested SageMaker:", resp["RequestedQuota"]["Status"])


if __name__ == "__main__":
    main()
