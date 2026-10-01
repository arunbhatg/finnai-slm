"""Upload dataset jsonl to the private training bucket."""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="datasets/v1")
    p.add_argument("--dir", type=Path, default=Path("data/out"))
    p.add_argument("--region", default="ap-south-1")
    args = p.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    for path in sorted(args.dir.iterdir()):
        if path.suffix not in {".jsonl", ".txt", ".json"} and path.name != "SHA256SUMS":
            continue
        key = f"{args.prefix.rstrip('/')}/{path.name}"
        extra = {"ServerSideEncryption": "AES256"}
        s3.upload_file(str(path), args.bucket, key, ExtraArgs=extra)
        print(f"s3://{args.bucket}/{key}")


if __name__ == "__main__":
    main()
