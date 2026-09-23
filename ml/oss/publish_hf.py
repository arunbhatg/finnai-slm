"""Publish FinnAI SLM v2 to Hugging Face (model + dataset + guide).

Requires HF_TOKEN or `hf auth login`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="finndot/finnai-slm-v2")
    p.add_argument("--dataset-repo", default="finndot/finnai-slm-data")
    p.add_argument("--merged", type=Path, default=None, help="Merged bf16 folder")
    p.add_argument("--adapter", type=Path, default=None, help="Optional LoRA adapter folder")
    p.add_argument("--data-dir", type=Path, default=None, help="Dataset jsonl dir (no nova_indic)")
    p.add_argument("--card", type=Path, default=Path(__file__).resolve().parents[1] / "release" / "MODEL_CARD.md")
    p.add_argument("--dataset-card", type=Path, default=Path(__file__).resolve().parents[1] / "release" / "DATASET_CARD.md")
    p.add_argument("--guide", type=Path, default=Path(__file__).resolve().parents[1] / "release" / "FINETUNE_GUIDE.md")
    p.add_argument("--eval-report", type=Path, default=Path(__file__).resolve().parents[1] / "eval" / "reports" / "v20260922-finnai-slm-v2.md")
    p.add_argument("--litertlm", type=Path, default=None)
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-model", action="store_true")
    p.add_argument("--skip-dataset", action="store_true")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not args.dry_run and not token:
        raise SystemExit("Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) before publishing")

    if args.dry_run:
        print("Would publish model →", args.repo, "from", args.merged)
        print("Would publish dataset →", args.dataset_repo, "from", args.data_dir)
        if args.litertlm:
            print("Would publish litertlm →", args.repo + "-litertlm")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    if not args.skip_model and args.merged and args.merged.exists():
        api.create_repo(args.repo, exist_ok=True, private=args.private, repo_type="model")
        print("uploading model folder", args.merged)
        api.upload_folder(
            folder_path=str(args.merged),
            repo_id=args.repo,
            repo_type="model",
            ignore_patterns=["*.pt", "optimizer*", "rng_state*", "scheduler*"],
        )
        if args.adapter and args.adapter.exists():
            api.upload_folder(
                folder_path=str(args.adapter),
                path_in_repo="adapter",
                repo_id=args.repo,
                repo_type="model",
            )
        for local, remote in [
            (args.card, "README.md"),
            (args.guide, "FINETUNE_GUIDE.md"),
            (args.eval_report, "eval/v20260922-finnai-slm-v2.md"),
        ]:
            if local and local.exists():
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=remote,
                    repo_id=args.repo,
                    repo_type="model",
                )
        print("published model", args.repo)

    if args.litertlm and args.litertlm.exists():
        litert_repo = args.repo + "-litertlm"
        api.create_repo(litert_repo, exist_ok=True, private=args.private, repo_type="model")
        api.upload_file(
            path_or_fileobj=str(args.litertlm),
            path_in_repo=args.litertlm.name,
            repo_id=litert_repo,
            repo_type="model",
        )
        if args.card.exists():
            api.upload_file(
                path_or_fileobj=str(args.card),
                path_in_repo="README.md",
                repo_id=litert_repo,
                repo_type="model",
            )
        print("published litertlm", litert_repo)

    if not args.skip_dataset and args.data_dir and args.data_dir.exists():
        api.create_repo(
            args.dataset_repo, exist_ok=True, private=args.private, repo_type="dataset"
        )
        # Exclude Nova surface text from public dump (AWS service terms)
        staging = args.data_dir / "_hf_staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        allow = {
            "train.jsonl",
            "val.jsonl",
            "test.jsonl",
            "test_sms.jsonl",
            "val_sms.jsonl",
            "chat_eval.jsonl",
            "counts.json",
            "SHA256SUMS",
            "test_sms_ids.txt",
        }
        for name in allow:
            src = args.data_dir / name
            if src.exists():
                shutil.copy2(src, staging / name)
        # Recompute SHA256SUMS without nova_indic if present
        sums = staging / "SHA256SUMS"
        if sums.exists():
            lines = [
                ln
                for ln in sums.read_text(encoding="utf-8").splitlines()
                if "nova_indic" not in ln
            ]
            sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
        meta = {
            "description": "Synthetic Indian bank SMS + finance coach data for FinnAI SLM",
            "license": "apache-2.0",
            "excludes": ["nova_indic.jsonl"],
            "note": "Nova Pro Indic expansions excluded from public dump; regenerate via ml/data/generate_nova_indic.py",
        }
        (staging / "dataset_info.json").write_text(json.dumps(meta, indent=2) + "\n")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=args.dataset_repo,
            repo_type="dataset",
        )
        if args.dataset_card.exists():
            api.upload_file(
                path_or_fileobj=str(args.dataset_card),
                path_in_repo="README.md",
                repo_id=args.dataset_repo,
                repo_type="dataset",
            )
        if args.guide.exists():
            api.upload_file(
                path_or_fileobj=str(args.guide),
                path_in_repo="FINETUNE_GUIDE.md",
                repo_id=args.dataset_repo,
                repo_type="dataset",
            )
        print("published dataset", args.dataset_repo)
        shutil.rmtree(staging, ignore_errors=True)

    print("done")


if __name__ == "__main__":
    main()
