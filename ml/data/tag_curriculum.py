"""Tag v3 rows into curriculum buckets and emit stage-specific train jsonl files."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42

STAGE_MIX = {
    # sms_core+flow+indic+neg lumped as sms_* via weights below
    1: {"sms": 0.80, "chat": 0.10, "general": 0.10},
    2: {"sms": 0.55, "chat": 0.30, "general": 0.15},
    3: {"sms": 0.35, "chat": 0.55, "general": 0.10},
    5: {"sms": 1.00, "chat": 0.00, "general": 0.00},
}


def bucket_for(row: dict) -> str:
    tid = str(row.get("template_id") or "")
    task = row.get("task")
    gold = row.get("gold")
    if task == "general":
        return "general"
    if task == "chat":
        if any(x in tid for x in ("wrong_month", "invent_refuse", "hard", "eval/coach")):
            return "coach_hard"
        if tid.startswith("coach/") and any(
            x in tid for x in ("_hi", "leak_hi", "afford_hi", "food_hi", "subs_hi", "emi_hi", "summary_hi", "bills_hi")
        ):
            return "coach_ground"
        if "nova" in tid or "hard" in tid:
            return "coach_hard" if "invent" in tid or "wrong_month" in tid else "coach_ground"
        return "coach_ground"
    # sms
    if gold in (None, {}) or (isinstance(gold, dict) and gold.get("amount") in (None, "") and gold.get("type") in (None, "")):
        return "sms_neg"
    if tid.startswith("indic_") or "/hi" in tid or "indic_" in tid or "nova_sms" in tid:
        return "sms_indic"
    if any(
        x in tid
        for x in (
            "bbps",
            "emi_",
            "wallet_",
            "mf_sip",
            "upi_refund",
            "cc_payment",
            "cc_statement",
            "recharge",
        )
    ):
        return "sms_flow"
    return "sms_core"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def mix_stage(rows: list[dict], stage: int, n: int) -> list[dict]:
    mix = STAGE_MIX[stage]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        b = r.get("bucket") or bucket_for(r)
        r = dict(r)
        r["bucket"] = b
        if stage == 3 and r["task"] == "chat":
            # upsample hard coach
            by_task["chat"].append(r)
            if b == "coach_hard":
                by_task["chat"].append(r)
        elif stage == 5:
            if r["task"] == "sms":
                by_task["sms"].append(r)
                if b == "sms_neg":
                    by_task["sms"].append(r)
        else:
            by_task[r["task"]].append(r)

    rng = random.Random(SEED + stage)
    out: list[dict] = []
    for task, frac in mix.items():
        if frac <= 0:
            continue
        pool = list(by_task.get(task) or [])
        rng.shuffle(pool)
        k = int(n * frac)
        if not pool:
            continue
        while len(pool) < k:
            pool = pool + pool
        out.extend(pool[:k])
    rng.shuffle(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Emit curriculum stage train files")
    p.add_argument("--train", type=Path, required=True, help="Tagged or untagged train.jsonl")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--stage1-n", type=int, default=20000)
    p.add_argument("--stage2-n", type=int, default=28000)
    p.add_argument("--stage3-n", type=int, default=16000)
    p.add_argument("--stage5-n", type=int, default=8000)
    args = p.parse_args()

    rows = load_jsonl(args.train)
    tagged = []
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        r = dict(r)
        r["bucket"] = bucket_for(r)
        counts[r["bucket"]] += 1
        tagged.append(r)

    write_jsonl(args.out_dir / "train_tagged.jsonl", tagged)
    sizes = {1: args.stage1_n, 2: args.stage2_n, 3: args.stage3_n, 5: args.stage5_n}
    for stage, n in sizes.items():
        stage_rows = mix_stage(tagged, stage, n)
        write_jsonl(args.out_dir / f"train_stage{stage}.jsonl", stage_rows)
        print(f"stage{stage}: {len(stage_rows)}")

    (args.out_dir / "bucket_counts.json").write_text(
        json.dumps({"buckets": dict(counts), "stages": sizes}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
