"""Build train/val/test jsonl with template-level splits and mix ratios."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from data.extract_parser_gold import extract_repo
from data.generate_finance_chat import frozen_eval_set, generate as gen_chat
from data.generate_general_instruct import generate as gen_general
from data.generate_india_flows import frozen_eval_slice as india_flows_eval
from data.generate_india_flows import generate as gen_india_flows
from data.generate_indic_sms_manual import frozen_eval_slice as indic_eval
from data.generate_indic_sms_manual import generate as gen_indic_manual
from data.generate_synthetic_sms import generate as gen_sms

SEED = 42

# v2 mix (legacy)
TRAIN_MIX_V2 = {"sms": 0.60, "chat": 0.25, "general": 0.15}
# v3: more coach for groundedness; SMS still majority
TRAIN_MIX_V3 = {"sms": 0.55, "chat": 0.30, "general": 0.15}


def _group_hash(split_key: str) -> str:
    return hashlib.sha256(f"{SEED}:{split_key}".encode()).hexdigest()


def assign_splits(rows: list[dict]) -> dict[str, str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["split_key"]].append(r)
    ordered = sorted(groups.items(), key=lambda kv: _group_hash(kv[0]))
    total = sum(len(v) for v in groups.values())
    targets = {"test": int(0.10 * total), "val": int(0.10 * total)}
    mapping: dict[str, str] = {}
    filled = {"test": 0, "val": 0}
    for key, items in ordered:
        if filled["test"] < targets["test"]:
            mapping[key] = "test"
            filled["test"] += len(items)
        elif filled["val"] < targets["val"]:
            mapping[key] = "val"
            filled["val"] += len(items)
        else:
            mapping[key] = "train"
    return mapping


def mix_train(rows: list[dict], n: int, mix: dict[str, float]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    rng = random.Random(SEED)
    out: list[dict] = []
    for task, frac in mix.items():
        pool = by[task]
        rng.shuffle(pool)
        k = int(n * frac)
        if not pool:
            continue
        while len(pool) < k:
            pool = pool + pool
        out.extend(pool[:k])
    rng.shuffle(out)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(".."))
    p.add_argument("--out", type=Path, default=Path("data/out"))
    p.add_argument("--train-size", type=int, default=None, help="Default 12000 (v2) or 28000 (v3)")
    p.add_argument(
        "--version",
        choices=("v2", "v3"),
        default="v3",
        help="v3 enables India flows + manual Indic SMS and 55/30/15 mix",
    )
    p.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="Verified jsonl from Nova data factory (or other extra slices)",
    )
    p.add_argument("--chat-n", type=int, default=None)
    p.add_argument("--general-n", type=int, default=None)
    p.add_argument("--india-flows-n", type=int, default=4000)
    p.add_argument("--indic-manual-n", type=int, default=2500)
    args = p.parse_args()

    if args.train_size is None:
        args.train_size = 28000 if args.version == "v3" else 12000
    if args.chat_n is None:
        args.chat_n = 9000 if args.version == "v3" else 3000
    if args.general_n is None:
        args.general_n = 3500 if args.version == "v3" else 1800
    mix = TRAIN_MIX_V3 if args.version == "v3" else TRAIN_MIX_V2

    sms = extract_repo(args.repo_root) + gen_sms()
    if args.version == "v3":
        sms = sms + gen_india_flows(max_pos=args.india_flows_n, max_neg=max(800, args.india_flows_n // 5))
        sms = sms + gen_indic_manual(max_rows=args.indic_manual_n)
    chat = gen_chat(n=args.chat_n)
    general = gen_general(n=args.general_n)
    extra: list[dict] = []
    for path in args.extra:
        with path.open(encoding="utf-8") as f:
            extra.extend(json.loads(line) for line in f if line.strip())
    all_rows = sms + chat + general + extra
    mapping = assign_splits(all_rows)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        split = mapping[r["split_key"]]
        r = dict(r)
        r["split"] = split
        buckets[split].append(r)

    buckets["train"] = mix_train(buckets["train"], args.train_size, mix)

    def sms_first(rows: list[dict]) -> list[dict]:
        sms_rows = [r for r in rows if r["task"] == "sms"]
        other = [r for r in rows if r["task"] != "sms"]
        return sms_rows + other

    buckets["val"] = sms_first(buckets["val"])
    buckets["test"] = sms_first(buckets["test"])

    out = args.out
    write_jsonl(out / "train.jsonl", buckets["train"])
    write_jsonl(out / "val.jsonl", buckets["val"])
    write_jsonl(out / "test.jsonl", buckets["test"])
    write_jsonl(out / "test_sms.jsonl", [r for r in buckets["test"] if r["task"] == "sms"])
    write_jsonl(out / "val_sms.jsonl", [r for r in buckets["val"] if r["task"] == "sms"])
    write_jsonl(out / "chat_eval.jsonl", frozen_eval_set())
    fixtures = Path(__file__).resolve().parents[1] / "eval" / "fixtures"
    write_jsonl(fixtures / "chat_eval.jsonl", frozen_eval_set())
    if args.version == "v3":
        write_jsonl(fixtures / "india_flows_eval.jsonl", india_flows_eval(200))
        write_jsonl(fixtures / "indic_sms_eval.jsonl", indic_eval(200))
        write_jsonl(out / "india_flows_eval.jsonl", india_flows_eval(200))
        write_jsonl(out / "indic_sms_eval.jsonl", indic_eval(200))

    sums = out / "SHA256SUMS"
    lines = []
    for name in sorted(p.name for p in out.glob("*.jsonl")):
        digest = sha256_file(out / name)
        lines.append(f"{digest}  {name}")
    ids = sorted(r["id"] for r in buckets["test"] if r["task"] == "sms")
    id_path = out / "test_sms_ids.txt"
    id_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    lines.append(f"{sha256_file(id_path)}  {id_path.name}")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")

    counts = {
        "version": args.version,
        "train_mix": mix,
        "train_size_target": args.train_size,
        **{
            split: {
                "n": len(rows),
                "by_task": {
                    t: sum(1 for r in rows if r["task"] == t)
                    for t in ("sms", "chat", "general")
                },
            }
            for split, rows in buckets.items()
        },
    }
    (out / "counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    manifest = Path(__file__).resolve().parent / "manifest"
    manifest.mkdir(parents=True, exist_ok=True)
    (manifest / "SHA256SUMS").write_text(sums.read_text(encoding="utf-8"), encoding="utf-8")
    (manifest / "counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
