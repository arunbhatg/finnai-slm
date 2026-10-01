"""Build on-policy DPO prefs by sampling the Stage-3 student on coach prompts.

Rejected = model replies that fail groundedness.
Chosen = gold assistant from the seed row (ledger-faithful).

Usage (GPU):
  python -m data.generate_onpolicy_prefs \\
    --seeds data/out_v3/train_tagged.jsonl \\
    --model-id train/output_v3_4b_stage3/merged \\
    --out data/out_v3/dpo_onpolicy.jsonl --n 2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval.metrics import groundedness


def _eid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def load_coach_seeds(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("task") != "chat":
                continue
            if not r.get("must_ground"):
                continue
            rows.append(r)
            if len(rows) >= limit * 3:
                break
    # Prefer hard buckets
    hard = [r for r in rows if "hard" in str(r.get("bucket") or "") or "invent" in str(r.get("template_id") or "") or "wrong_month" in str(r.get("template_id") or "")]
    rest = [r for r in rows if r not in hard]
    ordered = hard + rest
    return ordered[:limit]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=Path, required=True)
    p.add_argument("--model-id", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--samples-per-prompt", type=int, default=2)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seeds = load_coach_seeds(args.seeds, args.n)
    tok = AutoTokenizer.from_pretrained(str(args.model_id), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_id),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    out_rows = []
    for i, row in enumerate(seeds):
        messages = [m for m in row["messages"] if m["role"] != "assistant"]
        gold = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        prompt_text = "\n".join(m["content"] for m in messages)
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        rejected = None
        for s in range(args.samples_per_prompt):
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                )
            pred = tok.decode(gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            if not groundedness(pred, prompt_text):
                rejected = pred
                break
            if s == 0 and rejected is None:
                # keep last sample even if grounded — skip pair
                pass
        if not rejected:
            continue
        if groundedness(gold, prompt_text) is False:
            continue
        tid = f"onpolicy/{row.get('template_id')}/{i}"
        out_rows.append(
            {
                "id": _eid(tid, rejected[:80]),
                "template_id": tid,
                "task": "chat_dpo",
                "must_ground": row.get("must_ground"),
                "prompt": messages,
                "chosen": gold,
                "rejected": rejected,
            }
        )
        if len(out_rows) >= args.n:
            break
        if (i + 1) % 50 == 0:
            print(f"scanned={i+1} pairs={len(out_rows)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(out_rows), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
