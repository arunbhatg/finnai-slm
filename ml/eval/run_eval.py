"""Run SMS + chat eval. Backends: dummy (harness), hf (Transformers)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python -m eval.run_eval` from ml/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import (
    aggregate_sms,
    bootstrap_ci,
    bootstrap_diff_ci,
    groundedness,
    length_band,
    mcnemar_pvalue,
    no_markdown,
    pct,
    score_sms,
)
from eval.report import write_report


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def predict_dummy(row: dict) -> str:
    if row.get("task") == "sms":
        gold = row.get("gold") or {}
        if not gold:
            return "{}"
        return json.dumps(gold, ensure_ascii=False)
    msgs = row.get("messages") or []
    return msgs[-1]["content"] if msgs else ""


def predict_hf(rows: list[dict], model_id: str, max_new: int) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs: list[str] = []
    for i, row in enumerate(rows):
        messages = [m for m in row["messages"] if m["role"] != "assistant"]
        try:
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tok(text, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
        )
        new_tokens = gen[0][inputs["input_ids"].shape[1] :]
        outs.append(tok.decode(new_tokens, skip_special_tokens=True))
        if (i + 1) % 50 == 0:
            print(f"[{model_id}] {i + 1}/{len(rows)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outs


def eval_sms(rows: list[dict], preds: list[str]) -> dict:
    scores = [score_sms(p, r.get("gold")) for r, p in zip(rows, preds)]
    agg = aggregate_sms(scores)
    r_em = [float(s.r_em) for s in scores]
    amt = [float(s.amount_em) for s in scores]
    jv = [float(s.json_valid) for s in scores]
    return {
        "aggregate": {
            "n": agg.n,
            "r_em": agg.r_em,
            "r_em_pct": pct(agg.r_em),
            "r_em_ci": [pct(x) for x in bootstrap_ci(r_em)],
            "amount_em": agg.amount_em,
            "amount_em_pct": pct(agg.amount_em),
            "amount_em_ci": [pct(x) for x in bootstrap_ci(amt)],
            "json_valid": agg.json_valid,
            "json_valid_pct": pct(agg.json_valid),
            "json_valid_ci": [pct(x) for x in bootstrap_ci(jv)],
            "field_micro_f1": agg.field_micro_f1,
            "merchant_exact_pct": pct(agg.merchant_exact),
            "merchant_f1": agg.merchant_f1,
            "false_parse_pct": pct(agg.false_parse),
        },
        "r_em_vec": r_em,
        "amount_em_vec": amt,
        "json_valid_vec": jv,
        "scores": scores,
        "preds": preds,
    }


def eval_chat(rows: list[dict], preds: list[str]) -> dict:
    grounded = []
    md = []
    length = []
    for r, pred in zip(rows, preds):
        msgs = r.get("messages") or []
        prompt = "\n".join(m["content"] for m in msgs if m["role"] != "assistant")
        grounded.append(float(groundedness(pred, prompt)))
        md.append(float(no_markdown(pred)))
        length.append(float(length_band(pred)))
    g_mean = sum(grounded) / (len(grounded) or 1)
    return {
        "n": len(rows),
        "grounded": g_mean,
        "grounded_pct": pct(g_mean),
        "grounded_ci": [pct(x) for x in bootstrap_ci(grounded)],
        "no_markdown_pct": pct(sum(md) / (len(md) or 1)),
        "length_band_pct": pct(sum(length) / (len(length) or 1)),
        "grounded_vec": grounded,
        "preds": preds,
    }


def gates(ft: dict, base: dict, chat_ft: dict, chat_base: dict) -> dict:
    sms_ft, sms_base = ft["sms"], base["sms"]
    r_diff = sms_ft["aggregate"]["r_em"] - sms_base["aggregate"]["r_em"]
    r_ci = bootstrap_diff_ci(sms_ft["r_em_vec"], sms_base["r_em_vec"])
    amt_diff = sms_ft["aggregate"]["amount_em"] - sms_base["aggregate"]["amount_em"]
    g_diff = chat_ft["grounded"] - chat_base["grounded"]
    p_mc = mcnemar_pvalue(
        [bool(x) for x in sms_ft["r_em_vec"]],
        [bool(x) for x in sms_base["r_em_vec"]],
    )
    g1 = r_diff >= 0.03 and r_ci[0] > 0
    g2 = amt_diff >= -0.01
    g3 = g_diff >= -0.05
    g4 = sms_ft["aggregate"]["json_valid"] >= 0.95
    # Gate 5 is on-device; recorded separately.
    return {
        "r_em_diff_pp": pct(r_diff),
        "r_em_diff_ci_pp": [pct(r_ci[0]), pct(r_ci[1])],
        "amount_em_diff_pp": pct(amt_diff),
        "grounded_diff_pp": pct(g_diff),
        "mcnemar_p": p_mc,
        "gate1_r_em_lift": g1,
        "gate2_amount_noninf": g2,
        "gate3_chat_noninf": g3,
        "gate4_json_valid": g4,
        "gate5_ondevice": None,
        "ship": all([g1, g2, g3, g4]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sms", type=Path, required=True)
    p.add_argument("--chat", type=Path, required=True)
    p.add_argument("--backend", choices=("dummy", "hf"), default="dummy")
    p.add_argument("--model-id", default="dummy-gold")
    p.add_argument("--baseline-id", default="")
    p.add_argument("--product-id", default="", help="Optional Qwen2.5 product baseline")
    p.add_argument("--out", type=Path, default=Path("eval/reports"))
    p.add_argument("--tag", default="harness")
    p.add_argument("--job-id", default="")
    args = p.parse_args()

    sms_rows = load_jsonl(args.sms)
    chat_rows = load_jsonl(args.chat)

    def run(model_id: str) -> dict:
        if args.backend == "dummy":
            sms_pred = [predict_dummy(r) for r in sms_rows]
            chat_pred = [predict_dummy(r) for r in chat_rows]
        else:
            sms_pred = predict_hf(sms_rows, model_id, 256)
            chat_pred = predict_hf(chat_rows, model_id, 512)
        return {
            "model_id": model_id,
            "sms": eval_sms(sms_rows, sms_pred),
            "chat": eval_chat(chat_rows, chat_pred),
        }

    ft = run(args.model_id)
    # Dummy harness: compare gold dummy to a degraded dummy (empty JSON) as "base"
    if args.backend == "dummy":
        empty_sms = eval_sms(sms_rows, ["{}"] * len(sms_rows))
        empty_chat = eval_chat(chat_rows, ["I cannot see any numbers."] * len(chat_rows))
        base = {"model_id": "dummy-empty", "sms": empty_sms, "chat": empty_chat}
    elif args.baseline_id:
        base = run(args.baseline_id)
    else:
        base = ft

    product = run(args.product_id) if args.product_id and args.backend == "hf" else None

    fail_ids = []
    for row, score in zip(sms_rows, ft["sms"]["scores"]):
        if not score.r_em:
            fail_ids.append(
                {
                    "id": row.get("id"),
                    "template_id": row.get("template_id"),
                    "bank": row.get("bank"),
                }
            )

    summary = {
        "tag": args.tag,
        "backend": args.backend,
        "job_id": args.job_id,
        "candidate": {
            "model_id": ft["model_id"],
            "sms": ft["sms"]["aggregate"],
            "chat": {k: v for k, v in ft["chat"].items() if not k.endswith("_vec") and k != "preds"},
        },
        "baseline": {
            "model_id": base["model_id"],
            "sms": base["sms"]["aggregate"],
            "chat": {k: v for k, v in base["chat"].items() if not k.endswith("_vec") and k != "preds"},
        },
        "gates": gates(ft, base, ft["chat"], base["chat"]),
        "protocol": "docs/llm-eval-protocol.md",
        "sms_fail_cases": fail_ids[:20],
        "note": (
            "dummy backend verifies the harness. SHIP from dummy is not a product decision. "
            "HF backend compares candidate vs qwen3_base; gate5 on-device is separate."
        ),
    }
    if product:
        summary["product_baseline"] = {
            "model_id": product["model_id"],
            "sms": product["sms"]["aggregate"],
            "chat": {
                k: v
                for k, v in product["chat"].items()
                if not k.endswith("_vec") and k != "preds"
            },
        }
        # RQ3: INT4/FT R-EM must be >= qwen25_hf R-EM (default block)
        summary["gates"]["rq3_ge_qwen25"] = (
            ft["sms"]["aggregate"]["r_em"] >= product["sms"]["aggregate"]["r_em"]
        )
        summary["gates"]["ship"] = bool(
            summary["gates"]["ship"] and summary["gates"]["rq3_ge_qwen25"]
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_report(args.out / f"{args.tag}.md", summary)
    print(json.dumps({"ship": summary["gates"]["ship"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
