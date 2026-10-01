from __future__ import annotations

import json
from pathlib import Path


def write_report(path: Path, summary: dict) -> None:
    g = summary["gates"]
    ship = "YES" if g["ship"] else "NO"
    cand = summary["candidate"]
    base = summary["baseline"]
    lines = [
        f"# Eval report ({summary['tag']})",
        "",
        f"**SHIP: {ship}**",
        "",
        f"- Protocol: `{summary['protocol']}`",
        f"- Backend: `{summary['backend']}`",
        f"- Candidate: `{cand['model_id']}`",
        f"- Baseline: `{base['model_id']}`",
        "",
        summary.get("note", ""),
        "",
        "## SMS",
        "",
        "| Metric | Candidate | Baseline |",
        "| --- | --- | --- |",
        f"| n | {cand['sms']['n']} | {base['sms']['n']} |",
        f"| R-EM % | {cand['sms']['r_em_pct']} {cand['sms']['r_em_ci']} | {base['sms']['r_em_pct']} |",
        f"| Amount EM % | {cand['sms']['amount_em_pct']} | {base['sms']['amount_em_pct']} |",
        f"| JSON valid % | {cand['sms']['json_valid_pct']} | {base['sms']['json_valid_pct']} |",
        f"| Field micro-F1 | {cand['sms']['field_micro_f1']:.4f} | {base['sms']['field_micro_f1']:.4f} |",
        f"| Merchant exact % | {cand['sms']['merchant_exact_pct']} | {base['sms']['merchant_exact_pct']} |",
        f"| False-parse % | {cand['sms']['false_parse_pct']} | {base['sms']['false_parse_pct']} |",
        "",
        "## Chat",
        "",
        f"- Groundedness candidate {cand['chat']['grounded_pct']}% CI {cand['chat']['grounded_ci']}",
        f"- Groundedness baseline {base['chat']['grounded_pct']}%",
        f"- No-markdown {cand['chat']['no_markdown_pct']}%",
        f"- Length band {cand['chat']['length_band_pct']}%",
        "",
        "## Gates vs baseline",
        "",
        f"- Gate 1 R-EM lift: {g['gate1_r_em_lift']} (diff {g['r_em_diff_pp']} pp, CI {g['r_em_diff_ci_pp']})",
        f"- Gate 2 amount non-inf: {g['gate2_amount_noninf']} (diff {g['amount_em_diff_pp']} pp)",
        f"- Gate 3 chat non-inf: {g['gate3_chat_noninf']} (diff {g['grounded_diff_pp']} pp)",
        f"- Gate 4 JSON ≥95%: {g['gate4_json_valid']}",
        f"- Gate 5 on-device: {g['gate5_ondevice']}",
        f"- McNemar p (R-EM): {g['mcnemar_p']}",
    ]
    if "rq3_ge_qwen25" in g:
        lines.append(f"- RQ3 ≥ Qwen2.5 R-EM: {g['rq3_ge_qwen25']}")
    if summary.get("product_baseline"):
        pb = summary["product_baseline"]
        lines.extend(
            [
                "",
                "## Product baseline (Qwen2.5)",
                "",
                f"- Model: `{pb['model_id']}`",
                f"- R-EM %: {pb['sms']['r_em_pct']}",
                f"- Amount EM %: {pb['sms']['amount_em_pct']}",
                f"- Chat groundedness %: {pb['chat']['grounded_pct']}",
            ]
        )
    if summary.get("sms_fail_cases"):
        lines.extend(["", "## SMS fail cases (first 20)", ""])
        for c in summary["sms_fail_cases"]:
            lines.append(f"- `{c.get('bank')}` / `{c.get('template_id')}` id={c.get('id')}")
    lines.extend(
        [
            "",
            "On-device gate is filled by `python -m eval.ondevice_bench` on hardware.",
            "Final `MODEL_URL` ship also requires INT4 LiteRT conversion of the candidate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
