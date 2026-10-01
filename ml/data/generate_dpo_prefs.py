"""Build DPO preference pairs for Ask Finn groundedness.

Chosen = ledger-faithful answers (from generate_finance_chat).
Rejected = plausible but hallucinated ₹ / wrong merchants / markdown junk.

No external LLM required — deterministic rejects keep the OSS story clean.
Optional: pass --teacher-jsonl later to swap in teacher-ranked chosen answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.generate_finance_chat import HARD_QA, INDIC_QA, LEDGERS, QA, _ground, _system


def _eid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _reject_invent_rupees(led: dict) -> str:
    return (
        f"You only spent ₹99999 on mystery items this month — cut that immediately. "
        f"Also {led['freq'][0]} looks fine at ₹42 only. "
        f"**Buy a course** and ignore the ₹{led['expense']} total in your app."
    )


def _reject_wrong_income(led: dict) -> str:
    fake = "250000.00"
    return (
        f"Income is ₹{fake} so you are doing great. Net is roughly ₹{fake} minus ₹100. "
        f"Do not worry about the ledger showing ₹{led['income']}."
    )


def _reject_missing_month(led: dict) -> str:
    return (
        "In January 2024 you spent ₹87432.50 across 90 merchants including XYZ999. "
        f"Compared to now (₹{led['expense']}) that was worse — trust these January figures."
    )


REJECTORS = (
    ("invent_rupees", _reject_invent_rupees),
    ("wrong_income", _reject_wrong_income),
    ("missing_month", _reject_missing_month),
)


def build_pairs(n: int = 4000) -> list[dict]:
    pool = QA + INDIC_QA + HARD_QA
    rows: list[dict] = []
    i = 0
    while len(rows) < n:
        led = LEDGERS[i % len(LEDGERS)]
        qid, user, fn = pool[i % len(pool)]
        system = _system(led)
        chosen = fn(led)
        rname, rfn = REJECTORS[i % len(REJECTORS)]
        rejected = rfn(led)
        tid = f"dpo/{led['id']}/{qid}/{rname}"
        rows.append(
            {
                "id": _eid(tid, str(i)),
                "template_id": tid,
                "task": "chat_dpo",
                "must_ground": _ground(led),
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        i += 1
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="FinnAI v3 DPO preference factory")
    p.add_argument("--out", type=Path, default=Path("data/out_v3/dpo_prefs.jsonl"))
    p.add_argument("--n", type=int, default=4000)
    args = p.parse_args()
    rows = build_pairs(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(rows), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
