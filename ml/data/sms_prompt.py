"""Shared ChatML / SMS JSON helpers. Keep in lockstep with LlmSmsParser.buildPrompt."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any

SMS_SYSTEM = (
    "You are a financial transaction parser. Return ONLY a JSON object. "
    "Do not include markdown, thinking, or explanations."
)

SMS_USER_TEMPLATE = """Extract details from this SMS sent by "{sender}":
"{sms}"

Return ONLY a JSON object with these fields:
- amount: number (no currency symbols)
- merchant: string (payee name or source)
- type: one of [INCOME, EXPENSE, CREDIT, TRANSFER, INVESTMENT]
- account: string (last 4 digits only, or null)
- balance: number (or null)
- category: string (e.g. Salary, Food, Travel, Shopping, etc.)

If not a transaction, return empty JSON {{}}.
Do not include markdown formatting or explanations."""

TYPES = ("INCOME", "EXPENSE", "CREDIT", "TRANSFER", "INVESTMENT")


def sms_user_prompt(sms: str, sender: str) -> str:
    return SMS_USER_TEMPLATE.format(sender=sender, sms=sms)


def chatml(system: str, user: str, assistant: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def gold_json(record: dict[str, Any] | None) -> str:
    if not record:
        return "{}"
    payload = {
        "amount": _num_or_none(record.get("amount")),
        "merchant": record.get("merchant"),
        "type": record.get("type"),
        "account": record.get("account"),
        "balance": _num_or_none(record.get("balance")),
        "category": record.get("category"),
    }
    # Drop keys that are None so empty gold can still be `{}` for negatives.
    if payload["amount"] is None and payload["type"] is None:
        return "{}"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _num_or_none(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        d = Decimal(str(value).replace(",", "").replace("Rs", "").replace("INR", "").strip())
    except (InvalidOperation, AttributeError):
        return None
    if d == d.to_integral_value():
        return int(d)
    return float(d)
