"""SMS and chat metrics. Protocol: docs/llm-eval-protocol.md."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from scipy.stats import binomtest

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
NUM_RE = re.compile(r"(?:₹|Rs\.?|INR)?\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)")
MD_RE = re.compile(r"[*#`]")

SMS_FIELDS = ("amount", "type", "merchant", "account", "balance")


def strip_output(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    text = text.strip()
    text = FENCE_RE.sub("", text).strip()
    return text


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = strip_output(text)
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def dec(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    s = str(value)
    s = re.sub(r"(Rs\.?|INR|₹)", "", s, flags=re.I)
    s = s.replace(",", "").replace(" ", "").strip()
    if s.lower() in {"null", "none"}:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def amount_em(pred: Any, gold: Any) -> bool:
    a, b = dec(pred), dec(gold)
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.compare(b) == 0


def norm_merchant(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().casefold()
    s = re.sub(r"^upi/", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def merchant_exact(pred: Any, gold: Any) -> bool:
    return norm_merchant(pred) == norm_merchant(gold)


def merchant_token_f1(pred: Any, gold: Any) -> float:
    p = set(norm_merchant(pred).split()) - {""}
    g = set(norm_merchant(gold).split()) - {""}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    prec = tp / len(p)
    rec = tp / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def last4(value: Any) -> str | None:
    if value is None or str(value).strip().lower() in {"", "null", "none"}:
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    return digits[-4:]


def account_em(pred: Any, gold: Any) -> bool:
    return last4(pred) == last4(gold)


def type_em(pred: Any, gold: Any) -> bool:
    p = (str(pred).upper() if pred is not None else "")
    g = (str(gold).upper() if gold is not None else "")
    if not p and not g:
        return True
    return p == g


def is_empty_gold(gold: dict | None) -> bool:
    if not gold:
        return True
    return gold.get("amount") in (None, "") and gold.get("type") in (None, "")


@dataclass
class SmsItemScore:
    json_valid: bool
    r_em: bool
    amount_em: bool
    type_em: bool
    merchant_exact: bool
    merchant_f1: float
    account_em: bool
    balance_em: bool
    field_hits: int
    field_total: int
    false_parse: bool
    non_transaction: bool


def score_sms(pred_text: str, gold: dict | None) -> SmsItemScore:
    obj = parse_json_object(pred_text)
    non_txn = is_empty_gold(gold)
    if obj is None:
        return SmsItemScore(
            json_valid=False,
            r_em=False,
            amount_em=False,
            type_em=False,
            merchant_exact=False,
            merchant_f1=0.0,
            account_em=False,
            balance_em=False,
            field_hits=0,
            field_total=len(SMS_FIELDS),
            false_parse=False,
            non_transaction=non_txn,
        )
    if non_txn:
        empty = not obj or (obj.get("amount") in (None, "") and obj.get("type") in (None, ""))
        return SmsItemScore(
            json_valid=True,
            r_em=empty,
            amount_em=empty,
            type_em=empty,
            merchant_exact=empty,
            merchant_f1=1.0 if empty else 0.0,
            account_em=empty,
            balance_em=empty,
            field_hits=len(SMS_FIELDS) if empty else 0,
            field_total=len(SMS_FIELDS),
            false_parse=not empty,
            non_transaction=True,
        )
    gold = gold or {}
    a = amount_em(obj.get("amount"), gold.get("amount"))
    t = type_em(obj.get("type"), gold.get("type"))
    m = merchant_exact(obj.get("merchant"), gold.get("merchant"))
    acc = account_em(obj.get("account"), gold.get("account"))
    bal = amount_em(obj.get("balance"), gold.get("balance"))
    hits = int(a) + int(t) + int(m) + int(acc) + int(bal)
    return SmsItemScore(
        json_valid=True,
        r_em=a and t and m and acc and bal,
        amount_em=a,
        type_em=t,
        merchant_exact=m,
        merchant_f1=merchant_token_f1(obj.get("merchant"), gold.get("merchant")),
        account_em=acc,
        balance_em=bal,
        field_hits=hits,
        field_total=len(SMS_FIELDS),
        false_parse=False,
        non_transaction=False,
    )


def numbers_in(text: str) -> set[str]:
    found: set[str] = set()
    for m in NUM_RE.finditer(text or ""):
        d = dec(m.group(1))
        if d is None:
            continue
        found.add(format(d.normalize(), "f").rstrip("0").rstrip(".") if "." in format(d, "f") else str(int(d)))
        found.add(str(d))
        found.add(format(d, "f"))
    return found


def groundedness(reply: str, prompt: str) -> bool:
    reply_nums = numbers_in(strip_output(reply))
    prompt_nums = numbers_in(prompt)
    # Compare via Decimal so 7200.00 matches 7200
    def canon(xs: set[str]) -> set[Decimal]:
        out: set[Decimal] = set()
        for x in xs:
            d = dec(x)
            if d is not None:
                out.add(d)
        return out

    return canon(reply_nums).issubset(canon(prompt_nums))


def no_markdown(reply: str) -> bool:
    t = strip_output(reply)
    return MD_RE.search(t) is None


def length_band(reply: str, lo: int = 40, hi: int = 700) -> bool:
    n = len(strip_output(reply))
    return lo <= n <= hi


@dataclass
class Aggregate:
    n: int
    r_em: float
    amount_em: float
    type_em: float
    merchant_exact: float
    merchant_f1: float
    json_valid: float
    field_micro_f1: float
    false_parse: float
    chat_grounded: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def aggregate_sms(scores: list[SmsItemScore]) -> Aggregate:
    n = len(scores) or 1
    non_txn = [s for s in scores if s.non_transaction]
    hits = sum(s.field_hits for s in scores)
    tot = sum(s.field_total for s in scores) or 1
    return Aggregate(
        n=len(scores),
        r_em=sum(s.r_em for s in scores) / n,
        amount_em=sum(s.amount_em for s in scores) / n,
        type_em=sum(s.type_em for s in scores) / n,
        merchant_exact=sum(s.merchant_exact for s in scores) / n,
        merchant_f1=sum(s.merchant_f1 for s in scores) / n,
        json_valid=sum(s.json_valid for s in scores) / n,
        field_micro_f1=hits / tot,
        false_parse=(sum(s.false_parse for s in non_txn) / len(non_txn)) if non_txn else 0.0,
    )


def bootstrap_ci(values: list[float], n_boot: int = 10_000, seed: int = 42) -> tuple[float, float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_diff_ci(
    a: list[float], b: list[float], n_boot: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    n = len(aa)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = aa[idx].mean(axis=1) - bb[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def mcnemar_pvalue(pred_a: list[bool], pred_b: list[bool]) -> float:
    """McNemar exact two-sided on paired correctness vs baseline b."""
    n01 = sum(1 for x, y in zip(pred_a, pred_b) if (not x) and y)
    n10 = sum(1 for x, y in zip(pred_a, pred_b) if x and (not y))
    n = n01 + n10
    if n == 0:
        return 1.0
    # binomtest under H0 p=0.5 for discordant pairs
    return float(binomtest(n10, n, 0.5, alternative="two-sided").pvalue)


def pct(x: float) -> float:
    return round(100.0 * x, 2)
