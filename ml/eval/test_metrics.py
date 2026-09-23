"""Unit tests for frozen metrics (run from ml/: python -m eval.test_metrics)."""

from __future__ import annotations

from eval.metrics import (
    amount_em,
    groundedness,
    mcnemar_pvalue,
    parse_json_object,
    score_sms,
    strip_output,
)


def test_strip_think_and_fence() -> None:
    raw = "<think>secret</think>\n```json\n{\"amount\": 1}\n```"
    assert "secret" not in strip_output(raw)
    assert parse_json_object(raw)["amount"] == 1


def test_amount_normalize() -> None:
    assert amount_em("Rs.1,234.50", "1234.50")
    assert amount_em(5000, "5000.00")
    assert not amount_em("10", "11")


def test_strict_rem() -> None:
    gold = {
        "amount": "5000.00",
        "merchant": "AMAZON",
        "type": "EXPENSE",
        "account": "1234",
        "balance": "25000",
    }
    pred = '{"amount":5000,"merchant":"amazon","type":"EXPENSE","account":"xx1234","balance":"25,000.00"}'
    s = score_sms(pred, gold)
    assert s.r_em and s.json_valid


def test_negative_empty() -> None:
    s = score_sms("{}", None)
    assert s.r_em and s.non_transaction and not s.false_parse
    s2 = score_sms('{"amount":10,"type":"EXPENSE"}', None)
    assert s2.false_parse and not s2.r_em


def test_groundedness() -> None:
    prompt = "You spent ₹7200.00 on Food. Income ₹65000"
    assert groundedness("Food is ₹7200 of income ₹65000.", prompt)
    assert not groundedness("You spent ₹99999 on Food.", prompt)


def test_mcnemar_identical() -> None:
    v = [True, False, True]
    assert mcnemar_pvalue(v, v) == 1.0


if __name__ == "__main__":
    tests = [
        test_strip_think_and_fence,
        test_amount_normalize,
        test_strict_rem,
        test_negative_empty,
        test_groundedness,
        test_mcnemar_identical,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"passed {len(tests)}")
