"""India money-flow SMS templates beyond plain UPI debit (BBPS, EMI, wallets, MF, refunds, CC).

Privacy-safe synthetic only — no user inboxes.
"""

from __future__ import annotations

import hashlib
from itertools import product

from data.schema import example

# Reuse bank senders / formatting helpers from the core SMS generator.
from data.generate_synthetic_sms import LAST4, SENDERS, _amt_plain

BILLERS = [
    "BESCOM", "MSEDCL", "TATA_POWER", "ACT_FIBER", "AIRTEL_DTH",
    "JIOFIBER", "BWSSB", "IGL_GAS", "INDIAN_OIL", "BSNL",
]
LENDERS = [
    "BAJAJ_FINSERV", "HDFC_BANK_EMI", "ICICI_LOANS", "HOME_CREDIT",
    "TATA_CAPITAL", "AXIS_FINANCE", "KOTAK_EMI", "SBI_CARD_EMI",
]
WALLETS = ["PHONEPE_WALLET", "PAYTM_WALLET", "AMAZON_PAY", "MOBIKWIK", "FREECHARGE"]
MF_SCHEMES = [
    "NIFTY50_INDEX", "PARAG_PARIKH_FLEXI", "UTI_NIFTY", "SBI_BLUECHIP",
    "AXIS_MIDCAP", "ICICI_PRU_EQUITY", "MOTILAL_MIDCAP", "HDFC_FLEXICAP",
]
CC_MERCHANTS = ["AMAZON", "FLIPKART", "SWIGGY", "IRCTC", "MAKE_MY_TRIP", "APOLLO_PHARMACY"]

AMOUNTS = [
    "99.00", "149.00", "299.00", "499.00", "899.00", "1,299.00",
    "2,499.00", "4,999.00", "7,850.00", "12,500.00", "25,000.00",
]
BALANCES = ["3,450.25", "12,800.00", "48,200.50", "1,05,000.00", None]
DATES = ["03-01-26", "14-02-26", "22-03-26", "08-04-26", "19-05-26", "30-06-26"]


def _eid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _row(bank: str, template_id: str, sms: str, sender: str, gold: dict | None) -> dict:
    return example(
        example_id=_eid(template_id, sms, sender),
        split_key=f"{bank}|{template_id}",
        bank=bank,
        template_id=template_id,
        task="sms",
        sms=sms,
        sender=sender,
        gold=gold,
    )


def generate(max_pos: int = 4000, max_neg: int = 800) -> list[dict]:
    rows: list[dict] = []
    banks = list(SENDERS.keys())

    def bbps(biller, a, l4, d, bank):
        bal = BALANCES[0]
        sms = (
            f"{bank}: BBPS payment of Rs.{a} to {biller} from A/c XX{l4} on {d} successful. "
            f"Avl Bal Rs.{bal}. Ref BBPS{l4}{d[:2]}"
        )
        gold = {
            "amount": _amt_plain(a),
            "merchant": biller,
            "type": "EXPENSE",
            "account": l4,
            "balance": _amt_plain(bal or ""),
            "category": "Bills",
        }
        return _row(bank, "bbps_pay_v1", sms, SENDERS[bank], gold)

    def emi_debit(lender, a, l4, d, bank):
        bal = BALANCES[1]
        sms = (
            f"{bank}: EMI of Rs.{a} for {lender} auto-debited from A/c XX{l4} on {d}. "
            f"Avl Bal Rs.{bal}."
        )
        gold = {
            "amount": _amt_plain(a),
            "merchant": lender,
            "type": "EXPENSE",
            "account": l4,
            "balance": _amt_plain(bal or ""),
            "category": "EMI",
        }
        return _row(bank, "emi_autodebit_v1", sms, SENDERS[bank], gold)

    def wallet_load(wallet, a, l4, d, bank):
        bal = BALANCES[2]
        sms = (
            f"{bank}: Rs.{a} transferred to {wallet} from A/c XX{l4} on {d}. "
            f"Avl Bal Rs.{bal}."
        )
        gold = {
            "amount": _amt_plain(a),
            "merchant": wallet,
            "type": "TRANSFER",
            "account": l4,
            "balance": _amt_plain(bal or ""),
            "category": "Wallet",
        }
        return _row(bank, "wallet_load_v1", sms, SENDERS[bank], gold)

    def wallet_spend(wallet, a, l4, d, bank):
        # Wallet SMS often lacks bank A/c; still a real expense.
        sender = "JD-WALLET-S"
        sms = (
            f"{wallet}: Rs.{a} paid at MERCHANT{l4[-2:]} on {d}. "
            f"Wallet bal Rs.{BALANCES[0]}."
        )
        gold = {
            "amount": _amt_plain(a),
            "merchant": f"MERCHANT{l4[-2:]}",
            "type": "EXPENSE",
            "account": None,
            "balance": _amt_plain(BALANCES[0] or ""),
            "category": "Shopping",
        }
        return _row(bank, "wallet_spend_v1", sms, sender, gold)

    def mf_sip(scheme, a, l4, d, bank):
        bal = BALANCES[1]
        sms = (
            f"{bank}: SIP Rs.{a} for {scheme} debited from A/c XX{l4} on {d}. "
            f"Avl Bal Rs.{bal}. Mutual Fund"
        )
        gold = {
            "amount": _amt_plain(a),
            "merchant": scheme,
            "type": "INVESTMENT",
            "account": l4,
            "balance": _amt_plain(bal or ""),
            "category": "Investment",
        }
        return _row(bank, "mf_sip_v1", sms, SENDERS[bank], gold)

    def upi_refund(m, a, l4, d, bank):
        bal = BALANCES[2]
        sms = (
            f"{bank}: Refund Rs.{a} credited to A/c XX{l4} from {m} via UPI on {d}. "
            f"Avl Bal Rs.{bal}."
        )
        gold = {
            "amount": _amt_plain(a),
            "merchant": m,
            "type": "INCOME",
            "account": l4,
            "balance": _amt_plain(bal or ""),
            "category": "Refund",
        }
        return _row(bank, "upi_refund_v1", sms, SENDERS[bank], gold)

    def cc_statement(m, a, l4, d, bank):
        sms = (
            f"{bank} Credit Card xx{l4}: Total Amount Due Rs.{a} as of {d}. "
            f"Min Due Rs.{AMOUNTS[0]}. Pay by {DATES[(DATES.index(d) + 1) % len(DATES)]}."
        )
        # Statement alert is NOT a spend — gold empty (refuse / {}).
        return _row(bank, "cc_statement_v1", sms, SENDERS[bank], None)

    def cc_payment(m, a, l4, d, bank):
        bal = BALANCES[3]
        sms = (
            f"{bank}: Credit Card xx{l4} payment of Rs.{a} received from A/c XX{l4} on {d}. "
            f"Savings Avl Bal Rs.{bal}."
        )
        gold = {
            "amount": _amt_plain(a),
            "merchant": f"{bank}_CC_PAYMENT",
            "type": "TRANSFER",
            "account": l4,
            "balance": _amt_plain(bal or ""),
            "category": "Credit Card",
        }
        return _row(bank, "cc_payment_v1", sms, SENDERS[bank], gold)

    def recharge(biller, a, l4, d, bank):
        bal = BALANCES[0]
        sms = (
            f"{bank}: Recharge of Rs.{a} for {biller} paid from A/c XX{l4} on {d}. "
            f"Avl Bal Rs.{bal}."
        )
        gold = {
            "amount": _amt_plain(a),
            "merchant": biller,
            "type": "EXPENSE",
            "account": l4,
            "balance": _amt_plain(bal or ""),
            "category": "Bills",
        }
        return _row(bank, "recharge_v1", sms, SENDERS[bank], gold)

    builders_pos = [
        (bbps, BILLERS),
        (emi_debit, LENDERS),
        (wallet_load, WALLETS),
        (wallet_spend, WALLETS),
        (mf_sip, MF_SCHEMES),
        (upi_refund, CC_MERCHANTS),
        (cc_payment, CC_MERCHANTS),
        (recharge, BILLERS),
    ]

    n = 0
    combo_i = 0
    combos = list(product(AMOUNTS, LAST4, DATES))
    while n < max_pos and combo_i < len(combos) * len(builders_pos):
        a, l4, d = combos[combo_i % len(combos)]
        builder, pool = builders_pos[combo_i % len(builders_pos)]
        entity = pool[(combo_i // len(builders_pos)) % len(pool)]
        bank = banks[combo_i % len(banks)]
        rows.append(builder(entity, a, l4, d, bank))
        n += 1
        combo_i += 1

    # Explicit CC statement negatives (and other flow negatives)
    neg_templates = [
        (
            "cc_statement_v1",
            "{bank} Credit Card xx{l4}: Total Amount Due Rs.{a} as of {d}. Min Due Rs.99.00.",
            None,
        ),
        (
            "emi_reminder_v1",
            "{bank}: Reminder — EMI of Rs.{a} for {m} will debit on {d} from A/c XX{l4}.",
            None,
        ),
        (
            "bbps_failed_v1",
            "{bank}: BBPS payment of Rs.{a} to {m} FAILED. Amount not debited.",
            None,
        ),
        (
            "sip_paused_v1",
            "{bank}: Your SIP of Rs.{a} for {m} is paused. No debit on {d}.",
            None,
        ),
        (
            "wallet_kyc_v1",
            "{m}: Complete wallet KYC to continue using balance. Ignore if done.",
            None,
        ),
        (
            "mandate_otp_v1",
            "{bank}: {otp} is OTP to register e-mandate of Rs.{a} for {m}. Do not share.",
            None,
        ),
        (
            "reward_promo_v1",
            "Get 5% cashback on {m} bills via BBPS with {bank}. T&C.",
            None,
        ),
        (
            "upi_collect_request_v1",
            "UPI collect request of Rs.{a} from {m}. Approve in app if genuine. {bank}",
            None,
        ),
    ]

    n_neg = 0
    for i, (a, l4, d) in enumerate(combos):
        if n_neg >= max_neg:
            break
        bank = banks[i % len(banks)]
        tid, tmpl, _ = neg_templates[i % len(neg_templates)]
        m = (BILLERS + LENDERS + WALLETS + MF_SCHEMES)
        m = m[i % len(m)]
        sender = SENDERS[bank]
        sms = tmpl.format(
            bank=bank,
            otp=str(200000 + i)[-6:],
            m=m,
            l4=l4,
            d=d,
            a=a,
        )
        rows.append(_row(bank, tid, sms, sender, None))
        n_neg += 1

    return rows


def frozen_eval_slice(n: int = 200) -> list[dict]:
    """Held-out-style slice for v3 india-flows reporting (template ids prefixed eval/)."""
    raw = generate(max_pos=max(n, 50), max_neg=max(n // 5, 10))
    out = []
    for i, r in enumerate(raw):
        if len(out) >= n:
            break
        r = dict(r)
        r["template_id"] = f"eval/india_flows/{r['template_id']}/{i}"
        r["split_key"] = f"eval|{r['template_id']}"
        r["id"] = _eid(r["template_id"], r.get("sms", ""))
        out.append(r)
    return out
