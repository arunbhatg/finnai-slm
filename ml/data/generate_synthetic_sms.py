"""Grammar-based Indian bank SMS (no user inboxes)."""

from __future__ import annotations

import hashlib
from itertools import product

from data.schema import example

MERCHANTS = [
    "SWIGGY", "ZOMATO", "AMAZON", "FLIPKART", "BIGBASKET", "IRCTC",
    "NETFLIX", "SPOTIFY", "UBER", "OLA", "DMART", "RELIANCEJP",
    "AIRTELPAY", "JIOMART", "MYNTRA", "NYKAA", "PHONEPE", "GPAY",
    "PAYTM", "BHIM", "BOOKMYSHOW", "MAKEMYTRIP", "INDIANGO", "BLINKIT",
]

AMOUNTS = [
    "1.00", "12.50", "99", "149.00", "199.99", "250", "499.00", "750.25",
    "999", "1,249.00", "2,500", "4,999.00", "12,345.67", "25,000.00", "50,000",
]

LAST4 = ["1234", "0001", "9876", "4321", "5555", "1001"]
BALANCES = ["1,234.50", "8,000.00", "25,000.00", "1,02,450.75", None]
DATES = ["01-01-26", "15-06-26", "21-08-26", "09-09-26"]

SENDERS = {
    "HDFC": "VM-HDFCBK-S",
    "SBI": "VK-SBIUPI-S",
    "ICICI": "VM-ICICIB-S",
    "AXIS": "VM-AXISBK-S",
    "KOTAK": "VM-KOTAKB-S",
    "PNB": "VM-PUNJAB-S",
    "BOB": "VM-BOBTRN-S",
    "UNION": "VM-UNIONB-S",
    "IDFC": "VM-IDFCFB-S",
    "PAYTM": "JD-PAYTMB-S",
}


def _amt_plain(a: str) -> str:
    return a.replace(",", "")


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


def generate(max_pos: int = 9000, max_neg: int = 1500) -> list[dict]:
    rows: list[dict] = []
    combos = list(product(MERCHANTS, AMOUNTS, LAST4, DATES))

    def upi_debit(m, a, l4, d, bank):
        sender = SENDERS[bank]
        sms = f"Rs.{a} debited from A/c XX{l4} to {m} via UPI on {d}. Avl Bal Rs.{BALANCES[0]}- {bank} Bank"
        gold = {
            "amount": _amt_plain(a),
            "merchant": m,
            "type": "EXPENSE",
            "account": l4,
            "balance": _amt_plain(BALANCES[0] or ""),
            "category": "Shopping",
        }
        return _row(bank, "upi_debit_v1", sms, sender, gold)

    def cc_spend(m, a, l4, d, bank):
        sender = SENDERS[bank]
        sms = f"Spent Rs.{a} on {bank} Bank CREDIT Card xx{l4} at {m} on {d}. Not you? Call 1930"
        gold = {
            "amount": _amt_plain(a),
            "merchant": m,
            "type": "CREDIT",
            "account": l4,
            "balance": None,
            "category": "Shopping",
        }
        return _row(bank, "cc_spend_v1", sms, sender, gold)

    def salary(m, a, l4, d, bank):
        sender = SENDERS[bank]
        sms = f"INR {a} credited to A/c XX{l4} on {d} by ACME PAYROLL PVT LTD. Avl Bal INR {BALANCES[2]}"
        gold = {
            "amount": _amt_plain(a),
            "merchant": "ACME PAYROLL PVT LTD",
            "type": "INCOME",
            "account": l4,
            "balance": _amt_plain(BALANCES[2] or ""),
            "category": "Salary",
        }
        return _row(bank, "salary_credit_v1", sms, sender, gold)

    def neft_in(m, a, l4, d, bank):
        sender = SENDERS[bank]
        sms = f"NEFT Cr Rs.{a} from {m} to your A/c XX{l4} on {d}. Bal Rs.{BALANCES[1]}"
        gold = {
            "amount": _amt_plain(a),
            "merchant": m,
            "type": "INCOME",
            "account": l4,
            "balance": _amt_plain(BALANCES[1] or ""),
            "category": "Transfer",
        }
        return _row(bank, "neft_credit_v1", sms, sender, gold)

    def imps_out(m, a, l4, d, bank):
        sender = SENDERS[bank]
        sms = f"IMPS: Rs.{a} transferred from A/c XX{l4} to {m} on {d}. Ref 9{l4}881. {bank}"
        gold = {
            "amount": _amt_plain(a),
            "merchant": m,
            "type": "TRANSFER",
            "account": l4,
            "balance": None,
            "category": "Transfer",
        }
        return _row(bank, "imps_transfer_v1", sms, sender, gold)

    def atm(m, a, l4, d, bank):
        sender = SENDERS[bank]
        sms = f"INR {a} withdrawn from A/c XX{l4} at ATM {m[:6]} on {d}. Avl Bal Rs.{BALANCES[3]}"
        gold = {
            "amount": _amt_plain(a),
            "merchant": "ATM",
            "type": "EXPENSE",
            "account": l4,
            "balance": _amt_plain(BALANCES[3] or ""),
            "category": "Cash",
        }
        return _row(bank, "atm_withdraw_v1", sms, sender, gold)

    def inr_space(m, a, l4, d, bank):
        sender = SENDERS[bank]
        sms = f"Payment Successful! INR {a} from A/c ****{l4} to {m} via {bank} NetBanking."
        gold = {
            "amount": _amt_plain(a),
            "merchant": m,
            "type": "EXPENSE",
            "account": l4,
            "balance": None,
            "category": "Shopping",
        }
        return _row(bank, "netbanking_v1", sms, sender, gold)

    def rupee_word(m, a, l4, d, bank):
        sender = SENDERS[bank]
        sms = f"{bank}: Rupees {a} debited for UPI/{m}/UPI on {d} A/c {l4}"
        gold = {
            "amount": _amt_plain(a),
            "merchant": m,
            "type": "EXPENSE",
            "account": l4,
            "balance": None,
            "category": "Shopping",
        }
        return _row(bank, "upi_rupees_v1", sms, sender, gold)

    def hinglish_upi(m, a, l4, d, bank):
        sender = SENDERS[bank]
        sms = (
            f"Aapke A/c XX{l4} se Rs.{a} {m} ko UPI se {d} ko debit ho gaye. "
            f"Avl Bal Rs.{BALANCES[0]}. {bank}"
        )
        gold = {
            "amount": _amt_plain(a),
            "merchant": m,
            "type": "EXPENSE",
            "account": l4,
            "balance": _amt_plain(BALANCES[0] or ""),
            "category": "Shopping",
        }
        return _row(bank, "upi_hinglish_v1", sms, sender, gold)

    def hindi_upi(m, a, l4, d, bank):
        sender = SENDERS[bank]
        sms = (
            f"{bank}: आपके खाते XX{l4} से Rs.{a} {m} को UPI द्वारा {d} को काटे गए। "
            f"उपलब्ध शेष Rs.{BALANCES[0]}"
        )
        gold = {
            "amount": _amt_plain(a),
            "merchant": m,
            "type": "EXPENSE",
            "account": l4,
            "balance": _amt_plain(BALANCES[0] or ""),
            "category": "Shopping",
        }
        return _row(bank, "upi_hindi_v1", sms, sender, gold)

    def hinglish_credit(m, a, l4, d, bank):
        sender = SENDERS[bank]
        sms = f"Aapke A/c XX{l4} mein Rs.{a} credit hua {m} se {d} ko. Bal Rs.{BALANCES[1]}. {bank}"
        gold = {
            "amount": _amt_plain(a),
            "merchant": m,
            "type": "INCOME",
            "account": l4,
            "balance": _amt_plain(BALANCES[1] or ""),
            "category": "Transfer",
        }
        return _row(bank, "credit_hinglish_v1", sms, sender, gold)

    builders = [
        upi_debit,
        cc_spend,
        salary,
        neft_in,
        imps_out,
        atm,
        inr_space,
        rupee_word,
        hinglish_upi,
        hindi_upi,
        hinglish_credit,
    ]
    banks = list(SENDERS.keys())

    n = 0
    for i, (m, a, l4, d) in enumerate(combos):
        if n >= max_pos:
            break
        bank = banks[i % len(banks)]
        builder = builders[i % len(builders)]
        rows.append(builder(m, a, l4, d, bank))
        n += 1

    neg_templates = [
        ("otp_v1", "{bank}: {otp} is your OTP for {m}. Do not share with anyone.", None),
        ("kyc_v1", "{bank}: Complete KYC for A/c XX{l4} by {d} to avoid restrictions. Ignore if done.", None),
        ("promo_v1", "Get 10% off at {m} with {bank} CREDIT Card. T&C apply.", None),
        ("failed_upi_v1", "UPI payment of Rs.{a} to {m} FAILED. Money not debited. {bank}", None),
        ("balance_only_v1", "{bank}: Your A/c XX{l4} Avl Bal is Rs.{a} as on {d}.", None),
        ("mandate_v1", "e-Mandate of Rs.{a} for {m} registered on A/c XX{l4}. {bank}", None),
        ("sms_promo_g", "Win a voucher at {m}! Click now. -P", None),
        ("otp_hinglish_v1", "{bank}: Aapka OTP {otp} hai {m} ke liye. Kisi ko share mat kariye.", None),
        ("failed_upi_hi_v1", "UPI Rs.{a} {m} ko FAIL hua. Paise nahi kate. {bank}", None),
    ]
    n_neg = 0
    for i, (m, a, l4, d) in enumerate(combos):
        if n_neg >= max_neg:
            break
        bank = banks[i % len(banks)]
        tid, tmpl, _ = neg_templates[i % len(neg_templates)]
        sender = SENDERS[bank] if not tid.endswith("_g") else "VM-PROMO-P"
        sms = tmpl.format(bank=bank, otp=str(100000 + i)[-6:], m=m, l4=l4, d=d, a=a)
        rows.append(_row(bank if sender != "VM-PROMO-P" else "PROMO", tid, sms, sender, None))
        n_neg += 1

    return rows
