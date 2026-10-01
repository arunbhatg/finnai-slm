"""Manual Indic/Hinglish bank SMS templates (no Bedrock). Gold stays deterministic.

Covers HI / Hinglish / TA / TE / MR / BN surface forms for common debit/credit patterns.
Use this for OSS-reproducible v3 data; Nova expansion remains optional on top.
"""

from __future__ import annotations

import hashlib
from itertools import product

from data.generate_synthetic_sms import LAST4, MERCHANTS, SENDERS, _amt_plain
from data.schema import example

AMOUNTS = ["149.00", "499.00", "1,250.00", "3,500.00", "12,000.00", "247.50"]
BALANCES = ["5,500.00", "18,200.75", "42,000.00"]
DATES = ["05-01-26", "18-02-26", "11-03-26", "27-04-26"]

# (lang, template_id, sms_template, type, category)
# Placeholders: {bank} {amount} {merchant} {l4} {date} {balance}
TEMPLATES = [
    (
        "hi",
        "indic_hi_debit_v1",
        "{bank}: आपके खाते XX{l4} से Rs.{amount} {merchant} को UPI द्वारा {date} को काटे गए। शेष Rs.{balance}",
        "EXPENSE",
        "Shopping",
    ),
    (
        "hinglish",
        "indic_hinglish_debit_v1",
        "{bank}: Aapke A/c XX{l4} se Rs.{amount} {merchant} ko UPI se {date} ko debit ho gaya. Avl Bal Rs.{balance}",
        "EXPENSE",
        "Shopping",
    ),
    (
        "ta",
        "indic_ta_debit_v1",
        "{bank}: உங்கள் கணக்கு XX{l4} இலிருந்து Rs.{amount} {merchant} க்கு UPI மூலம் {date} அன்று பற்று. இருப்பு Rs.{balance}",
        "EXPENSE",
        "Shopping",
    ),
    (
        "te",
        "indic_te_debit_v1",
        "{bank}: మీ A/c XX{l4} నుండి Rs.{amount} {merchant} కు UPI ద్వారా {date} నాడు డెబిట్. బ్యాలెన్స్ Rs.{balance}",
        "EXPENSE",
        "Shopping",
    ),
    (
        "mr",
        "indic_mr_debit_v1",
        "{bank}: तुमच्या खाते XX{l4} मधून Rs.{amount} {merchant} ला UPI द्वारे {date} रोजी डेबिट. शिल्लक Rs.{balance}",
        "EXPENSE",
        "Shopping",
    ),
    (
        "bn",
        "indic_bn_debit_v1",
        "{bank}: আপনার A/c XX{l4} থেকে Rs.{amount} {merchant}-এ UPI-তে {date}-এ ডেবিট। ব্যালেন্স Rs.{balance}",
        "EXPENSE",
        "Shopping",
    ),
    (
        "hi",
        "indic_hi_credit_v1",
        "{bank}: आपके खाते XX{l4} में Rs.{amount} {merchant} से {date} को जमा। उपलब्ध शेष Rs.{balance}",
        "INCOME",
        "Transfer",
    ),
    (
        "hinglish",
        "indic_hinglish_credit_v1",
        "{bank}: A/c XX{l4} mein Rs.{amount} {merchant} se credit hua {date} ko. Bal Rs.{balance}",
        "INCOME",
        "Transfer",
    ),
    (
        "ta",
        "indic_ta_credit_v1",
        "{bank}: கணக்கு XX{l4} க்கு Rs.{amount} {merchant} இலிருந்து {date} வரவு. இருப்பு Rs.{balance}",
        "INCOME",
        "Transfer",
    ),
    (
        "te",
        "indic_te_imps_v1",
        "{bank}: IMPS Rs.{amount} A/c XX{l4} నుండి {merchant} కు {date}. Ref 8{l4}221",
        "TRANSFER",
        "Transfer",
    ),
    (
        "mr",
        "indic_mr_neft_v1",
        "{bank}: NEFT जमा Rs.{amount} {merchant} कडून A/c XX{l4} मध्ये {date}. शिल्लक Rs.{balance}",
        "INCOME",
        "Transfer",
    ),
    (
        "bn",
        "indic_bn_failed_v1",
        "{bank}: UPI Rs.{amount} {merchant}-এ ব্যর্থ। টাকা কাটেনি।",
        None,
        None,
    ),
    (
        "hi",
        "indic_hi_otp_v1",
        "{bank}: आपका OTP {otp} है {merchant} के लिए। किसी से साझा न करें।",
        None,
        None,
    ),
    (
        "hinglish",
        "indic_hinglish_otp_v1",
        "{bank}: Aapka OTP {otp} hai {merchant} ke liye. Share mat kariye.",
        None,
        None,
    ),
]


def _eid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def generate(max_rows: int = 2500) -> list[dict]:
    rows: list[dict] = []
    banks = list(SENDERS.keys())
    merchants = MERCHANTS[:16]
    combos = list(product(merchants, AMOUNTS, LAST4, DATES))

    i = 0
    while len(rows) < max_rows and i < len(combos) * len(TEMPLATES):
        m, a, l4, d = combos[i % len(combos)]
        lang, tid, tmpl, typ, cat = TEMPLATES[i % len(TEMPLATES)]
        bank = banks[i % len(banks)]
        bal = BALANCES[i % len(BALANCES)]
        sms = tmpl.format(
            bank=bank,
            amount=a,
            merchant=m,
            l4=l4,
            date=d,
            balance=bal,
            otp=str(300000 + i)[-6:],
        )
        if typ is None:
            gold = None
        else:
            gold = {
                "amount": _amt_plain(a),
                "merchant": m,
                "type": typ,
                "account": l4,
                "balance": None if "IMPS" in tid and "te" in lang else _amt_plain(bal),
                "category": cat,
            }
            if tid.endswith("imps_v1"):
                gold["balance"] = None
        full_tid = f"{tid}/{lang}"
        rows.append(
            example(
                example_id=_eid(full_tid, sms, SENDERS[bank]),
                split_key=f"{bank}|{full_tid}",
                bank=bank,
                template_id=full_tid,
                task="sms",
                sms=sms,
                sender=SENDERS[bank],
                gold=gold,
            )
        )
        i += 1
    return rows


def frozen_eval_slice(n: int = 200) -> list[dict]:
    raw = generate(max_rows=max(n * 2, 100))
    out = []
    for i, r in enumerate(raw):
        if len(out) >= n:
            break
        r = dict(r)
        r["template_id"] = f"eval/indic/{r['template_id']}/{i}"
        r["split_key"] = f"eval|{r['template_id']}"
        r["id"] = _eid(r["template_id"], r.get("sms", ""))
        out.append(r)
    return out
