"""Apache-friendly general instruct so SFT does not collapse to JSON-only."""

from __future__ import annotations

import hashlib

from data.schema import example

PAIRS = [
    (
        "general/json_format",
        "Reply with a JSON object {\"ok\": true, \"n\": 3} and nothing else.",
        '{"ok":true,"n":3}',
    ),
    (
        "general/refuse_unknown",
        "What is the user's bank password?",
        "I do not have that. FinnDot never asks for bank passwords. Use your bank app instead.",
    ),
    (
        "general/sip",
        "Explain a SIP in one short paragraph with no markdown.",
        "A SIP is a systematic investment plan: you invest a fixed amount on a schedule, usually into a mutual fund. It averages purchase price over time. It is not a guaranteed return and you can pause it in the fund app.",
    ),
    (
        "general/emi",
        "What is EMI in two sentences?",
        "EMI is the equated monthly installment you pay to repay a loan. It mixes principal and interest so the loan ends on a fixed date if you pay on time.",
    ),
    (
        "general/emergency",
        "How many months of expenses should an emergency fund cover?",
        "A common target is three to six months of essential expenses in a liquid account. Start with one month if that is what you can do now.",
    ),
    (
        "general/credit_util",
        "What is a healthy credit card utilisation?",
        "Keeping utilisation under about 30 percent of the limit is a common guideline. Paying the statement in full still matters more than the ratio alone.",
    ),
    (
        "general/plain",
        "Write a two-line reminder to review subscriptions. No asterisks.",
        "Open your bank and UPI apps and list every auto-pay. Cancel anything you did not use this month.",
    ),
    (
        "general/inr",
        "Convert 2 lakh rupees to a plain number.",
        "200000",
    ),
    (
        "general/fd",
        "Is a fixed deposit risk-free in one sentence?",
        "Bank fixed deposits are low risk relative to equities but inflation and bank limits still apply.",
    ),
    (
        "general/budget",
        "Give a 50-30-20 budget split in one line.",
        "About 50 percent needs, 30 percent wants, 20 percent saving or debt payoff, adjusted to your rent and income.",
    ),
    (
        "general/sip_hi",
        "SIP kya hota hai, ek chhote paragraph mein, bina markdown ke.",
        "SIP ek systematic investment plan hai: aap ek fixed amount niyam se, aksar mutual fund mein, daalte ho. Price time ke saath average hota hai. Return guaranteed nahi hai, fund app se pause kar sakte ho.",
    ),
    (
        "general/emi_hi",
        "EMI kya hai, do vakya.",
        "EMI wo monthly installment hai jisse loan chhutta hai. Isme principal aur interest milte hain, time par payment se loan date par khatam hota hai.",
    ),
    (
        "general/bbps",
        "What is BBPS in one short paragraph?",
        "BBPS is Bharat Bill Payment System: a NPCI network to pay utility and other bills across banks and apps with a receipt reference. It does not mean the biller was paid twice if you see both BBPS and bank SMS for one payment.",
    ),
    (
        "general/upi_limit",
        "What is a common UPI per-day limit people hit?",
        "Many banks set UPI limits around one lakh rupees per day for P2M or P2P, but your bank app shows the exact cap. Limits differ for new accounts and overnight hours.",
    ),
    (
        "general/80c",
        "Name three Section 80C style options in one line.",
        "ELSS mutual funds, PPF, and life insurance premiums are common 80C routes; check current tax rules before investing.",
    ),
    (
        "general/npci",
        "What does NPCI do in one sentence?",
        "NPCI operates retail payment systems in India including UPI, IMPS, and BBPS so banks and apps can move money on shared rails.",
    ),
    (
        "general/wallet_vs_bank",
        "Wallet balance vs bank balance — one clear difference.",
        "Bank SMS reflects your account; wallet SMS reflects prepaid wallet balance, which is separate unless you just loaded it from the bank.",
    ),
    (
        "general/sip_vs_emi",
        "SIP vs EMI in two sentences.",
        "SIP invests a fixed amount on a schedule, usually into mutual funds. EMI repays a loan; it is an outflow you owe, not an investment contribution.",
    ),
    (
        "general/hinglish_budget",
        "Simple budget tip in Hinglish, two lines.",
        "Pehle rent, EMI, aur bills nikaalo. Jo bache usme 20 percent saving ya emergency fund ki taraf rakhne ki koshish karo.",
    ),
]


def generate(n: int = 3500) -> list[dict]:
    rows: list[dict] = []
    i = 0
    while len(rows) < n:
        tid, user, assistant = PAIRS[i % len(PAIRS)]
        eid = hashlib.sha256(f"{tid}|{i}".encode()).hexdigest()[:16]
        rows.append(
            example(
                example_id=eid,
                split_key=f"general|{tid}",
                bank="GENERAL",
                template_id=tid,
                task="general",
                system="You are FinnDot AI, a concise personal finance assistant. Plain text unless JSON is requested.",
                user=user if i % 7 else user + " Keep it short.",
                assistant=assistant,
            )
        )
        i += 1
    return rows
