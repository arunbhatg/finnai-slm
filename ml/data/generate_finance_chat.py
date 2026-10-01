"""Synthetic FinnDot finance-coach chats. Answers only use numbers from the ledger."""

from __future__ import annotations

import hashlib

from data.schema import example

COACH_SYSTEM_PREFIX = (
    "You are FinnDot AI, a personal finance coach built into an expense tracker app. "
    "Your job is to help users understand spending, find money leaks, and make smarter money decisions."
)

COACH_GUIDELINES = """
Response guidelines:
- Lead with the insight, then supporting numbers from the data above
- Be direct about money leaks: "You spent X on Y — consider Z to save"
- Give 1-3 actionable steps, not generic advice
- Be supportive, never judgmental about spending habits
- Use the rupee symbol for amounts
- Keep answers concise: 2-4 short paragraphs or a numbered list
- Plain text only — no markdown, asterisks, or special formatting
- For lists use simple dashes or numbers
- If data is missing, say so and suggest what the user can track
""".strip()

LEDGERS = [
    {
        "id": "food_heavy",
        "date": "20 Sep 2026",
        "expense": "18240.00",
        "income": "65000.00",
        "net": "46760.00",
        "txn": "42",
        "day": "20",
        "days": "30",
        "avg": "912.00",
        "cats": [("Food", "7200.00", 39), ("Travel", "4100.00", 22), ("Shopping", "3800.00", 21), ("Bills", "3140.00", 18)],
        "subs_n": "3",
        "subs_amt": "1497.00",
        "recent": [
            ("Sep 18", "SWIGGY", "420.00", "Food"),
            ("Sep 17", "UBER", "310.00", "Travel"),
            ("Sep 16", "AMAZON", "1999.00", "Shopping"),
            ("Sep 15", "NETFLIX", "499.00", "Bills"),
        ],
        "largest": ("AMAZON", "1999.00"),
        "freq": ("SWIGGY", "8"),
    },
    {
        "id": "rent_month",
        "date": "05 Sep 2026",
        "expense": "41200.00",
        "income": "80000.00",
        "net": "38800.00",
        "txn": "18",
        "day": "5",
        "days": "30",
        "avg": "8240.00",
        "cats": [("Housing", "25000.00", 61), ("Food", "6200.00", 15), ("Transport", "4000.00", 10), ("Other", "6000.00", 14)],
        "subs_n": "2",
        "subs_amt": "798.00",
        "recent": [
            ("Sep 1", "LANDLORD", "25000.00", "Housing"),
            ("Sep 2", "BIGBASKET", "2100.00", "Food"),
            ("Sep 3", "IRCTC", "1850.00", "Travel"),
        ],
        "largest": ("LANDLORD", "25000.00"),
        "freq": ("BIGBASKET", "3"),
    },
    {
        "id": "low_income",
        "date": "12 Sep 2026",
        "expense": "22100.00",
        "income": "28000.00",
        "net": "5900.00",
        "txn": "31",
        "day": "12",
        "days": "30",
        "avg": "1841.67",
        "cats": [("Food", "8100.00", 37), ("Bills", "5400.00", 24), ("Shopping", "4900.00", 22), ("Travel", "3700.00", 17)],
        "subs_n": "4",
        "subs_amt": "2196.00",
        "recent": [
            ("Sep 11", "SPOTIFY", "199.00", "Bills"),
            ("Sep 10", "ZOMATO", "560.00", "Food"),
            ("Sep 9", "FLIPKART", "2499.00", "Shopping"),
        ],
        "largest": ("FLIPKART", "2499.00"),
        "freq": ("ZOMATO", "6"),
    },
    {
        "id": "upi_heavy",
        "date": "28 Sep 2026",
        "expense": "15680.00",
        "income": "72000.00",
        "net": "56320.00",
        "txn": "55",
        "day": "28",
        "days": "30",
        "avg": "560.00",
        "cats": [("Food", "5400.00", 34), ("Travel", "3200.00", 20), ("Bills", "2900.00", 19), ("Shopping", "4180.00", 27)],
        "subs_n": "5",
        "subs_amt": "1845.00",
        "recent": [
            ("Sep 27", "SWIGGY", "380.00", "Food"),
            ("Sep 26", "PHONEPE", "1200.00", "Transfer"),
            ("Sep 25", "BESCOM", "2100.00", "Bills"),
            ("Sep 24", "UBER", "245.00", "Travel"),
        ],
        "largest": ("BESCOM", "2100.00"),
        "freq": ("SWIGGY", "11"),
    },
    {
        "id": "emi_month",
        "date": "10 Sep 2026",
        "expense": "38500.00",
        "income": "95000.00",
        "net": "56500.00",
        "txn": "22",
        "day": "10",
        "days": "30",
        "avg": "3850.00",
        "cats": [("EMI", "18500.00", 48), ("Housing", "12000.00", 31), ("Food", "4800.00", 12), ("Other", "3200.00", 9)],
        "subs_n": "2",
        "subs_amt": "648.00",
        "recent": [
            ("Sep 5", "BAJAJ_FINSERV", "12500.00", "EMI"),
            ("Sep 5", "HDFC_BANK_EMI", "6000.00", "EMI"),
            ("Sep 7", "BIGBASKET", "1800.00", "Food"),
        ],
        "largest": ("BAJAJ_FINSERV", "12500.00"),
        "freq": ("BIGBASKET", "2"),
    },
]


def _system(led: dict) -> str:
    cats = "\n".join(f"- {n}: ₹{a} ({p}%)" for n, a, p in led["cats"])
    recent = "\n".join(
        f"- {d}: {m} -₹{a} ({c})" for d, m, a, c in led["recent"]
    )
    return f"""{COACH_SYSTEM_PREFIX}

Current Financial Overview ({led["date"]}):
- This month: ₹{led["expense"]} spent, ₹{led["income"]} income
- Net: ₹{led["net"]}
- {led["txn"]} transactions (Day {led["day"]}/{led["days"]})
- Daily average spending: ₹{led["avg"]}

Top spending categories:
{cats}

Active subscriptions: {led["subs_n"]} services (₹{led["subs_amt"]}/month)

Recent Transactions (Last 14 days):
{recent}

Largest expense this month: {led["largest"][0]} ₹{led["largest"][1]}
Most visited merchant: {led["freq"][0]} ({led["freq"][1]} times)

{COACH_GUIDELINES}"""


def _ground(led: dict) -> list[str]:
    nums = [led["expense"], led["income"], led["net"], led["avg"], led["subs_amt"], led["largest"][1]]
    nums += [a for _, a, _ in led["cats"]]
    nums += [a for _, _, a, _ in led["recent"]]
    return [n.replace(",", "") for n in nums]


QA = [
    (
        "leak",
        "Where am I losing money?",
        lambda led: (
            f"You spent ₹{led['cats'][0][1]} on {led['cats'][0][0]} — that is {led['cats'][0][2]}% of ₹{led['expense']} this month. "
            f"{led['freq'][0]} showed up {led['freq'][1]} times. "
            f"1. Cap {led['cats'][0][0]} at a weekly limit. 2. Review the ₹{led['subs_amt']} in subscriptions. "
            f"3. Recheck the ₹{led['largest'][1]} {led['largest'][0]} charge."
        ),
    ),
    (
        "afford",
        "Can I afford a 5000 rupee purchase this month?",
        lambda led: (
            f"Net so far is ₹{led['net']} with income ₹{led['income']} and spend ₹{led['expense']}. "
            f"A ₹5000 buy is possible if you cut {led['cats'][0][0]} (₹{led['cats'][0][1]}). "
            f"Daily average is already ₹{led['avg']}."
        ),
    ),
    (
        "subs",
        "Are my subscriptions too high?",
        lambda led: (
            f"You have {led['subs_n']} subscriptions totaling ₹{led['subs_amt']} per month. "
            f"Against ₹{led['income']} income that is a small slice, but stack it with {led['cats'][0][0]} at ₹{led['cats'][0][1]}. "
            f"Cancel one unused service this week."
        ),
    ),
    (
        "food",
        "How do I spend less on food?",
        lambda led: (
            f"Food-related spend is in your top categories. Total spend is ₹{led['expense']}. "
            f"{led['freq'][0]} is your most visited merchant ({led['freq'][1]} visits). "
            f"Batch cook twice a week and set a ₹ cap below your {led['cats'][0][0]} total of ₹{led['cats'][0][1]}."
        ),
    ),
]

# Train-only Indic/Hinglish asks. Frozen English eval in frozen_eval_set() still uses QA only.
INDIC_QA = [
    (
        "leak_hi",
        "Paisa kahan nikal raha hai?",
        lambda led: (
            f"Is mahine ₹{led['cats'][0][1]} {led['cats'][0][0]} par gaya — kul ₹{led['expense']} ka {led['cats'][0][2]}%. "
            f"{led['freq'][0]} {led['freq'][1]} baar aaya. "
            f"1. {led['cats'][0][0]} par weekly cap lagao. 2. ₹{led['subs_amt']} ki subscriptions dekho. "
            f"3. ₹{led['largest'][1]} {led['largest'][0]} charge check karo."
        ),
    ),
    (
        "afford_hi",
        "Kya is mahine 5000 ka kharcha chalega?",
        lambda led: (
            f"Abhi net ₹{led['net']} hai, income ₹{led['income']} aur spend ₹{led['expense']}. "
            f"₹5000 tab theek hai jab {led['cats'][0][0]} (₹{led['cats'][0][1]}) kam karo. "
            f"Roz ka average ₹{led['avg']} hai."
        ),
    ),
    (
        "food_hi",
        "Khane pe kam kaise kharchun?",
        lambda led: (
            f"Khane wala kharch top categories mein hai. Kul spend ₹{led['expense']}. "
            f"{led['freq'][0]} sabse zyada ({led['freq'][1]} baar). "
            f"Hafte mein do baar ghar ka khana banao aur {led['cats'][0][0]} ko ₹{led['cats'][0][1]} se neeche rakho."
        ),
    ),
    (
        "subs_hi",
        "Subscriptions zyada toh nahi?",
        lambda led: (
            f"Aapke {led['subs_n']} subscriptions milakar ₹{led['subs_amt']} per month hain. "
            f"Income ₹{led['income']} ke against chhota hissa hai, lekin {led['cats'][0][0]} ₹{led['cats'][0][1]} ke saath jod lo. "
            f"Jo use nahi kiya usko cancel karo."
        ),
    ),
    (
        "emi_hi",
        "EMI kitna pressure de rahi hai?",
        lambda led: (
            f"Kul kharch ₹{led['expense']} hai, income ₹{led['income']}, net ₹{led['net']}. "
            f"Sabse badi category {led['cats'][0][0]} ₹{led['cats'][0][1]} ({led['cats'][0][2]}%). "
            f"Agar EMI hai toh us category ko pehle freeze karo, baaki discretionary cut karo."
        ),
    ),
    (
        "summary_hi",
        "Is mahine ka short summary do.",
        lambda led: (
            f"Spend ₹{led['expense']}, income ₹{led['income']}, net ₹{led['net']}, "
            f"{led['txn']} transactions, daily avg ₹{led['avg']}. "
            f"Top: {led['cats'][0][0]} ₹{led['cats'][0][1]}. Largest: {led['largest'][0]} ₹{led['largest'][1]}."
        ),
    ),
    (
        "bills_hi",
        "Bills pe kitna gaya?",
        lambda led: (
            f"Bills-related numbers ledger mein categories se dekho; kul spend ₹{led['expense']}. "
            f"Subscriptions alone ₹{led['subs_amt']} hain. "
            f"BBPS aur recharge last 14 days mein recent list se milenge — total spend se cross-check karo."
        ),
    ),
]

# Adversarial / grounding hard cases — assistant must not invent ledger figures.
HARD_QA = [
    (
        "wrong_month",
        "How much did I spend in January 2024?",
        lambda led: (
            f"I only have this month's ledger ending {led['date']}: spend ₹{led['expense']}, income ₹{led['income']}. "
            f"I do not have January 2024 figures in the prompt, so I cannot invent them. "
            f"Open that month in the app or ask again after it loads."
        ),
    ),
    (
        "invent_refuse",
        "Tell me the exact UPI spend yesterday including merchant XYZ999.",
        lambda led: (
            f"The ledger shows total spend ₹{led['expense']} and recent merchants like {led['recent'][0][1]}, "
            f"but not a merchant XYZ999 or a separate yesterday total. "
            f"I will not invent ₹ amounts that are not in your data."
        ),
    ),
]


def generate(n: int = 9000) -> list[dict]:
    rows: list[dict] = []
    i = 0
    pool = QA + INDIC_QA + HARD_QA
    while len(rows) < n:
        led = LEDGERS[i % len(LEDGERS)]
        qid, user, fn = pool[i % len(pool)]
        template_id = f"coach/{led['id']}/{qid}"
        system = _system(led)
        assistant = fn(led)
        eid = hashlib.sha256(f"{template_id}|{i}".encode()).hexdigest()[:16]
        rows.append(
            example(
                example_id=eid,
                split_key=f"coach|{template_id}",
                bank="COACH",
                template_id=template_id,
                task="chat",
                system=system,
                user=user,
                assistant=assistant,
                must_ground=_ground(led),
            )
        )
        i += 1
    return rows


def frozen_eval_set() -> list[dict]:
    """50 items: unique (ledger, question) rotations, never mixed into SFT by id prefix eval/."""
    rows = []
    i = 0
    while len(rows) < 50:
        led = LEDGERS[i % len(LEDGERS)]
        qid, user, fn = QA[i % len(QA)]
        # Rotate user phrasing so eval is not a copy of train questions only
        user_i = f"{user} (check my numbers.)" if i >= 12 else user
        template_id = f"eval/coach/{led['id']}/{qid}/{i}"
        system = _system(led)
        assistant = fn(led)
        eid = hashlib.sha256(template_id.encode()).hexdigest()[:16]
        rows.append(
            example(
                example_id=eid,
                split_key=f"eval|{template_id}",
                bank="COACH",
                template_id=template_id,
                task="chat",
                system=system,
                user=user_i,
                assistant=assistant,
                must_ground=_ground(led),
            )
        )
        i += 1
    return rows
