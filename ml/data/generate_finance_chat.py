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
]


def generate(n: int = 3000) -> list[dict]:
    rows: list[dict] = []
    i = 0
    pool = QA + INDIC_QA
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
