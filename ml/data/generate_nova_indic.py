"""Expand Indic/Hinglish *surface* text with Nova Pro. Gold labels stay ours.

Nova is not the trainer: QLoRA still runs on Qwen3. Nova only proposes SMS
paraphrases and translations of our coach Q/A. A row is kept only if amounts,
last-4, merchant, and ledger figures still match the seed gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from data.generate_finance_chat import INDIC_QA, LEDGERS, QA, _ground, _system
from data.generate_synthetic_sms import BALANCES, LAST4, MERCHANTS, SENDERS, _amt_plain, _row
from data.nova_client import converse, parse_json_list, parse_json_obj
from data.schema import example

LANGS = ("hi", "hinglish", "ta", "te", "mr", "bn")

SMS_PROMPT = """You write Indian bank SMS paraphrases for a privacy-safe synthetic dataset.
Return ONLY a JSON array of objects: [{{"lang": "...", "sms": "..."}}]
Languages to use: {langs}
Rules:
- Keep these exact tokens somewhere in each sms: amount {amount}, merchant {merchant}, last4 {last4}
- Do not invent a different amount, merchant, or account
- Mix Devanagari/Tamil/Telugu/Marathi/Bengali and Hinglish; bank SMS style, not literary
- No user names, phone numbers, or real account numbers other than XX{last4}
Seed SMS:
{seed}
"""

CHAT_PROMPT = """Translate this FinnDot finance-coach turn into {lang}.
Return ONLY JSON: {{"user":"...","assistant":"..."}}
Keep every rupee figure and merchant name exactly as written (same digits).
Plain text, no markdown.
User:
{user}
Assistant:
{assistant}
"""


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def sms_ok(sms: str, gold: dict | None) -> bool:
    if not sms or len(sms) < 20:
        return False
    if not gold:
        return True
    amt = _digits(str(gold.get("amount") or ""))
    if amt and amt not in _digits(sms):
        return False
    acc = str(gold.get("account") or "")
    if acc and acc not in sms:
        return False
    merch = str(gold.get("merchant") or "")
    if merch and merch.casefold() not in sms.casefold():
        return False
    return True


HINDI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def chat_ok(assistant: str, must: list[str]) -> bool:
    blob = _digits(assistant.translate(HINDI_DIGITS))
    matched = 0
    for n in must:
        d = _digits(n)
        if not d:
            continue
        if d in blob:
            matched += 1
        elif "." in n:
            integer_part = _digits(n.split(".")[0])
            if integer_part and integer_part in blob:
                matched += 1
    return matched >= 3


AMOUNTS = ["499.00", "1250.00", "3500.00", "89.00", "15000.00", "247.50"]


def expand_sms(n: int) -> list[dict]:
    seeds = []
    bank_list = list(SENDERS.keys())
    for ai, amt in enumerate(AMOUNTS):
        for mi, m in enumerate(MERCHANTS[:12]):
            i = ai * 12 + mi
            l4, bank = LAST4[i % len(LAST4)], bank_list[i % len(bank_list)]
            gold = {
                "amount": _amt_plain(amt),
                "merchant": m,
                "type": "EXPENSE",
                "account": l4,
                "balance": _amt_plain(BALANCES[0] or ""),
                "category": "Shopping",
            }
            seed = (
                f"{bank}: Rs.{amt} debited from A/c XX{l4} to {m} via UPI. "
                f"Avl Bal Rs.{BALANCES[0]}"
            )
            seeds.append((bank, m, amt, l4, gold, seed))

    rows: list[dict] = []
    for si, (bank, m, a, l4, gold, seed) in enumerate(seeds):
        if len(rows) >= n:
            break
        print(f"[sms] seed {si+1}/{len(seeds)}  bank={bank}  have={len(rows)}/{n}", flush=True)
        prompt = SMS_PROMPT.format(
            langs=", ".join(LANGS),
            amount=a,
            merchant=m,
            last4=l4,
            seed=seed,
        )
        try:
            items = parse_json_list(converse(prompt))
        except Exception as e:
            print("nova sms skip", e, flush=True)
            continue
        kept = 0
        for j, item in enumerate(items):
            if len(rows) >= n:
                break
            if not isinstance(item, dict):
                continue
            sms = str(item.get("sms") or "").strip()
            lang = str(item.get("lang") or "und")[:12]
            if not sms_ok(sms, gold):
                continue
            tid = f"nova_sms/{lang}/{bank}/{m}/{a}"
            row = _row(bank, tid, sms, SENDERS[bank], gold)
            row["id"] = hashlib.sha256(f"{tid}|{sms}".encode()).hexdigest()[:16]
            row["split_key"] = f"{bank}|{tid}|{j}"
            rows.append(row)
            kept += 1
        print(f"  -> kept {kept} from Nova response", flush=True)
    print(f"[sms] DONE total={len(rows)}", flush=True)
    return rows


def expand_chat(n: int) -> list[dict]:
    rows: list[dict] = []
    i = 0
    pool = QA + INDIC_QA
    while len(rows) < n:
        led = LEDGERS[i % len(LEDGERS)]
        qid, user, fn = pool[i % len(pool)]
        lang = LANGS[i % len(LANGS)]
        assistant = fn(led)
        must = _ground(led)
        prompt = CHAT_PROMPT.format(lang=lang, user=user, assistant=assistant)
        i += 1
        if i % 10 == 0:
            print(f"[chat] call {i}  have={len(rows)}/{n}", flush=True)
        try:
            obj = parse_json_obj(converse(prompt))
        except Exception as e:
            print(f"nova chat skip ({i})", e, flush=True)
            continue
        if not obj:
            continue
        u = str(obj.get("user") or "").strip()
        a = str(obj.get("assistant") or "").strip()
        if not u or not chat_ok(a, must):
            continue
        tid = f"coach/nova/{lang}/{led['id']}/{qid}"
        eid = hashlib.sha256(f"{tid}|{u}".encode()).hexdigest()[:16]
        rows.append(
            example(
                example_id=eid,
                split_key=f"coach|{tid}",
                bank="COACH",
                template_id=tid,
                task="chat",
                system=_system(led),
                user=u,
                assistant=a,
                must_ground=must,
            )
        )
    print(f"[chat] DONE total={len(rows)}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Nova Pro data factory → verified Indic jsonl")
    p.add_argument("--out", type=Path, default=Path("data/out/nova_indic.jsonl"))
    p.add_argument("--sms", type=int, default=400)
    p.add_argument("--chat", type=int, default=400)
    args = p.parse_args()
    rows = expand_sms(args.sms) + expand_chat(args.chat)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(rows), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
