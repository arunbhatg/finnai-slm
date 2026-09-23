"""Extract ParserTestCase gold from parser-core Kotlin tests."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from data.schema import example

STRING = r'"(?:\\.|[^"\\])*"'
IDENT = r"[A-Za-z_][A-Za-z0-9_]*"

TYPE_MAP = {
    "INCOME": "INCOME",
    "EXPENSE": "EXPENSE",
    "CREDIT": "CREDIT",
    "TRANSFER": "TRANSFER",
    "INVESTMENT": "INVESTMENT",
}


def _unquote(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        import ast

        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s[1:-1]
    return s


def _balanced_calls(src: str, fn: str) -> list[str]:
    out: list[str] = []
    needle = fn + "("
    i = 0
    while True:
        j = src.find(needle, i)
        if j < 0:
            break
        k = j + len(needle)
        depth = 1
        start = k
        while k < len(src) and depth:
            if src[k] == "(":
                depth += 1
            elif src[k] == ")":
                depth -= 1
            k += 1
        out.append(src[start : k - 1])
        i = k
    return out


def _field(block: str, name: str) -> str | None:
    m = re.search(rf"{name}\s*=\s*({STRING}|{IDENT}(?:\.{IDENT})*|true|false|null)", block)
    if not m:
        return None
    return m.group(1)


def _bigdecimal(block: str, name: str) -> str | None:
    m = re.search(rf'{name}\s*=\s*BigDecimal\(\s*({STRING})\s*\)', block)
    if m:
        return _unquote(m.group(1))
    return None


def _expected(block: str) -> dict | None:
    inner = _balanced_calls(block, "ExpectedTransaction")
    if not inner:
        return None
    e = inner[0]
    type_raw = _field(e, "type") or ""
    tname = type_raw.split(".")[-1]
    gold = {
        "amount": _bigdecimal(e, "amount"),
        "type": TYPE_MAP.get(tname),
        "merchant": _unquote(_field(e, "merchant") or "") or None if _field(e, "merchant") else None,
        "account": _unquote(_field(e, "accountLast4") or "") or None if _field(e, "accountLast4") else None,
        "balance": _bigdecimal(e, "balance"),
        "category": None,
    }
    if _field(e, "merchant"):
        gold["merchant"] = _unquote(_field(e, "merchant") or "")
    if _field(e, "accountLast4"):
        gold["account"] = _unquote(_field(e, "accountLast4") or "")
    return gold


def extract_file(path: Path) -> list[dict]:
    src = path.read_text(encoding="utf-8")
    bank = path.stem.replace("Test", "").replace("Parser", "")
    rows: list[dict] = []
    for idx, block in enumerate(_balanced_calls(src, "ParserTestCase")):
        name = _unquote(_field(block, "name") or f"case_{idx}")
        message = _field(block, "message")
        sender = _field(block, "sender")
        if not message or not sender:
            continue
        sms = _unquote(message)
        sender_s = _unquote(sender)
        should = (_field(block, "shouldParse") or "true").strip()
        gold = _expected(block) if should != "false" else None
        if should == "false":
            gold = None
        elif gold is None:
            continue
        template_id = f"parser/{path.stem}/{idx}"
        split_key = f"{bank}|{template_id}"
        eid = hashlib.sha256(f"{template_id}|{sms}".encode()).hexdigest()[:16]
        rows.append(
            example(
                example_id=eid,
                split_key=split_key,
                bank=bank,
                template_id=template_id,
                task="sms",
                sms=sms,
                sender=sender_s,
                gold=gold,
            )
        )
    return rows


def extract_repo(repo_root: Path) -> list[dict]:
    tests = Path(repo_root) / "parser-core" / "src" / "test" / "kotlin"
    rows: list[dict] = []
    for path in sorted(tests.rglob("Test*Parser.kt")):
        rows.extend(extract_file(path))
    return rows


if __name__ == "__main__":
    import json
    import sys

    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    data = extract_repo(root)
    print(json.dumps({"count": len(data), "banks": sorted({r["bank"] for r in data})}, indent=2))
