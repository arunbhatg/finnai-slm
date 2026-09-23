"""Bedrock Nova Pro as a *data factory* (surface text only), not a trainer."""

from __future__ import annotations

import json
import os
import re
import sys

import boto3
from botocore.config import Config

DEFAULT_MODEL = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")
DEFAULT_REGION = os.environ.get("NOVA_REGION", os.environ.get("AWS_REGION", "us-east-1"))

_BOTO_CFG = Config(
    read_timeout=60,
    connect_timeout=10,
    retries={"max_attempts": 3, "mode": "adaptive"},
)
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=DEFAULT_REGION, config=_BOTO_CFG)
    return _client


def converse(prompt: str, *, max_tokens: int = 1200, temperature: float = 0.4) -> str:
    import time as _time
    client = _get_client()
    for attempt in range(5):
        try:
            resp = client.converse(
                modelId=DEFAULT_MODEL,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            parts = resp.get("output", {}).get("message", {}).get("content", [])
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
            _time.sleep(1.0)
            return text
        except client.exceptions.ThrottlingException:
            wait = 2 ** attempt * 3
            print(f"  throttled, backoff {wait}s", flush=True)
            _time.sleep(wait)
    raise RuntimeError("Nova throttled after 5 retries")


def parse_json_obj(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    obj = json.loads(text[start : end + 1])
    return obj if isinstance(obj, dict) else {}


def parse_json_list(text: str) -> list:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        obj = parse_json_obj(text)
        return [obj] if obj else []
    data = json.loads(text[start : end + 1])
    return data if isinstance(data, list) else []
