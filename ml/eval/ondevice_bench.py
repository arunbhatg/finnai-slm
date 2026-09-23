"""On-device latency notes. Fill mediapipe_qwen25_baseline.json on a phone before swapping runtimes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

TEMPLATE = {
    "captured_at": None,
    "device": {
        "marketing_name": "",
        "soc": "",
        "ram_gb": None,
        "android": "",
    },
    "runtime": "litert-lm",
    "model_file": "",
    "n_sms": 100,
    "n_chat": 20,
    "ttft_ms_warm_median": None,
    "decode_tok_s": None,
    "peak_rss_mb": None,
    "json_valid_pct": None,
    "notes": "Run the same 100 SMS + 20 chat prompts as ml/eval/fixtures after the model loads once (warm).",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("eval/reports/ondevice_placeholder.json"))
    p.add_argument("--device", default="")
    args = p.parse_args()
    doc = dict(TEMPLATE)
    doc["captured_at"] = datetime.now(timezone.utc).isoformat()
    doc["device"]["marketing_name"] = args.device
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
