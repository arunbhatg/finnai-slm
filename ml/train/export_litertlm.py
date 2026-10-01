"""Export merged HF checkpoint to LiteRT-LM INT4 nothink .litertlm."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


NOTHINK_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--quantize", default="dynamic_int4_block32")
    p.add_argument("--kv-cache-max-len", type=int, default=1280)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    flags = {
        "checkpoint": str(args.checkpoint.resolve()),
        "quantize": args.quantize,
        "kv_cache_max_len": args.kv_cache_max_len,
        "nothink_prefix": NOTHINK_PREFIX,
        "tool": "litert-torch export_hf",
    }
    (args.out_dir / "export_flags.json").write_text(json.dumps(flags, indent=2) + "\n")

    cmd = [
        sys.executable,
        "-m",
        "litert_torch",
        "export_hf",
        "--model",
        str(args.checkpoint),
        "--output_dir",
        str(args.out_dir),
        "--quantize",
        args.quantize,
    ]
    print("Running:", " ".join(cmd))
    print("If litert-torch is missing, install it on Linux (SageMaker / g5.2xlarge) per")
    print("https://developers.google.com/edge/litert/conversion/pytorch/genai")
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        py = args.out_dir / "export_hf_fallback.py"
        py.write_text(
            """
# Fallback documented in Google AI Edge Qwen3 conversion.
# Run on a Linux GPU box with litert-torch installed:
#
# from litert_torch.generative.examples.qwen import qwen3
# from litert_torch.generative.utilities import converter, export_config as ec_lib
# from litert_torch.generative.layers import kv_cache as kv_utils
# import json
#
# CKPT = r"{ckpt}"
# ec = ec_lib.ExportConfig()
# ec.kvcache_layout = kv_utils.KV_LAYOUT_TRANSPOSED
# ec.mask_as_input = True
# converter.convert_to_litert(
#     qwen3.build_1_7b_model(CKPT),
#     output_path=r"{out}",
#     output_name_prefix="qwen3_1.7b_finndot_nothink",
#     prefill_seq_len=[8, 64, 128, 256, 512, 1024],
#     kv_cache_max_len={kv},
#     quantize="{q}",
#     export_config=ec,
#     output_format="litertlm",
#     hf_tokenizer_model_path=CKPT + "/tokenizer.json",
#     stop_token_ids=[151645, 151643],
#     user_prompt_prefix="<|im_start|>user\\n",
#     user_prompt_suffix="<|im_end|>\\n",
#     model_prompt_prefix="<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n",
#     model_prompt_suffix="<|im_end|>\\n",
# )
""".format(
                ckpt=args.checkpoint.resolve(),
                out=args.out_dir.resolve(),
                kv=args.kv_cache_max_len,
                q=args.quantize,
            ),
            encoding="utf-8",
        )
        print("Wrote", py, "because export_hf failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
