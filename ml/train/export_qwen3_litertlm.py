"""Convert FinnAI merged Qwen3-1.7B HF checkpoint → LiteRT-LM INT4 nothink .litertlm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--quantize", default="dynamic_int4_block32")
    p.add_argument("--kv-cache-max-len", type=int, default=1280)
    p.add_argument("--name-prefix", default="qwen3_1.7b_finndot_nothink")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(args.checkpoint.resolve())
    out = str(args.out_dir.resolve())

    flags = {
        "checkpoint": ckpt,
        "quantize": args.quantize,
        "kv_cache_max_len": args.kv_cache_max_len,
        "name_prefix": args.name_prefix,
        "nothink_prefix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "tool": "litert_torch.generative convert_to_litert",
    }
    (args.out_dir / "export_flags.json").write_text(json.dumps(flags, indent=2) + "\n")

    # Prefer Generative API (explicit nothink + litertlm pack)
    try:
        from litert_torch.generative.examples.qwen import qwen3
        from litert_torch.generative.layers import kv_cache as kv_utils
        from litert_torch.generative.utilities import converter
        from litert_torch.generative.utilities import export_config as ec_lib
    except ImportError:
        # Older package name
        try:
            from ai_edge_torch.generative.examples.qwen import qwen3
            from ai_edge_torch.generative.layers import kv_cache as kv_utils
            from ai_edge_torch.generative.utilities import converter
            from ai_edge_torch.generative.utilities import export_config as ec_lib
        except ImportError as e:
            print("litert generative API missing:", e, file=sys.stderr)
            sys.exit(2)

    ec = ec_lib.ExportConfig()
    ec.kvcache_layout = kv_utils.KV_LAYOUT_TRANSPOSED
    if hasattr(ec, "mask_as_input"):
        ec.mask_as_input = True

    model = qwen3.build_1_7b_model(ckpt)
    converter.convert_to_litert(
        model,
        output_path=out,
        output_name_prefix=args.name_prefix,
        prefill_seq_len=[8, 64, 128, 256, 512, 1024],
        kv_cache_max_len=args.kv_cache_max_len,
        quantize=args.quantize,
        export_config=ec,
        output_format="litertlm",
        hf_tokenizer_model_path=str(Path(ckpt) / "tokenizer.json"),
        stop_token_ids=[151645, 151643],
        user_prompt_prefix="<|im_start|>user\n",
        user_prompt_suffix="<|im_end|>\n",
        model_prompt_prefix="<|im_start|>assistant\n<think>\n\n</think>\n\n",
        model_prompt_suffix="<|im_end|>\n",
    )
    print("export ok →", out)


if __name__ == "__main__":
    main()
