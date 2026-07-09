#!/usr/bin/env python3
"""Download the NeMo cache-aware streaming FastConformer Hybrid RNNT/CTC BPE checkpoint.

Model: nvidia/stt_en_fastconformer_hybrid_large_streaming_multi
  - Cache-aware streaming FastConformer
  - Hybrid architecture: RNN-T (Transducer) + CTC decoders sharing one encoder
  - BPE (SentencePiece) tokenizer
  - Trained with multiple look-ahead sizes (0ms / 80ms / 480ms / 1040ms)

The script offers two download paths:
  1. Via NeMo (loads/validates the model and caches the .nemo file).
  2. Via huggingface_hub (just fetches the raw .nemo file, no NeMo import needed).
"""

import argparse
import os
import shutil
import sys

MODEL_NAME = "stt_en_fastconformer_hybrid_large_streaming_multi"
HF_REPO_ID = f"nvidia/{MODEL_NAME}"
HF_FILENAME = f"{MODEL_NAME}.nemo"


def download_with_nemo(output_dir: str) -> str:
    """Download + instantiate the model with NeMo, then copy the .nemo into output_dir."""
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    print(f"[nemo] Loading pretrained model '{MODEL_NAME}' ...")
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=MODEL_NAME)

    # NeMo caches the downloaded artifact; surface it and copy to output_dir.
    src = getattr(model, "_cfg_path", None) or _find_cached_nemo()
    os.makedirs(output_dir, exist_ok=True)
    dst = os.path.join(output_dir, HF_FILENAME)

    if src and os.path.isfile(src):
        shutil.copyfile(src, dst)
        print(f"[nemo] Copied cached checkpoint to: {dst}")
    else:
        # Fall back to re-saving the loaded model.
        model.save_to(dst)
        print(f"[nemo] Saved model to: {dst}")
    return dst


def _find_cached_nemo() -> str | None:
    """Locate the .nemo file inside NeMo's default cache directory."""
    cache_root = os.path.expanduser("~/.cache/torch/NeMo")
    for root, _dirs, files in os.walk(cache_root):
        for f in files:
            if f == HF_FILENAME:
                return os.path.join(root, f)
    return None


def download_with_hf(output_dir: str) -> str:
    """Download just the raw .nemo file from the Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    os.makedirs(output_dir, exist_ok=True)
    print(f"[hf] Downloading '{HF_FILENAME}' from '{HF_REPO_ID}' ...")
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=output_dir,
    )
    print(f"[hf] Downloaded checkpoint to: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--method",
        choices=["nemo", "hf"],
        default="hf",
        help="Download via NeMo (validates the model) or huggingface_hub (raw file). Default: hf",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints",
        help="Directory to save the .nemo checkpoint. Default: ./checkpoints",
    )
    args = parser.parse_args()

    try:
        if args.method == "nemo":
            path = download_with_nemo(args.output_dir)
        else:
            path = download_with_hf(args.output_dir)
    except ImportError as e:
        pkg = "nemo_toolkit['asr']" if args.method == "nemo" else "huggingface_hub"
        print(f"ERROR: missing dependency ({e}). Install it with:\n"
              f"    pip install {pkg}", file=sys.stderr)
        return 1

    print(f"\nDone. Checkpoint: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
