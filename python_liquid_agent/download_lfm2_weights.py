"""
Downloads the official pretrained LiquidAI/LFM2.5-350M checkpoint (config.json, tokenizer.json,
chat_template.jinja, and model.safetensors) from HuggingFace into ./lfm25-350m (or LFM2_MODEL_DIR).
"""

import os
import subprocess
import sys

DEFAULT_DIR = os.environ.get(
    "LFM2_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm25-350m"),
)
HF_BASE = "https://huggingface.co/LiquidAI/LFM2.5-350M/resolve/main"
FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "model.safetensors",
]


def download_all(target_dir: str = DEFAULT_DIR):
    os.makedirs(target_dir, exist_ok=True)
    for fname in FILES:
        dest = os.path.join(target_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[OK] {fname} already present ({os.path.getsize(dest):,} bytes)")
            continue
        url = f"{HF_BASE}/{fname}"
        print(f"[DOWNLOADING] {url} -> {dest}")
        subprocess.run(
            ["curl", "-L", "--retry", "5", "-C", "-", url, "-o", dest],
            check=True,
        )
    print(f"\nAll LiquidAI/LFM2.5-350M checkpoint files ready in: {target_dir}")


if __name__ == "__main__":
    download_all()
