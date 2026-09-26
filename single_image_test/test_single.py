import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from PIL import Image, ExifTags

# 1. Clean up noisy logs and warnings completely
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = CURRENT_DIR / "output.json"


def format_size(size_bytes: int) -> str:
    """Formats file size in human-readable units."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def extract_metadata(image_path: str, img: Image.Image) -> dict:
    """Extracts file, image, and EXIF metadata."""
    stat = os.stat(image_path)
    width, height = img.size
    aspect_ratio = round(width / height, 2) if height > 0 else None

    # Extract EXIF tags
    exif_details = {}
    try:
        raw_exif = img._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if tag_name in ("MakerNote", "UserComment"):
                    continue
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="ignore").strip("\x00")
                    except Exception:
                        continue
                elif hasattr(value, "numerator") and hasattr(value, "denominator"):
                    value = float(value)
                elif isinstance(value, tuple):
                    value = list(value)
                elif not isinstance(value, (int, float, str, list, dict, bool)):
                    value = str(value)
                exif_details[tag_name] = value
    except Exception:
        pass

    return {
        "file_name": os.path.basename(image_path),
        "file_path": os.path.abspath(image_path),
        "file_size_bytes": stat.st_size,
        "file_size_human": format_size(stat.st_size),
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "dimensions": {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio
        },
        "format": img.format or os.path.splitext(image_path)[1].lstrip(".").upper(),
        "mode": img.mode,
        "date_taken": exif_details.get("DateTimeOriginal") or exif_details.get("DateTime"),
        "camera_make": exif_details.get("Make"),
        "camera_model": exif_details.get("Model"),
        "exif_summary": {k: exif_details[k] for k in list(exif_details.keys())[:10]} if exif_details else {}
    }


def load_moondream(device: str):
    """Loads Moondream2 tokenizer and model onto GPU."""
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MOONDREAM_ID,
        revision=Config.MOONDREAM_REV
    )
    model = AutoModelForCausalLM.from_pretrained(
        Config.MOONDREAM_ID,
        trust_remote_code=True,
        revision=Config.MOONDREAM_REV,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    return tokenizer, model


def describe_image(
    image_path: str,
    prompt: str = None,
    max_new_tokens: int = 60,
    model=None,
    tokenizer=None,
    device=None,
    verbose: bool = True
) -> dict:
    """Generates description and records granular timing for each phase."""
    overall_start = time.perf_counter()
    image_path = os.path.abspath(os.path.expanduser(image_path))
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    if prompt is None:
        prompt = "Briefly describe this image in 1 concise sentences."

    if device is None:
        device = Config.DEVICE

    # Phase 1: Model Loading (Disk -> RAM -> GPU)
    t0 = time.perf_counter()
    if model is None or tokenizer is None:
        tokenizer, model = load_moondream(device)
    t_model_load = round(time.perf_counter() - t0, 3)

    # Phase 2: Image Reading, EXIF & Preprocessing
    t0 = time.perf_counter()
    with Image.open(image_path) as raw_img:
        metadata = extract_metadata(image_path, raw_img)
        img_rgb = raw_img.convert("RGB")
        img_rgb.thumbnail((768, 768))
    t_image_prep = round(time.perf_counter() - t0, 3)

    # Phase 3: Vision Feature Encoding (GPU)
    t0 = time.perf_counter()
    with torch.inference_mode():
        enc_image = model.encode_image(img_rgb)
    t_vision_encode = round(time.perf_counter() - t0, 3)

    # Phase 4: Autoregressive Text Generation (GPU)
    t0 = time.perf_counter()
    with torch.inference_mode():
        description = model.answer_question(
            enc_image,
            prompt,
            tokenizer,
            max_new_tokens=max_new_tokens
        )
    t_text_gen = round(time.perf_counter() - t0, 3)

    total_time = round(time.perf_counter() - overall_start, 3)
    pure_inference_time = round(t_vision_encode + t_text_gen, 3)
    timestamp_now = datetime.now().isoformat()

    timing_breakdown = {
        "model_load_gpu_seconds": t_model_load,
        "image_prep_seconds": t_image_prep,
        "vision_encode_seconds": t_vision_encode,
        "text_generation_seconds": t_text_gen,
        "pure_gpu_inference_seconds": pure_inference_time,
        "total_pipeline_seconds": total_time
    }

    result = {
        "generated_at": timestamp_now,
        "generation_time_seconds": pure_inference_time,
        "generation_time_human": f"{pure_inference_time:.3f}s",
        "total_time_seconds": total_time,
        "total_time_human": f"{total_time:.3f}s",
        "timing_breakdown": timing_breakdown,
        "description": description.strip() if description else "",
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "model_info": {
            "model_id": Config.MOONDREAM_ID,
            "revision": Config.MOONDREAM_REV,
            "device": device,
            "dtype": "torch.float16"
        },
        "metadata": metadata
    }

    if verbose:
        print("\n" + "─" * 46)
        print("⏱️  Execution Time Breakdown:")
        print(f" • [1/4] Model load into GPU ({device}) : {t_model_load:.2f}s")
        print(f" • [2/4] Image read & EXIF metadata : {t_image_prep:.2f}s")
        print(f" • [3/4] Vision encoder (GPU)       : {t_vision_encode:.2f}s")
        print(f" • [4/4] Text generation (GPU)      : {t_text_gen:.2f}s")
        print("─" * 46)
        print(f"⚡ Total Pipeline Latency          : {total_time:.2f}s")
        print("─" * 46)
        print(f"\n📝 Description:\n\"{result['description']}\"\n")

    return result


def append_to_output_json(new_entry: dict, output_file: Path = OUTPUT_FILE):
    """
    Appends the new result to the top of output.json (newest entries first).
    """
    existing_entries = []
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    loaded = json.loads(content)
                    if isinstance(loaded, list):
                        existing_entries = loaded
                    elif isinstance(loaded, dict):
                        existing_entries = [loaded]
        except Exception as e:
            print(f"Warning: Could not read existing output.json ({e}). Creating new list.")
            existing_entries = []

    # Prepend new entry to the top
    combined = [new_entry] + existing_entries

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved to {output_file.name} (Total entries: {len(combined)})")


def main():
    parser = argparse.ArgumentParser(
        description="Process a single image with Moondream2, show clean logs & save to output.json"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to the image to describe (e.g. /path/to/photo.jpg)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for Moondream2 (default: concise 1-sentence description)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Maximum tokens to generate (default: 60)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON to terminal in addition to the concise summary"
    )

    args = parser.parse_args()

    image_path = args.image_path
    if not image_path:
        image_path = input("Enter image path: ").strip()

    # Strip quotes if copied from terminal
    image_path = image_path.strip("'\"")

    if not image_path:
        print("❌ Error: No image path provided.")
        sys.exit(1)

    try:
        result = describe_image(
            image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            verbose=True
        )

        if args.json:
            print("\n=== Full JSON ===")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("=================\n")

        # Prepend to output.json
        append_to_output_json(result)

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
