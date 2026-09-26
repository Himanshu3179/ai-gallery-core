import os
import sys
import gc
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from PIL import Image, ExifTags

# Suppress noisy logs
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    BlipProcessor,
    BlipForConditionalGeneration
)

CURRENT_DIR = Path(__file__).resolve().parent
COMPARISON_FILE = CURRENT_DIR / "comparison_output.json"


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def get_image_info(image_path: str) -> tuple:
    stat = os.stat(image_path)
    with Image.open(image_path) as img:
        width, height = img.size
        fmt = img.format or "UNKNOWN"
        mode = img.mode
        rgb_img = img.convert("RGB")
    meta = {
        "file_name": os.path.basename(image_path),
        "file_path": os.path.abspath(image_path),
        "file_size_human": format_size(stat.st_size),
        "dimensions": f"{width}x{height}",
        "format": fmt,
        "mode": mode
    }
    return rgb_img, meta


def cleanup_memory(device: str):
    gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device == "cuda" and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()


# -------------------------------------------------------------
# Model Runners
# -------------------------------------------------------------

def run_moondream(img: Image.Image, device: str) -> dict:
    """Runs vikhyatk/moondream2 (~1.86B)"""
    model_id = Config.MOONDREAM_ID
    revision = Config.MOONDREAM_REV
    
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=revision,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    t_load = round(time.perf_counter() - t0, 3)

    img_thumb = img.copy()
    img_thumb.thumbnail((768, 768))

    t0 = time.perf_counter()
    with torch.inference_mode():
        enc = model.encode_image(img_thumb)
        desc = model.answer_question(
            enc,
            "Briefly describe this image in 1 concise sentences.",
            tokenizer,
            max_new_tokens=60
        )
    t_infer = round(time.perf_counter() - t0, 3)

    del model, tokenizer
    cleanup_memory(device)

    return {
        "model_id": model_id,
        "name": "Moondream2",
        "parameters": "1.86B",
        "model_load_seconds": t_load,
        "inference_seconds": t_infer,
        "total_seconds": round(t_load + t_infer, 3),
        "description": desc.strip() if desc else ""
    }


def run_smolvlm(img: Image.Image, device: str) -> dict:
    """Runs HuggingFaceTB/SmolVLM-500M-Instruct (~500M)"""
    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        _attn_implementation="eager"
    ).to(device)
    model.eval()
    t_load = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    with torch.inference_mode():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in 1 concise sentence."}
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        desc = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    t_infer = round(time.perf_counter() - t0, 3)

    del model, processor
    cleanup_memory(device)

    return {
        "model_id": model_id,
        "name": "SmolVLM-500M",
        "parameters": "500M",
        "model_load_seconds": t_load,
        "inference_seconds": t_infer,
        "total_seconds": round(t_load + t_infer, 3),
        "description": desc.strip() if desc else ""
    }


def run_blip(img: Image.Image, device: str) -> dict:
    """Runs Salesforce/blip-image-captioning-base (~220M)"""
    model_id = "Salesforce/blip-image-captioning-base"

    t0 = time.perf_counter()
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    t_load = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    with torch.inference_mode():
        inputs = processor(img, return_tensors="pt").to(device, torch.float16)
        out = model.generate(**inputs, max_new_tokens=40)
        desc = processor.decode(out[0], skip_special_tokens=True)
    t_infer = round(time.perf_counter() - t0, 3)

    del model, processor
    cleanup_memory(device)

    return {
        "model_id": model_id,
        "name": "BLIP-Base",
        "parameters": "220M",
        "model_load_seconds": t_load,
        "inference_seconds": t_infer,
        "total_seconds": round(t_load + t_infer, 3),
        "description": desc.strip() if desc else ""
    }


def run_git(img: Image.Image, device: str) -> dict:
    """Runs microsoft/git-base (~176M)"""
    model_id = "microsoft/git-base"

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    t_load = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    with torch.inference_mode():
        inputs = processor(images=img, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values.to(torch.float16)
        generated_ids = model.generate(pixel_values=pixel_values, max_new_tokens=40)
        desc = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    t_infer = round(time.perf_counter() - t0, 3)

    del model, processor
    cleanup_memory(device)

    return {
        "model_id": model_id,
        "name": "GIT-Base",
        "parameters": "176M",
        "model_load_seconds": t_load,
        "inference_seconds": t_infer,
        "total_seconds": round(t_load + t_infer, 3),
        "description": desc.strip() if desc else ""
    }


MODEL_REGISTRY = {
    "moondream": ("Moondream2 (1.86B)", run_moondream),
    "smolvlm":   ("SmolVLM (500M)", run_smolvlm),
    "blip":      ("BLIP-Base (220M)", run_blip),
    "git":       ("GIT-Base (176M)", run_git)
}


def append_to_comparison_json(entry: dict, filepath: Path = COMPARISON_FILE):
    existing = []
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    loaded = json.loads(content)
                    if isinstance(loaded, list):
                        existing = loaded
                    elif isinstance(loaded, dict):
                        existing = [loaded]
        except Exception as e:
            print(f"Warning: Could not read existing comparison file ({e}). Starting fresh.")
            existing = []

    combined = [entry] + existing
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved comparison report to {filepath.name} (Total runs: {len(combined)})")


def main():
    parser = argparse.ArgumentParser(
        description="Compare 4 lightweight Vision-Language Models on a single image"
    )
    parser.add_argument("image_path", nargs="?", help="Path to image file")
    parser.add_argument(
        "--models",
        type=str,
        default="blip,git,smolvlm,moondream",
        help="Comma-separated models to test (options: blip, git, smolvlm, moondream)"
    )

    args = parser.parse_args()

    image_path = args.image_path
    if not image_path:
        image_path = input("Enter image path: ").strip()

    image_path = image_path.strip("'\"")
    if not image_path or not os.path.isfile(image_path):
        print(f"❌ Error: Valid image file not found at '{image_path}'")
        sys.exit(1)

    device = Config.DEVICE
    print(f"\n🚀 Running VLM Comparison on {device.upper()}...")
    print(f"📷 Image: {image_path}")

    rgb_img, img_meta = get_image_info(image_path)
    selected_keys = [k.strip().lower() for k in args.models.split(",") if k.strip().lower() in MODEL_REGISTRY]

    results = []
    for idx, key in enumerate(selected_keys, 1):
        label, runner = MODEL_REGISTRY[key]
        print(f"\n[{idx}/{len(selected_keys)}] Evaluating {label}...")
        try:
            res = runner(rgb_img, device)
            print(f"   ⏱️  Load: {res['model_load_seconds']:.2f}s | Infer: {res['inference_seconds']:.2f}s | Total: {res['total_seconds']:.2f}s")
            print(f"   💬 \"{res['description']}\"")
            results.append(res)
        except Exception as e:
            print(f"   ❌ Error running {label}: {e}")
            results.append({
                "model_id": key,
                "name": label,
                "error": str(e)
            })

    # Terminal Comparison Summary Table
    print("\n" + "=" * 90)
    print("📊 VLM BENCHMARK & QUALITY COMPARISON")
    print(f"Image: {img_meta['file_name']} ({img_meta['dimensions']}, {img_meta['file_size_human']})")
    print("=" * 90)
    print(f"{'Model':<14} | {'Size':<6} | {'Load (s)':<9} | {'Infer (s)':<10} | {'Total (s)':<10} | {'Description'}")
    print("-" * 14 + "-+-" + "-" * 6 + "-+-" + "-" * 9 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 30)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<14} | ERROR  | {r['error']}")
        else:
            short_desc = (r['description'][:50] + "...") if len(r['description']) > 50 else r['description']
            print(f"{r['name']:<14} | {r['parameters']:<6} | {r['model_load_seconds']:<9.2f} | {r['inference_seconds']:<10.2f} | {r['total_seconds']:<10.2f} | {short_desc}")

    print("=" * 90)

    # Save to comparison JSON
    report_entry = {
        "timestamp": datetime.now().isoformat(),
        "image_metadata": img_meta,
        "device": device,
        "models": results
    }
    append_to_comparison_json(report_entry)


if __name__ == "__main__":
    main()
