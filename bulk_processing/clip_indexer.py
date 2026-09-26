import os
import sys
import json
import time
import shutil
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from PIL import Image, ExifTags

# Clean up noisy logs and warnings completely
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from transformers import CLIPProcessor, CLIPModel

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = "/Users/apple/Documents/elements/himanshu/oneplus/Camera"
TAGS_FILE = CURRENT_DIR / "tags.json"
OUTPUT_FILE = CURRENT_DIR / "clip_gallery_index_camera.json"
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"

# Strict calibrated threshold (prevents false positive hallucinations)
MIN_CALIBRATED_THRESHOLD = 0.235


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def clean_exif_val(val):
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="ignore").strip("\x00")
        except Exception:
            return None
    elif isinstance(val, (int, float, str, bool)):
        return val
    elif isinstance(val, (list, tuple)):
        return [clean_exif_val(x) for x in val]
    elif isinstance(val, dict):
        return {str(k): clean_exif_val(v) for k, v in val.items()}
    return str(val)


def extract_metadata(image_path: Path, img: Image.Image) -> dict:
    stat = os.stat(image_path)
    width, height = img.size
    aspect_ratio = round(width / height, 2) if height > 0 else None

    exif_details = {}
    try:
        raw_exif = img._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if tag_name in ("MakerNote", "UserComment"):
                    continue
                cleaned = clean_exif_val(value)
                if cleaned is not None:
                    exif_details[tag_name] = cleaned
    except Exception:
        pass

    return {
        "file_name": image_path.name,
        "file_path": str(image_path.resolve()),
        "file_size_bytes": stat.st_size,
        "file_size_human": format_size(stat.st_size),
        "dimensions": {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio
        },
        "date_taken": exif_details.get("DateTimeOriginal") or exif_details.get("DateTime"),
        "camera_make": exif_details.get("Make"),
        "camera_model": exif_details.get("Model"),
        "format": image_path.suffix.lstrip(".").upper()
    }


def load_tags(tags_path: Path = TAGS_FILE) -> list:
    with open(tags_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_tags = []
    for category, tag_list in data.get("categories", {}).items():
        all_tags.extend(tag_list)
    return sorted(list(set(all_tags)))


def calibrate_confidence_pct(score: float, min_val: float = 0.20, max_val: float = 0.32) -> float:
    scaled = (score - min_val) / (max_val - min_val)
    clamped = max(0.05, min(0.99, scaled))
    return round(clamped * 100, 1)


def atomic_save(output_path: Path, payload: dict):
    """Saves JSON atomically via temp file to guarantee no file corruption if interrupted."""
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, output_path)


def render_horizontal_bar(
    current: int,
    total: int,
    rate: float,
    filename: str,
    tag_str: str,
    bar_width: int = 22
) -> str:
    """Renders a sleek in-place ANSI horizontal progress bar with real-time throughput.
    
    Guarantees no multi-line wrapping so the terminal stays on a single cleanly updating line.
    """
    pct = (current / total) if total > 0 else 0.0
    pct = min(max(pct, 0.0), 1.0)
    filled_len = int(bar_width * pct)
    empty_len = bar_width - filled_len

    # High-contrast dual-tone bar: bright cyan progress, subtle dark gray empty
    bar_styled = f"\033[1;36m{'█' * filled_len}\033[0m\033[2;37m{'░' * empty_len}\033[0m"

    cols = shutil.get_terminal_size((100, 24)).columns

    # Dynamic status line header
    # \r returns cursor to column 0, \033[K erases the entire current line
    header = (
        f"\r\033[K⚡ [{bar_styled}] "
        f"\033[1;37m{current:>3}/{total}\033[0m "
        f"(\033[1;32m{pct*100:>5.1f}%\033[0m) "
        f"| \033[1;33m{rate:>4.1f} img/s\033[0m"
    )

    # Calculate remaining printable columns to strictly prevent terminal line wrapping
    visible_header_len = 3 + 2 + bar_width + 1 + 8 + 9 + 3 + 12
    remaining_space = cols - visible_header_len - 6

    if remaining_space < 12:
        return header

    # Truncate filename if needed
    fn_display = filename[:18] + ("…" if len(filename) > 18 else "")
    suffix = f" | 📷 \033[0;36m{fn_display}\033[0m"

    remaining_for_tag = remaining_space - len(fn_display) - 6
    if tag_str and remaining_for_tag > 8:
        tag_display = tag_str[:remaining_for_tag] + ("…" if len(tag_str) > remaining_for_tag else "")
        suffix += f" ➔ \033[0;37m{tag_display}\033[0m"

    return header + suffix


def run_clip_indexing(
    images_dir: Path = DEFAULT_IMAGES_DIR,
    output_file: Path = OUTPUT_FILE,
    batch_size: int = 16,
    threshold: float = MIN_CALIBRATED_THRESHOLD,
    device: str = "auto"
):
    images_dir = Path(images_dir).resolve()

    # Resolve compute device and precision
    req_device = (device or "auto").lower()
    if req_device == "cpu":
        resolved_device = "cpu"
        model_dtype = torch.float32
        device_label = "CPU"
    elif req_device == "mps":
        resolved_device = "mps" if torch.backends.mps.is_available() else "cpu"
        model_dtype = torch.float16 if resolved_device == "mps" else torch.float32
        device_label = "Apple Silicon GPU (MPS)" if resolved_device == "mps" else "CPU (MPS unavailable)"
    elif req_device == "cuda":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_dtype = torch.float16 if resolved_device == "cuda" else torch.float32
        device_label = "NVIDIA GPU (CUDA)" if resolved_device == "cuda" else "CPU (CUDA unavailable)"
    else:  # "auto"
        if torch.backends.mps.is_available():
            resolved_device = "mps"
            model_dtype = torch.float16
            device_label = "Apple Silicon GPU (MPS)"
        elif torch.cuda.is_available():
            resolved_device = "cuda"
            model_dtype = torch.float16
            device_label = "NVIDIA GPU (CUDA)"
        else:
            resolved_device = "cpu"
            model_dtype = torch.float32
            device_label = "CPU"

    print("═" * 75)
    print("⚡ HIGH-SPEED CLIP-ONLY BATCH INDEXER (Zero VLM / Pure Vectors & Tags)")
    print("═" * 75)
    print(f"📁 Image Directory    : {images_dir}")
    print(f"🧠 Dual-Encoder Model : {DEFAULT_MODEL_ID}")
    print(f"💻 Compute Device     : {device_label}")
    print(f"📦 Batch Size         : {batch_size}")
    print(f"🎯 Rejection Threshold: {threshold} (eliminates false positive tags)")
    print(f"💾 Output Destination : {output_file.name} (Fresh rewrite)")
    print("─" * 75)

    # 1. Discover all images
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    image_paths = sorted([
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ])
    total_images = len(image_paths)
    print(f"📸 Total Images Found : {total_images}")

    if total_images == 0:
        print("❌ No images found in directory. Exiting.")
        return

    # 2. Load Model & Processor
    print(f"\n⏳ [1/2] Loading CLIP Model onto {device_label}...")
    t_start_model = time.perf_counter()
    processor = CLIPProcessor.from_pretrained(DEFAULT_MODEL_ID)
    model = CLIPModel.from_pretrained(DEFAULT_MODEL_ID, torch_dtype=model_dtype).to(resolved_device)
    model.eval()
    print(f"   ✅ Model loaded in {time.perf_counter() - t_start_model:.2f}s")

    # 3. Pre-compute Taxonomy Vectors (Calculated ONCE)
    tags = load_tags()
    tag_prompts = [f"a photo of {t}" for t in tags]
    text_inputs = processor(text=tag_prompts, return_tensors="pt", padding=True).to(resolved_device)
    with torch.inference_mode():
        tag_features = model.get_text_features(**text_inputs)
        tag_features = tag_features / tag_features.norm(dim=-1, keepdim=True)
    print(f"   ✅ Pre-computed {len(tags)} tag vectors on {device_label} (Calculated once)")

    # 4. Process Images in Batches with Progressive Saving
    print(f"\n⏳ [2/2] Processing {total_images} Images in Batches of {batch_size}...")
    indexed_records = []
    t_pipeline_start = time.perf_counter()

    for i in range(0, total_images, batch_size):
        chunk = image_paths[i : i + batch_size]
        batch_images = []
        batch_meta = []

        # Read and prepare batch with live per-image progress feedback
        for idx_p, p in enumerate(chunk):
            current_count = len(indexed_records) + idx_p
            elapsed = time.perf_counter() - t_pipeline_start
            current_rate = current_count / max(elapsed, 0.01) if current_count > 0 else 0.0
            sys.stdout.write(render_horizontal_bar(current_count, total_images, current_rate, p.name, "reading..."))
            sys.stdout.flush()

            try:
                with Image.open(p) as img:
                    meta = extract_metadata(p, img)
                    rgb = img.convert("RGB")
                    rgb.thumbnail((768, 768))
                    batch_images.append(rgb)
                    batch_meta.append(meta)
            except Exception:
                pass

        if not batch_images:
            continue

        # Vector Forward Pass
        inputs = processor(images=batch_images, return_tensors="pt").to(resolved_device)
        with torch.inference_mode():
            img_feats = model.get_image_features(**inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            # Dot-product similarity against all taxonomy tags
            sim_matrix = (img_feats @ tag_features.T).cpu()

        # Build records
        for idx_in_batch, meta in enumerate(batch_meta):
            rec_id = len(indexed_records) + 1
            raw_vec = img_feats[idx_in_batch].cpu().tolist()
            sims = sim_matrix[idx_in_batch].tolist()

            # Filter tags by threshold (Rejection Filter)
            passing = []
            for t_idx, score in enumerate(sims):
                if score >= threshold:
                    passing.append({
                        "tag": tags[t_idx],
                        "confidence_score": round(score, 4),
                        "confidence_pct": calibrate_confidence_pct(score)
                    })

            passing.sort(key=lambda x: x["confidence_score"], reverse=True)
            top_calibrated = passing[:4]

            record = {
                "id": f"img_{rec_id:03d}",
                "file_name": meta["file_name"],
                "file_path": meta["file_path"],
                "file_size_human": meta["file_size_human"],
                "file_size_bytes": meta["file_size_bytes"],
                "vector_embedding": raw_vec,
                "calibrated_tags": top_calibrated,
                "is_unclassified": len(top_calibrated) == 0,
                "exif": {
                    "dimensions": meta["dimensions"],
                    "date_taken": meta["date_taken"],
                    "camera_make": meta["camera_make"],
                    "camera_model": meta["camera_model"],
                    "format": meta["format"]
                }
            }
            indexed_records.append(record)

        # Progressive Atomic Save: Even if interrupted, all processed images are safely on disk!
        current_payload = {
            "pipeline": "clip-only-tier1",
            "generated_at": datetime.now().isoformat(),
            "total_images": len(indexed_records),
            "rejection_threshold": threshold,
            "taxonomy_tag_count": len(tags),
            "images": indexed_records
        }
        atomic_save(output_file, current_payload)

        # Real-time in-place progress update (updates exact same line, zero newlines)
        elapsed = time.perf_counter() - t_pipeline_start
        rate = len(indexed_records) / max(elapsed, 0.01)
        latest = indexed_records[-1]
        tag_str = (
            ", ".join([f"{t['tag']} ({t['confidence_pct']}%)" for t in latest["calibrated_tags"][:2]])
            if latest["calibrated_tags"] else "(unclassified)"
        )
        sys.stdout.write(render_horizontal_bar(len(indexed_records), total_images, rate, latest["file_name"], tag_str))
        sys.stdout.flush()

    # Move to a new line once indexing is done before printing completion summary
    sys.stdout.write("\n")
    sys.stdout.flush()

    total_time = time.perf_counter() - t_pipeline_start
    throughput = total_images / total_time
    avg_ms = (total_time / total_images) * 1000

    print("\n" + "═" * 75)
    print("🎉 CLIP-ONLY INDEXING COMPLETE!")
    print("═" * 75)
    print(f"📸 Total Images Processed: {len(indexed_records)}")
    print(f"⏱️  Total Processing Time : {total_time:.2f} seconds")
    print(f"⚡ Throughput            : {throughput:.1f} images / second")
    print(f"🎯 Average Time Per Photo: {avg_ms:.2f} ms")
    print(f"🏁 Projected 10k Photos  : {(10000 / throughput / 60):.2f} minutes!")
    print(f"💾 Clean JSON Output Saved: {output_file.name}")
    print("═" * 75)


def main():
    parser = argparse.ArgumentParser(description="Pure CLIP Bulk Indexer (No Moondream, Vectors + Calibrated Tags + EXIF)")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_IMAGES_DIR), help="Path to images directory")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for GPU/CPU (default: 16)")
    parser.add_argument("--threshold", type=float, default=MIN_CALIBRATED_THRESHOLD, help="Tag threshold (default: 0.235)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to use: 'mps' (Mac GPU), 'cpu' (Processor), 'cuda' (NVIDIA GPU), or 'auto' (default: auto)"
    )
    parser.add_argument("--cpu", action="store_true", help="Shortcut flag to force CPU execution")
    args = parser.parse_args()

    chosen_device = "cpu" if args.cpu else args.device

    run_clip_indexing(
        images_dir=Path(args.dir),
        output_file=Path(args.output),
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=chosen_device
    )


if __name__ == "__main__":
    main()
