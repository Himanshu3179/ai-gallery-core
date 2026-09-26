import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from PIL import Image

# Suppress noisy logs
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
TAGS_FILE = CURRENT_DIR / "tags.json"
RESULTS_FILE = CURRENT_DIR / "bulk_results.json"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "images"
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


def load_tags(tags_path: Path = TAGS_FILE) -> list:
    """Loads all tags from tags.json into a flat list."""
    with open(tags_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_tags = []
    for category, tag_list in data.get("categories", {}).items():
        all_tags.extend(tag_list)
    return sorted(list(set(all_tags)))


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def run_bulk_indexing(
    images_dir: Path = DEFAULT_IMAGES_DIR,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 16,
    top_k_tags: int = 5,
    min_confidence: float = 0.20
):
    device = Config.DEVICE
    images_dir = Path(images_dir).resolve()
    
    print("═" * 70)
    print("🚀 HIGH-THROUGHPUT BULK IMAGE INDEXER & PROBABILISTIC TAGGER")
    print("═" * 70)
    print(f"📁 Image Directory    : {images_dir}")
    print(f"🧠 Dual-Encoder Model : {model_id}")
    print(f"💻 Compute Device     : {device.upper()}")
    print(f"📦 Batch Size         : {batch_size}")
    
    # 1. Discover all images
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    image_paths = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    image_paths.sort()
    total_images = len(image_paths)
    print(f"📸 Total Images Found : {total_images}")

    if total_images == 0:
        print("❌ No images found in directory. Exiting.")
        return

    # 2. Load Model & Processor
    print("\n⏳ [1/3] Loading CLIP Model onto GPU...")
    t_start_model = time.perf_counter()
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    model.eval()
    model_load_time = round(time.perf_counter() - t_start_model, 2)
    print(f"   ✅ Model loaded in {model_load_time}s")

    # 3. Load & Pre-Encode Taxonomy Matrix
    tags = load_tags()
    print(f"\n⏳ [2/3] Pre-computing Embeddings for {len(tags)} Diverse Tags...")
    t_start_tags = time.perf_counter()
    tag_prompts = [f"a photo of {t}" for t in tags]
    text_inputs = processor(text=tag_prompts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        tag_features = model.get_text_features(**text_inputs)
        tag_features = tag_features / tag_features.norm(dim=-1, keepdim=True)
    tags_load_time = round((time.perf_counter() - t_start_tags) * 1000, 2)
    print(f"   ✅ Tag Matrix ({len(tags)} × 512) compiled in {tags_load_time} ms (Calculated ONCE)")

    # 4. Process Images in Batches
    print(f"\n⏳ [3/3] Processing {total_images} Images in Batches of {batch_size}...")
    indexed_records = []
    
    overall_start = time.perf_counter()
    pure_gpu_times = []

    for i in range(0, total_images, batch_size):
        chunk_paths = image_paths[i : i + batch_size]
        batch_images = []
        valid_chunk = []

        # Read batch
        for p in chunk_paths:
            try:
                with Image.open(p) as img:
                    batch_images.append(img.convert("RGB"))
                    valid_chunk.append(p)
            except Exception as e:
                print(f"   ⚠️ Could not read {p.name}: {e}")

        if not batch_images:
            continue

        # GPU Batch Forward Pass
        t_gpu_start = time.perf_counter()
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.inference_mode():
            img_features = model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            # Matrix multiply against ALL pre-compiled tag vectors
            sim_matrix = img_features @ tag_features.T
            prob_matrix = torch.softmax(sim_matrix * 100, dim=-1)

        t_gpu_end = time.perf_counter()
        pure_gpu_times.append(t_gpu_end - t_gpu_start)

        # Extract per-image metadata & top tags
        for idx, p in enumerate(valid_chunk):
            sims = sim_matrix[idx].cpu().tolist()
            probs = prob_matrix[idx].cpu().tolist()

            ranked = sorted(
                zip(tags, sims, probs),
                key=lambda x: x[1],
                reverse=True
            )

            top_tags = [
                {
                    "tag": t,
                    "confidence_score": round(s, 4),
                    "probability_pct": round(p * 100, 1)
                }
                for t, s, p in ranked[:top_k_tags]
                if s >= min_confidence
            ]

            stat = os.stat(p)
            indexed_records.append({
                "file_name": p.name,
                "file_path": str(p.resolve()),
                "file_size_bytes": stat.st_size,
                "file_size_human": format_size(stat.st_size),
                "top_tags": top_tags,
                "primary_tag": top_tags[0]["tag"] if top_tags else "unclassified",
                "primary_confidence": top_tags[0]["confidence_score"] if top_tags else 0.0,
                "embedding": img_features[idx].cpu().tolist()
            })

        processed_so_far = min(i + batch_size, total_images)
        pct = (processed_so_far / total_images) * 100
        print(f"   • Processed {processed_so_far:>3}/{total_images} images ({pct:>5.1f}%) ...", end="\r", flush=True)

    total_pipeline_time = time.perf_counter() - overall_start
    total_pure_gpu = sum(pure_gpu_times)
    avg_per_image_ms = (total_pipeline_time / total_images) * 1000
    throughput = total_images / total_pipeline_time

    print(f"\n   ✅ All {total_images} images processed successfully!\n")

    # 5. Display Benchmark Summary
    print("═" * 70)
    print("📊 BULK INGESTION BENCHMARK RESULTS")
    print("═" * 70)
    print(f"⏱️  Total Processing Time : {total_pipeline_time:.2f} seconds")
    print(f"⚡ Throughput            : {throughput:.1f} images / second")
    print(f"🎯 Average Time Per Photo: {avg_per_image_ms:.2f} ms")
    print(f"🏁 Projected 10k Images  : {(10000 / throughput):.1f}s (~{(10000 / throughput / 60):.2f} minutes!)")
    print("─" * 70)
    print("🏷️  Sample Categorization (First 8 Images):")
    for r in indexed_records[:8]:
        tag_str = ", ".join([f"{t['tag']} ({t['probability_pct']}%)" for t in r["top_tags"][:3]])
        print(f"  • {r['file_name'][:30]:<30} ➔ {tag_str}")
    print("═" * 70)

    # 6. Save Full Results
    summary_report = {
        "generated_at": datetime.now().isoformat(),
        "total_images": total_images,
        "taxonomy_tag_count": len(tags),
        "performance": {
            "total_time_seconds": round(total_pipeline_time, 2),
            "throughput_images_per_second": round(throughput, 1),
            "average_ms_per_image": round(avg_per_image_ms, 2),
            "projected_10k_minutes": round((10000 / throughput / 60), 2)
        },
        "images": indexed_records
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved full indexed data & vector embeddings to: {RESULTS_FILE.name}")


def main():
    parser = argparse.ArgumentParser(description="Bulk Image Indexer & Multi-Label Tagger")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_IMAGES_DIR), help="Path to images directory")
    parser.add_argument("--batch-size", type=int, default=16, help="GPU batch size (default: 16)")
    parser.add_argument("--top-k", type=int, default=5, help="Top K tags to keep per image (default: 5)")
    args = parser.parse_args()

    run_bulk_indexing(
        images_dir=Path(args.dir),
        batch_size=args.batch_size,
        top_k_tags=args.top_k
    )


if __name__ == "__main__":
    main()
