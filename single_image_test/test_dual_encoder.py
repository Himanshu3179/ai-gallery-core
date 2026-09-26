import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from PIL import Image

# Suppress noisy warnings
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
from transformers import CLIPProcessor, CLIPModel

CURRENT_DIR = Path(__file__).resolve().parent
DUAL_ENCODER_OUTPUT = CURRENT_DIR / "dual_encoder_output.json"
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


def load_clip(model_id: str, device: str):
    """Loads CLIP model and processor onto device."""
    t0 = time.perf_counter()
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    load_time = round(time.perf_counter() - t0, 3)
    return processor, model, load_time


def benchmark_image(
    image_path: str,
    test_queries: list = None,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = None
) -> dict:
    if device is None:
        device = Config.DEVICE

    image_path = os.path.abspath(os.path.expanduser(image_path))
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    processor, model, load_time = load_clip(model_id, device)

    # 1. Read Image
    t0 = time.perf_counter()
    with Image.open(image_path) as raw_img:
        img_rgb = raw_img.convert("RGB")
        width, height = raw_img.size
        stat = os.stat(image_path)
    image_prep_time = round(time.perf_counter() - t0, 4)

    # 2. Direct Vector Encoding (ViT forward pass)
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    
    # Warmup pass
    with torch.inference_mode():
        _ = model.get_image_features(**inputs)

    # Precision latency test (average of 10 runs)
    times = []
    with torch.inference_mode():
        for _ in range(10):
            t_start = time.perf_counter()
            image_features = model.get_image_features(**inputs)
            times.append(time.perf_counter() - t_start)

    avg_inference_seconds = sum(times) / len(times)
    avg_inference_ms = round(avg_inference_seconds * 1000, 2)

    # Normalize image embedding
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    embedding_vector = image_features.squeeze(0).cpu().tolist()

    # 3. Zero-shot query matching demonstration
    if not test_queries:
        test_queries = [
            "person smiling in a gym",
            "a woman taking a selfie in mirror",
            "a man singing on stage with microphone",
            "a computer screen with video call",
            "a cat sleeping on a sofa",
            "food on a dining table",
            "a car parked outside"
        ]

    t0 = time.perf_counter()
    text_inputs = processor(text=test_queries, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0).cpu().tolist()
    text_search_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    ranked_queries = [
        {"query": q, "similarity_score": round(s, 4)}
        for q, s in sorted(zip(test_queries, similarities), key=lambda x: x[1], reverse=True)
    ]

    result = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "device": device,
        "embedding_dimensions": len(embedding_vector),
        "performance": {
            "model_load_seconds": load_time,
            "image_prep_seconds": image_prep_time,
            "image_encoding_ms": avg_inference_ms,
            "image_encoding_seconds": round(avg_inference_seconds, 5),
            "text_search_latency_ms": text_search_latency_ms,
            "projected_throughput_images_per_second": round(1.0 / avg_inference_seconds, 1),
            "projected_10k_images_time_seconds": round(10000 * avg_inference_seconds, 1),
            "projected_10k_images_time_minutes": round((10000 * avg_inference_seconds) / 60, 2)
        },
        "image_metadata": {
            "file_name": os.path.basename(image_path),
            "file_path": image_path,
            "dimensions": f"{width}x{height}",
            "size_bytes": stat.st_size
        },
        "zero_shot_search_demo": ranked_queries,
        "embedding_preview": embedding_vector[:8]
    }

    return result


def append_to_output(entry: dict, filepath: Path = DUAL_ENCODER_OUTPUT):
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
        except Exception:
            existing = []

    combined = [entry] + existing
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved result to {filepath.name} (Total entries: {len(combined)})")


def main():
    parser = argparse.ArgumentParser(
        description="Direct Dual-Encoder (CLIP/SigLIP) Vector Benchmark for Ultra-Fast Search"
    )
    parser.add_argument("image_path", nargs="?", help="Path to the image to encode")
    parser.add_argument("--queries", type=str, default=None, help="Comma-separated test search queries")

    args = parser.parse_args()

    image_path = args.image_path
    if not image_path:
        image_path = input("Enter image path: ").strip()

    image_path = image_path.strip("'\"")
    if not image_path or not os.path.isfile(image_path):
        print(f"❌ Error: Valid image not found at '{image_path}'")
        sys.exit(1)

    queries = [q.strip() for q in args.queries.split(",")] if args.queries else None

    print(f"\n⚡ Benchmarking Direct Dual-Encoder...")
    res = benchmark_image(image_path, test_queries=queries)

    perf = res["performance"]
    print("\n" + "═" * 60)
    print("🚀 DIRECT DUAL-ENCODER BENCHMARK RESULTS")
    print("═" * 60)
    print(f"📷 Image               : {res['image_metadata']['file_name']} ({res['image_metadata']['dimensions']})")
    print(f"🧠 Model               : {res['model_id']} ({res['embedding_dimensions']}d vector)")
    print(f"💻 Device              : {res['device'].upper()}")
    print("─" * 60)
    print(f"⏱️  Image Encoding Time : {perf['image_encoding_ms']} ms ({perf['image_encoding_seconds']:.4f}s)")
    print(f"⚡ Throughput          : {perf['projected_throughput_images_per_second']} images/sec (single thread)")
    print(f"🏁 10,000 Images Time  : {perf['projected_10k_images_time_seconds']}s (~{perf['projected_10k_images_time_minutes']} minutes!)")
    print("─" * 60)
    print("🎯 Zero-Shot Semantic Search Match Scores:")
    for item in res["zero_shot_search_demo"]:
        bar = "█" * int(item["similarity_score"] * 40)
        print(f"  [{item['similarity_score']:.3f}] {item['query']:<42} {bar}")
    print("═" * 60 + "\n")

    append_to_output(res)


if __name__ == "__main__":
    main()
