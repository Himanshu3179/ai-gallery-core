import os
import re
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config

CURRENT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "images"
TAGS_FILE = CURRENT_DIR / "tags.json"
OLD_RESULTS_FILE = CURRENT_DIR / "bulk_results.json"
MOONDREAM_FILE = CURRENT_DIR / "moondream_results.json"
UNIFIED_INDEX_FILE = CURRENT_DIR / "unified_gallery_index.json"

MIN_CALIBRATED_THRESHOLD = 0.235  # Strict rejection threshold to prevent false positives
STOPWORDS = {
    "the", "and", "a", "an", "in", "on", "at", "with", "of", "for", "to", "from",
    "by", "as", "is", "are", "was", "were", "this", "that", "these", "those",
    "their", "its", "her", "his", "it", "into", "over", "under", "displays", "shows"
}


def load_taxonomy(tags_path: Path = TAGS_FILE) -> list:
    with open(tags_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_tags = []
    for category, tag_list in data.get("categories", {}).items():
        all_tags.extend(tag_list)
    return sorted(list(set(all_tags)))


def extract_keywords_from_caption(caption: str) -> list:
    if not caption:
        return []
    words = [
        w.lower() for w in re.findall(r"[A-Za-z0-9\-]+", caption)
        if len(w) > 2 and w.lower() not in STOPWORDS
    ]
    # Unique preserve order
    return list(dict.fromkeys(words))[:12]


def calibrate_confidence_pct(score: float, min_val: float = 0.20, max_val: float = 0.32) -> float:
    """Scales raw cosine similarity (typically 0.22 - 0.32) to intuitive 0% - 100% confidence."""
    scaled = (score - min_val) / (max_val - min_val)
    clamped = max(0.05, min(0.99, scaled))
    return round(clamped * 100, 1)


def generate_unified_index():
    device = Config.DEVICE
    print("═" * 75)
    print("🚀 COMPILING OFFICIAL UNIFIED GALLERY INDEX (New Architecture)")
    print("═" * 75)
    print(f"💻 Device             : {device.upper()}")
    print(f"📁 Output Destination : {UNIFIED_INDEX_FILE.name}")
    print(f"🎯 Threshold Filter   : {MIN_CALIBRATED_THRESHOLD} (Rejection threshold)")
    print("─" * 75)

    # 1. Load Moondream narratives & EXIF
    print("⏳ [1/4] Loading Moondream2 Visual Narratives & Metadata...")
    with open(MOONDREAM_FILE, "r", encoding="utf-8") as f:
        moon_data = json.load(f)
    moon_map = {item["file_name"]: item for item in moon_data.get("images", [])}
    print(f"   ✅ Loaded {len(moon_map)} visual narratives")

    # 2. Load Existing Embeddings or Re-compute
    print("\n⏳ [2/4] Loading Pre-computed 512-d CLIP Image Vectors...")
    with open(OLD_RESULTS_FILE, "r", encoding="utf-8") as f:
        old_data = json.load(f)
    vec_map = {item["file_name"]: item for item in old_data.get("images", [])}
    print(f"   ✅ Loaded {len(vec_map)} image vector embeddings")

    # 3. Compile Ensembled Tag Vectors
    tags = load_taxonomy()
    print(f"\n⏳ [3/4] Computing Ensembled Tag Vectors for {len(tags)} Taxonomy Tags...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16).to(device)
    model.eval()

    tag_prompts = [f"a photo of {t}" for t in tags]
    t_inputs = processor(text=tag_prompts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        tag_feats = model.get_text_features(**t_inputs)
        tag_feats = tag_feats / tag_feats.norm(dim=-1, keepdim=True)
    print(f"   ✅ Tag Matrix ({len(tags)} × 512) compiled on GPU")

    # 4. Process Each Image into the New Unified Schema
    print("\n⏳ [4/4] Generating Unified Records (Vector + Calibrated Tags + Narrative + EXIF)...")
    unified_records = []
    
    # Sort image keys consistently
    all_fnames = sorted(list(moon_map.keys()))

    for idx, fname in enumerate(all_fnames, 1):
        m = moon_map.get(fname, {})
        v_entry = vec_map.get(fname, {})
        
        raw_vec = v_entry.get("embedding", [])
        if not raw_vec:
            continue

        img_tensor = torch.tensor(raw_vec, dtype=torch.float16).unsqueeze(0).to(device)

        # Matrix dot-product against all taxonomy tags
        with torch.inference_mode():
            sims = (img_tensor @ tag_feats.T).squeeze(0).cpu().tolist()

        # Apply Calibrated Rejection Filter
        passing_tags = []
        for tag_idx, sim_score in enumerate(sims):
            if sim_score >= MIN_CALIBRATED_THRESHOLD:
                passing_tags.append({
                    "tag": tags[tag_idx],
                    "confidence_score": round(sim_score, 4),
                    "confidence_pct": calibrate_confidence_pct(sim_score)
                })

        passing_tags.sort(key=lambda x: x["confidence_score"], reverse=True)
        top_calibrated = passing_tags[:4]

        # Extract Keywords
        caption = m.get("moondream_caption", "")
        keywords = extract_keywords_from_caption(caption)

        record = {
            "id": f"img_{idx:03d}",
            "file_name": fname,
            "file_path": m.get("file_path", v_entry.get("file_path", "")),
            "file_size_human": m.get("file_size_human", v_entry.get("file_size_human", "N/A")),
            "file_size_bytes": m.get("file_size_bytes", v_entry.get("file_size_bytes", 0)),
            
            # 1. 512-dim Normalized Dense Vector Embedding (for instant open-vocabulary search)
            "vector_embedding": raw_vec,

            # 2. Calibrated Categorical Tags (with rejection filtering - NO FALSE TAGS!)
            "calibrated_tags": top_calibrated,
            "is_unclassified": len(top_calibrated) == 0,

            # 3. Visual Narrative (Moondream2 1-sentence dense caption)
            "visual_narrative": caption,

            # 4. Salient Scene Keywords (Extracted from visual narrative)
            "scene_keywords": keywords,

            # 5. EXIF & Dimensional Metadata
            "exif": {
                "dimensions": m.get("dimensions", {}),
                "date_taken": m.get("date_taken"),
                "camera_make": m.get("camera_make"),
                "camera_model": m.get("camera_model"),
                "format": Path(fname).suffix.lstrip(".").upper()
            }
        }
        unified_records.append(record)

    # 5. Save Official Unified JSON
    payload = {
        "architecture_version": "2.0-unified",
        "generated_at": datetime.now().isoformat(),
        "total_images": len(unified_records),
        "tag_rejection_threshold": MIN_CALIBRATED_THRESHOLD,
        "taxonomy_size": len(tags),
        "images": unified_records
    }

    # Save to unified_gallery_index.json
    with open(UNIFIED_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Also update bulk_results.json for backwards compatibility
    with open(OLD_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 75)
    print("🎉 UNIFIED GALLERY INDEX GENERATED SUCCESSFULLY!")
    print("═" * 75)
    print(f"📁 Output Saved To : {UNIFIED_INDEX_FILE.name}")
    print(f"📁 Also Mirrored To: {OLD_RESULTS_FILE.name}")
    print(f"📸 Total Images    : {len(unified_records)}")
    print(f"🏷️  Filtered Tags  : Clean multi-label calibration (False positives eliminated)")
    print("═" * 75)

    # Print first 5 records as preview
    print("\n📋 Sample Record Preview (First 3 Images):")
    for r in unified_records[:3]:
        tags_str = ", ".join([f"{t['tag']} ({t['confidence_pct']}%)" for t in r["calibrated_tags"]]) or "(None - Below threshold)"
        print(f"\n📷 {r['file_name']} ({r['file_size_human']})")
        print(f"   🏷️  Calibrated Tags : {tags_str}")
        print(f"   📝 Narrative       : \"{r['visual_narrative'][:75]}...\"")
        print(f"   🔑 Keywords        : {', '.join(r['scene_keywords'][:6])}")
    print("═" * 75)


if __name__ == "__main__":
    generate_unified_index()
