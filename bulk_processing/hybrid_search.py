import os
import re
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

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
UNIFIED_INDEX_FILE = CURRENT_DIR / "unified_gallery_index.json"
CLIP_RESULTS_FILE = CURRENT_DIR / "bulk_results.json"
MOONDREAM_RESULTS_FILE = CURRENT_DIR / "moondream_results.json"
DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


class HybridSearchEngine:
    def __init__(self, device: str = None):
        self.device = device or Config.DEVICE
        print("═" * 75)
        print("🧠 INITIALIZING HYBRID SEARCH ENGINE (Vector + Dense Caption + OCR)")
        print(f"💻 Compute Device : {self.device.upper()}")
        
        # 1. Load Unified Gallery Index (New Architecture)
        unified_data = self._load_json(UNIFIED_INDEX_FILE)
        if unified_data and "images" in unified_data:
            self.records = unified_data["images"]
            print(f"📄 Loaded Index   : unified_gallery_index.json (Architecture v{unified_data.get('architecture_version', '2.0')})")
        else:
            self.clip_data = self._load_json(CLIP_RESULTS_FILE)
            self.moondream_data = self._load_json(MOONDREAM_RESULTS_FILE)
            self.records = self._merge_records()
            print(f"📄 Loaded Index   : fallback merged files")

        print(f"📸 Indexed Images : {len(self.records)}")

        # 2. Build GPU Vector Matrix
        print("⚡ Caching Image Vectors in Unified GPU Memory...")
        t0_vec = time.perf_counter()
        vectors = [r.get("vector_embedding") or r.get("embedding", []) for r in self.records]
        self.img_matrix = torch.tensor(vectors, dtype=torch.float16).to(self.device)
        print(f"   ✅ Matrix shape ({self.img_matrix.shape[0]} × {self.img_matrix.shape[1]}) cached in {(time.perf_counter() - t0_vec)*1000:.1f}ms")

        # 3. Load Model
        print("⏳ Loading CLIP Text Encoder onto GPU...")
        t0_model = time.perf_counter()
        self.processor = CLIPProcessor.from_pretrained(DEFAULT_MODEL_ID)
        self.model = CLIPModel.from_pretrained(DEFAULT_MODEL_ID, torch_dtype=torch.float16).to(self.device)
        self.model.eval()
        print(f"   ✅ CLIP loaded in {time.perf_counter() - t0_model:.2f}s")
        print("═" * 75)

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: could not load {path.name}: {e}")
            return {}

    def _merge_records(self) -> list:
        clip_images = self.clip_data.get("images", [])
        moon_images = self.moondream_data.get("images", [])
        moon_map = {item["file_name"]: item for item in moon_images}

        merged = []
        for c in clip_images:
            fname = c["file_name"]
            m = moon_map.get(fname, {})
            merged.append({
                "file_name": fname,
                "file_path": c.get("file_path", m.get("file_path", "")),
                "file_size": c.get("file_size_human", m.get("file_size_human", "N/A")),
                "embedding": c.get("embedding", []),
                "caption": m.get("moondream_caption", ""),
                "clip_tags": c.get("top_tags", []),
                "date_taken": m.get("date_taken", "N/A"),
                "camera": f"{m.get('camera_make', '')} {m.get('camera_model', '')}".strip() or "N/A"
            })
        return merged

    def search(self, query: str, top_k: int = 5, vector_weight: float = 0.65, text_weight: float = 0.35) -> list:
        t0 = time.perf_counter()
        clean_q = query.strip().lower()
        if not clean_q:
            return []

        # 1. Encode text query into 512-d vector (Open-Vocabulary Vector Search)
        prompts = [clean_q, f"a photo of {clean_q}"]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            t_feats = self.model.get_text_features(**inputs)
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
            # Average prompt ensemble
            t_feat = t_feats.mean(dim=0, keepdim=True)
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)

            # Cosine similarity matrix multiplication
            cos_sims = (t_feat @ self.img_matrix.T).squeeze(0).cpu().tolist()

        # 2. Compute Keyword / Caption Match Score
        query_words = [w for w in re.split(r"\W+", clean_q) if len(w) > 2]
        pattern = re.compile(rf"\b{re.escape(clean_q)}\b", re.IGNORECASE) if len(clean_q) > 2 else None

        scored_results = []
        for idx, rec in enumerate(self.records):
            v_score = cos_sims[idx]
            raw_caption = rec.get("visual_narrative") or rec.get("caption", "")
            caption = raw_caption.lower()
            tags = rec.get("calibrated_tags") or rec.get("clip_tags", [])
            file_size = rec.get("file_size_human") or rec.get("file_size", "N/A")

            # Text bonus calculation
            t_score = 0.0
            reasons = []

            # Exact phrase match in Moondream caption
            if pattern and pattern.search(caption):
                t_score += 0.40
                reasons.append("Exact phrase in visual caption")
            else:
                # Word-level overlap in caption
                matched_words = [w for w in query_words if w in caption]
                if matched_words:
                    word_ratio = len(matched_words) / max(len(query_words), 1)
                    t_score += 0.25 * word_ratio
                    reasons.append(f"Caption matched: {', '.join(matched_words)}")

            # Calibrated tag match
            matched_tags = [
                t["tag"] for t in tags
                if clean_q in t["tag"].lower() or any(w in t["tag"].lower() for w in query_words)
            ]
            if matched_tags:
                t_score += 0.15
                reasons.append(f"Tag matched: {matched_tags[0]}")

            # Fused Hybrid Score (normalized to 0-1 range)
            final_score = (vector_weight * v_score) + (text_weight * t_score)

            scored_results.append({
                "file_name": rec["file_name"],
                "file_path": rec["file_path"],
                "file_size": file_size,
                "caption": raw_caption,
                "vector_score": round(v_score, 4),
                "text_score": round(t_score, 4),
                "final_score": round(final_score, 4),
                "reasons": reasons,
                "tags": tags
            })

        # Rank by final fused score
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        search_time_ms = (time.perf_counter() - t0) * 1000

        return scored_results[:top_k], search_time_ms


def print_search_results(results: list, query: str, elapsed_ms: float):
    print("\n" + "═" * 75)
    print(f"🔎 HYBRID SEARCH RESULTS FOR: \"{query}\" ({len(results)} matches in {elapsed_ms:.2f} ms)")
    print("═" * 75)

    if not results:
        print("  ⚠️  No matching images found.")
        print("═" * 75)
        return

    for rank, r in enumerate(results, 1):
        print(f"#{rank:<2} 📷 {r['file_name']:<30} | Score: {r['final_score']:.3f} (Vec: {r['vector_score']:.3f}) | Size: {r['file_size']}")
        if r["caption"]:
            print(f"    📝 Caption : \"{r['caption']}\"")
        if r["reasons"]:
            print(f"    💡 Matches : {' | '.join(r['reasons'])}")
        print(f"    📂 Path    : {r['file_path']}")
        print("─" * 75)


def interactive_mode(engine: HybridSearchEngine):
    print("\n" + "═" * 75)
    print("💬 INTERACTIVE HYBRID SEARCH (Type any query, phrase, or keyword)")
    print("   Try: 'basketball', 'netflix', 'shoes', 'pink umbrella', 'temple', 'cake', 'red car'")
    print("   Type 'q' or 'exit' to quit.")
    print("═" * 75)

    while True:
        try:
            q = input("\n🔎 Enter search query: ").strip()
            if not q or q.lower() in ("q", "exit", "quit"):
                print("👋 Exiting search.")
                break

            results, elapsed_ms = engine.search(q, top_k=5)
            print_search_results(results, q, elapsed_ms)
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Hybrid Image Search Engine (Vector + Dense Caption + OCR)")
    parser.add_argument("--query", "-q", type=str, help="Search query string")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results (default: 5)")
    args = parser.parse_args()

    engine = HybridSearchEngine()

    if args.query:
        results, elapsed_ms = engine.search(args.query, top_k=args.top_k)
        print_search_results(results, args.query, elapsed_ms)
    else:
        interactive_mode(engine)


if __name__ == "__main__":
    main()
