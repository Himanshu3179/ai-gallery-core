# 🚀 Bulk Image Processing & Hybrid Search Engine

This directory contains the tools, datasets, and benchmark implementations for **High-Throughput Bulk Ingestion (10,000+ photos)** and **Triple-Tier Open-Vocabulary Hybrid Search** on Apple Silicon (MPS).

For the full technical diagnosis of why naive tag classification failed and how the hybrid architecture resolves it, read:
👉 **[Search & Tagging Architecture & Failure Analysis](file:///Users/apple/Desktop/projects/ai-gallery-core/docs/SEARCH_AND_TAGGING_ARCHITECTURE.md)**

---

## 📁 Directory Structure

```
bulk_processing/
├── tags.json               # Curated taxonomy across 8 diverse categories
├── bulk_indexer.py         # GPU batch image-to-vector indexer (~176ms/image for 100 images)
├── bulk_moondream.py       # Moondream2 1-sentence caption generator & EXIF extractor
├── hybrid_search.py        # Core Triple-Tier Hybrid Engine (Vector + Caption + OCR)
├── search.py               # Fast CLI & interactive search entrypoint
├── bulk_results.json       # Pre-computed 512-d CLIP vectors & metadata for all 100 images
├── moondream_results.json  # Full 100-image captions, EXIF, and latency logs
├── COMPARISON_REPORT.md    # 100-image side-by-side table (CLIP tags vs. Moondream captions)
└── README.md               # This documentation
```

---

## ⚡ The Winning Hybrid Architecture

| Layer | Technology | Latency | Purpose |
|---|---|---|---|
| **Tier 1: Semantic Vector Space** | CLIP (`ViT-B/32`) | **~1 – 3 ms** | Open-vocabulary search across any arbitrary concept/word |
| **Tier 2: Dense Visual Narrative** | Moondream2 (`1.86B`) | **~3.5 s** | OCR, specific brands/names, actions, and fine-grained visual relationships |
| **Tier 3: Score Fusion** | Hybrid Ranker | **< 0.1 ms** | Combines vector cosine similarity with exact phrase/keyword matches |

$$\text{Final Score} = 0.65 \times \text{CosineSim}(\mathbf{q}, \mathbf{d}_{\text{image}}) + 0.35 \times \text{Score}_{\text{CaptionText}}$$

---

## 🚀 How to Run

### 1. Ingest & Index New Images (CLIP Vectorizer)
```bash
python bulk_processing/bulk_indexer.py --batch-size 16 --top-k 5
```

### 2. Generate 1-Sentence Descriptions (Moondream2)
```bash
python bulk_processing/bulk_moondream.py --prompt "Briefly describe this image in 1 concise sentences."
```

### 3. Run Hybrid Search (Command-Line)
Search for any entity, object, action, brand, or scene:
```bash
python bulk_processing/search.py --query "basketball"
python bulk_processing/search.py --query "netflix"
python bulk_processing/search.py --query "pink umbrella"
python bulk_processing/search.py --query "birthday cake"
python bulk_processing/search.py --query "temple"
python bulk_processing/search.py --query "red sports car"
```

### 4. Interactive Live Search Explorer
Launch the interactive terminal interface:
```bash
python bulk_processing/search.py
```
