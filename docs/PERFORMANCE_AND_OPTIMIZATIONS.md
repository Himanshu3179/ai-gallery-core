# ⚡ VLM Performance & Batch Optimization Guide

This document captures the empirical performance benchmarks, architectural bottlenecks, and optimization blueprint for Vision-Language Models (specifically **Moondream2**) running on Apple Silicon (MPS) within `ai-gallery-core`.

---

## 📌 1. Pipeline Latency Breakdown

When running inference on an image, the pipeline consists of four distinct phases:

```
[ Image Path ] 
      │
      ▼
┌─────────────────────────────────┐
│ 1. Model Load into GPU (MPS)    │  ~9.0s – 9.4s (Disk → Unified Memory / VRAM)
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ 2. Image Read & EXIF Metadata   │  ~0.05s – 0.09s (Pillow decode & EXIF parse)
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ 3. Vision Encoder (SigLIP)      │  ~0.95s – 1.03s (Vision Transformer forward pass)
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ 4. Autoregressive Text Gen      │  ~2.50s – 2.80s (Phi-1.5 token decoding)
└─────────────────────────────────┘
      │
      ▼
[ Caption & Metadata JSON ]
```

### Empirical Measurements (Sample Benchmark):
* **Phase 1 (Model Load into GPU):** `9.07s` – Cold model initialization from SSD to Apple GPU (MPS).
* **Phase 2 (Image Read & EXIF):** `0.05s` – Image loading and EXIF tag parsing.
* **Phase 3 (Vision Encoder):** `0.96s` – SigLIP image embedding generation.
* **Phase 4 (Text Generation):** `2.53s` – Language model generating a concise 1-sentence caption.
* **Pure GPU Inference (Phases 3 + 4):** `~3.49s`
* **Total CLI Script Latency:** `12.61s`

> [!NOTE]
> In a single-image CLI invocation (`test_single.py`), the model must be loaded from scratch every time (~9s). In batch ingestion (`src/pipeline/ingest.py`), the model is loaded **only once** and kept in memory.

---

## 🚀 2. Optimizations Applied

### A. Autograd Elimination
* **Action:** Wrapped inference inside `with torch.inference_mode():` and called `model.eval()`.
* **Impact:** Eliminates PyTorch backward-graph construction and GPU memory tracking overhead on MPS.

### B. Prompt & Token Cap Optimization
* **Initial State:** Prompted for an exhaustive paragraph (`"Describe this image in detail..."`), taking **~8.8s** with ~120 generated tokens.
* **Optimized State:** Prompted for a concise 1-sentence description with `max_new_tokens=60` (or `35`):
  * Concise prompt (`max_new_tokens=60`): **2.71s** (Over **3x faster**).
  * Strict 1-sentence (`max_new_tokens=35`): **2.38s** (Nearly **4x faster**).
* **Quality Retention:** Retains all essential search keywords (people, clothing, actions, background objects, colors, settings) without redundant filler words.

### C. Log Noise Suppression
* Suppressed verbose Hugging Face `transformers` warnings (`PhiForCausalLM has generative capabilities...`) and tokenizers parallelism messages for clean, structured CLI logs.

---

## 📦 3. Bulk & Batch Ingestion Blueprint (100–200+ Photos)

When processing bulk images in [`src/pipeline/ingest.py`](../src/pipeline/ingest.py), apply the following strategies:

### 1. Amortized Model Loading (Automatic Win)
* In batch mode, the `9.07s` model loading penalty occurs **once**.
* For 100 images, overhead drops to **0.09s per image**.

### 2. Batched Vision Encoding (SigLIP Parallelism)
* Currently, images are encoded one at a time (`0.96s` each).
* The SigLIP vision encoder supports batching. By passing a batch of 8 images (`model.encode_image(batch_of_8)`):
  * Sequential: `8 × 0.96s = 7.68s`
  * Batched on Apple GPU: **~1.5s total** (~5x throughput on vision encoding).

### 3. Asynchronous Image Preloading (`ThreadPoolExecutor`)
* While the GPU is busy generating captions for batch $N$, Python background worker threads can load, resize, and parse EXIF for batch $N+1$.
* **Impact:** Hides disk I/O completely (`0s` perceived image read latency).

### 4. Batch Database Commits
* Avoid committing to PostgreSQL (`conn.commit()`) after every single image.
* Accumulate 50–100 records and insert in bulk via `execute_values` or `execute_batch` to prevent database I/O blocking.

---

## 📊 4. Projected Bulk Ingestion Comparison (100 Images)

| Pipeline Phase | Unoptimized Sequential | Optimized Batch Pipeline |
| :--- | :---: | :---: |
| **Model Load** | 9.0s (once) | 9.0s (once) |
| **Image Read & EXIF** | 5.0s (50ms × 100) | **0s** (Preloaded in background thread) |
| **Vision Encoding** | 96.0s (0.96s × 100) | **~25s** (Batched in chunks of 8) |
| **Text Generation** | 250.0s (2.5s × 100) | **~140s** (Tuned token cap & KV cache) |
| **Database Writes** | 8.0s (100 individual commits) | **0.2s** (Single bulk insert) |
| **Total Ingestion Time** | **~6.1 minutes** | **~2.8 minutes** (>2x overall speedup) |

---

## 🔍 5. Alternative Models Comparison

| Model | Parameters | Latency on Apple Silicon | Pros | Cons for Gallery Use |
| :--- | :--- | :--- | :--- | :--- |
| **Moondream2** *(Current)* | ~1.86B | ~2.5s – 3.5s per image (MPS) | Solid scene comprehension, lightweight memory footprint. | Autoregressive token decoding is bottlenecked on raw PyTorch MPS. |
| **SmolVLM-500M** | 500M | **~0.8s – 1.2s** per image | Extremely fast, under 1 GB VRAM, compact token footprint. | Slightly shorter captions, less nuanced detail on complex scenes. |
| **Qwen2.5-VL-3B** | ~3B | ~3.0s – 4.5s per image | Best-in-class OCR, reads text on shirts/signs, sharp spatial detection. | Higher VRAM requirement (~3.5 GB in 4-bit). |
| **MLX-VLM (Engine)** | 1.8B – 3B | **~0.4s – 0.8s** per image | Direct Metal / Neural Engine execution via Apple's native MLX framework. | Mac-only (ideal for local Apple Silicon deployments). |
| **CLIP / SigLIP (Dual-Encoder)** | ~150M – 220M | **~3.5 ms (0.0035s)** per image ⚡ | Direct image-to-vector embedding. Indexes **10,000 photos in ~35 seconds**. | No natural language prose output (vectors only). |

---

## ⚡ 6. The 10,000+ Image Scaling Blueprint: Direct Dual-Encoder

Tested live on Apple Silicon GPU (`single_image_test/test_dual_encoder.py`):
* **Model:** `openai/clip-vit-base-patch32` (512-dimension vector)
* **Single Image Latency:** **3.47 ms – 3.56 ms**
* **Throughput:** **~285 images/second** on a single thread.
* **10,000 Images Indexing Time:** **~35 seconds total**.
* **Zero-Shot Search Accuracy:**
  * For gym photo: `"person smiling in a gym"` scored **0.299** (Rank 1).
  * For singer photo: `"a man singing on stage with microphone"` scored **0.258** (Rank 1).
  * For screenshot: `"video call on computer screen"` scored **0.293** (Rank 1).

### Recommended Two-Tier Production Architecture:
1. **Tier 1 (Instant Vector Ingestion):** Index 10,000 photos with CLIP in **<1 minute** so all photos are immediately searchable.
2. **Tier 2 (Background Prose & Faces):** Run Moondream2 & InsightFace as an idle background daemon to generate rich captions and face clusters.
