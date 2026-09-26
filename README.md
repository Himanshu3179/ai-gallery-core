# 📸 Prospo AI Gallery (Core)

> A local-first, privacy-focused photo management engine capable of semantic search ("a cat sleeping on a sofa") and facial recognition ("photos of Himanshu").

## 🌟 Features

- **Semantic Understanding:** Uses **Moondream2** (VLM) to generate detailed descriptions of every image.
- **Face Recognition:** Uses **InsightFace (Buffalo_L)** with Apple Neural Engine optimization to detect and identify people with >99% accuracy.
- **Vector Search:** Stores embeddings in **PostgreSQL (pgvector)** for lightning-fast similarity queries.
- **Hybrid Querying:** Supports complex queries like _"Show me photos of Himanshu at the beach."_
- **Privacy-First:** Runs 100% locally on your machine (Mac M-Series optimized).

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **Database:** PostgreSQL 16 + `pgvector` extension
- **AI Models:**
  - _Scene:_ `vikhyatk/moondream2` (Vision-Language Model, float16)
  - _Faces:_ `InsightFace Buffalo_L` (CoreML/Apple Neural Engine optimized)
  - _Embeddings:_ `all-MiniLM-L6-v2` (Sentence Transformers)
  - _Benchmarks & Scaling:_ [Performance & Batch Optimization Guide](docs/PERFORMANCE_AND_OPTIMIZATIONS.md)
- **Infrastructure:** Docker Compose

---

## 🗄️ Database Schema

The system uses a normalized 3-table architecture:

1. **`image_metadata`** (The Context)

   - Stores image paths, detailed AI captions, and JSON metadata.
   - Vector Index: _HNSW_ on Scene Embeddings.

2. **`people`** (The Identities)

   - Unique list of known persons (e.g., "Himanshu", "Sarah").

3. **`face_detections`** (The Link)
   - Links specific faces found in an image to a `person_id`.
   - Stores the raw Face Vector for accurate re-clustering.

---

## 🚀 Getting Started

### 1. Prerequisites

- Docker Desktop installed & running.
- Python 3.10+ installed.
- Mac M-Series chip (optimized for Apple Neural Engine).

### 2. Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd ai-gallery-core

# 2. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Start Database
docker-compose up -d
```

### 3. Configure Image Source

Edit the `.env` file to point to your image directory:

```bash
IMAGE_SOURCE_DIR=/path/to/your/photos
```

---

## 📋 Usage Guide

### 🆕 Fresh Start (First Time Setup)

If you're setting up the project for the first time or want to start from scratch:

```bash
# 1. Ensure database is running
docker-compose up -d

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Run the ingestion pipeline
python -m src.pipeline.ingest
```

This will:
- **Phase 1:** Analyze all images with Moondream2 (scene understanding)
- **Phase 2:** Detect and extract face embeddings with InsightFace
- **Phase 3:** Cluster faces into unique identities using DBSCAN

### ➕ Adding New Images (Incremental Update)

To add new images on top of existing data (e.g., adding 100 new photos to your existing 1000s):

```bash
# 1. Copy new images to your IMAGE_SOURCE_DIR
# 2. Run the same ingestion command
python -m src.pipeline.ingest
```

**Smart Resume:** The pipeline automatically:
- ✅ Skips already processed images (checks `image_metadata` table)
- ✅ Only processes new images in Phase 1 & 2
- ✅ Re-runs clustering (Phase 3) to update identities with new faces

### 🔄 Resume After Interruption

If the process stops or crashes during ingestion:

```bash
# Simply re-run the ingestion command
python -m src.pipeline.ingest
```

**Resume Behavior by Phase:**

| Phase | Resume Capability | Details |
|-------|------------------|---------|
| **Phase 1** (Scene Analysis) | ✅ **Full Resume** | Skips images already in `image_metadata` table |
| **Phase 2** (Face Detection) | ✅ **Full Resume** | Skips images already processed for faces |
| **Phase 3** (Clustering) | ⚠️ **Always Runs** | Re-clusters ALL faces to update identities |

**Note:** Phase 3 (clustering) always runs to ensure face groupings are accurate with the complete dataset.

### 🔄 Re-cluster Faces Only

If you want to re-cluster existing faces without re-processing images:

```bash
# The clustering runs automatically, but you can also use:
python -m src.pipeline.ingest
# (It will skip Phase 1 & 2 if no new images are found)
```

### 🌐 Start the Web Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Start FastAPI backend
python -m src.server

# In another terminal, start the frontend
cd frontend
npm install
npm run dev
```

Access the gallery at `http://localhost:5173`

---

## 🛠 Troubleshooting

### Database Connection Issues

```bash
# Check if PostgreSQL is running
docker-compose ps

# Restart database
docker-compose restart

# View logs
docker-compose logs -f
```

### Memory Issues During Processing

If you encounter memory issues with large image collections:
- Process images in smaller batches
- Reduce `det_size` in `src/models/face.py` (currently 640x640)
- Close other applications to free up RAM

### Re-initialize Database (⚠️ Deletes All Data)

```bash
# Stop and remove containers
docker-compose down -v

# Start fresh
docker-compose up -d
python -m src.pipeline.ingest
```
