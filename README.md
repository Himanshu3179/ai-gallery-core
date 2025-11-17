# 📸 Prospo AI Gallery (Core)

> A local-first, privacy-focused photo management engine capable of semantic search ("a cat sleeping on a sofa") and facial recognition ("photos of Himanshu").

## 🌟 Features

- **Semantic Understanding:** Uses **Moondream2** (VLM) to generate detailed descriptions of every image.
- **Face Recognition:** Uses **ArcFace** (DeepFace) to detect and identify people with >99% accuracy.
- **Vector Search:** Stores embeddings in **PostgreSQL (pgvector)** for lightning-fast similarity queries.
- **Hybrid Querying:** Supports complex queries like _"Show me photos of Himanshu at the beach."_
- **Privacy-First:** Runs 100% locally on your machine (Mac M-Series optimized).

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **Database:** PostgreSQL 16 + `pgvector` extension
- **AI Models:**
  - _Scene:_ `vikhyatk/moondream2` (Vision-Language Model)
  - _Faces:_ `ArcFace` (via DeepFace)
  - _Embeddings:_ `all-MiniLM-L6-v2` (Sentence Transformers)
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
