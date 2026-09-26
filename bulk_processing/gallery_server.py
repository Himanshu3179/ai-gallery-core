"""
Fast CLIP-powered gallery server.

Serves the images indexed in clip_gallery_index.json with instant
substring-free semantic search (CLIP text-image cosine similarity),
paginated infinite-scroll browsing, on-the-fly cached thumbnails, and
a single-page HTML/JS frontend.

Run:
    python bulk_processing/gallery_server.py
"""

import io
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
from transformers import CLIPModel, CLIPProcessor

CURRENT_DIR = Path(__file__).resolve().parent
INDEX_FILE = CURRENT_DIR / "clip_gallery_index_camera.json"
TAGS_FILE = CURRENT_DIR / "tags.json"
THUMB_CACHE_DIR = CURRENT_DIR / ".thumb_cache"
STATIC_DIR = CURRENT_DIR / "static"

MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_PAGE_SIZE = 40
THUMB_SIZE = 380

THUMB_CACHE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Gallery Search")


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Static assets change constantly during development — without this,
    browsers happily keep serving a stale app.js/style.css after an edit."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response

# ---------------------------------------------------------------------------
# In-memory index: loaded once at startup
# ---------------------------------------------------------------------------
records: list = []
file_name_to_path: dict = {}
embedding_matrix: Optional[np.ndarray] = None  # (N, 512), rows are unit vectors
taxonomy_tags: list = []

device: str = "cpu"
processor: Optional[CLIPProcessor] = None
model: Optional[CLIPModel] = None


def load_index():
    global records, file_name_to_path, embedding_matrix

    if not INDEX_FILE.exists():
        raise RuntimeError(f"Index file not found: {INDEX_FILE}")

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_records = data.get("images", [])
    vectors = []
    clean_records = []

    for rec in raw_records:
        path = rec.get("file_path")
        if not path or not Path(path).exists():
            continue
        vectors.append(rec["vector_embedding"])
        clean_records.append(rec)
        file_name_to_path[rec["file_name"]] = path

    records = clean_records
    embedding_matrix = np.array(vectors, dtype=np.float32)
    print(f"[gallery] Loaded {len(records)} images from {INDEX_FILE.name}")


def load_taxonomy_tags():
    global taxonomy_tags
    if not TAGS_FILE.exists():
        taxonomy_tags = []
        return
    with open(TAGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    tags = []
    for tag_list in data.get("categories", {}).values():
        tags.extend(tag_list)
    taxonomy_tags = sorted(set(tags))


def load_clip_model():
    global device, processor, model

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"[gallery] Loading CLIP text encoder onto {device}...")
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    model.eval()
    print("[gallery] CLIP ready.")


@app.on_event("startup")
def on_startup():
    load_index()
    load_taxonomy_tags()
    load_clip_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def serialize_card(rec: dict, score: Optional[float] = None) -> dict:
    dims = rec.get("exif", {}).get("dimensions", {}) or {}
    card = {
        "id": rec["id"],
        "file_name": rec["file_name"],
        "width": dims.get("width"),
        "height": dims.get("height"),
        "tags": [t["tag"] for t in rec.get("calibrated_tags", [])],
    }
    if score is not None:
        card["score"] = round(score, 4)
    return card


def serialize_detail(rec: dict, score: Optional[float] = None) -> dict:
    detail = serialize_card(rec, score)
    detail["file_size_human"] = rec.get("file_size_human")
    detail["tags"] = [
        {"tag": t["tag"], "confidence_pct": t["confidence_pct"]}
        for t in rec.get("calibrated_tags", [])
    ]
    exif = rec.get("exif", {}) or {}
    detail["date_taken"] = exif.get("date_taken")
    detail["camera"] = " ".join(
        filter(None, [exif.get("camera_make"), exif.get("camera_model")])
    ).strip() or None
    return detail


def embed_query(query: str) -> np.ndarray:
    prompts = [query, f"a photo of {query}"]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feat = feats.mean(dim=0, keepdim=True)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32).reshape(-1)


def build_thumbnail(src_path: Path, dest_path: Path):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        img.save(dest_path, "JPEG", quality=82)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/images")
def list_images(offset: int = 0, limit: int = DEFAULT_PAGE_SIZE):
    total = len(records)
    page = records[offset : offset + limit]
    return {
        "items": [serialize_card(r) for r in page],
        "offset": offset,
        "limit": limit,
        "total": total,
        "has_more": offset + limit < total,
    }


def match_taxonomy_tags(clean_q: str) -> list:
    """Find known taxonomy tags (e.g. 'dessert or cake') that the query refers
    to (e.g. 'cake'). These tags were already calibrated per-image by the
    indexer, so matching against them is far more precise than raw CLIP
    cosine similarity on free text."""
    q_lower = clean_q.lower()
    words = [w for w in q_lower.split() if len(w) > 2]
    matches = []
    for tag in taxonomy_tags:
        tag_lower = tag.lower()
        if q_lower in tag_lower or any(w in tag_lower for w in words):
            matches.append(tag_lower)
    return matches


def rank_all_by_relevance(clean_q: str, matched_tags: list) -> list:
    """Ranks every single image by relevance to the query — nothing is ever
    excluded, only reordered. Images with a calibrated taxonomy tag match
    (high precision) sort above everything else, ranked among themselves by
    tag confidence. Every other image still gets a raw CLIP similarity score
    and is ranked by that, so the full gallery is always available via
    infinite scroll in decreasing order of probability."""
    query_vec = embed_query(clean_q)
    vec_scores = embedding_matrix @ query_vec

    ranked = []
    for idx, rec in enumerate(records):
        tag_score = 0.0
        if matched_tags:
            for t in rec.get("calibrated_tags", []):
                if t["tag"].lower() in matched_tags:
                    tag_score = max(tag_score, t["confidence_pct"] / 100.0)
        vec_score = float(vec_scores[idx])
        display_score = tag_score if tag_score > 0 else vec_score
        # Sort key: tag matches first (grouped by confidence), then everyone
        # else by raw vector similarity — never dropped, just ranked lower.
        ranked.append(((tag_score, vec_score), display_score, rec))

    ranked.sort(key=lambda entry: entry[0], reverse=True)
    return [(display_score, rec) for _, display_score, rec in ranked]


@app.get("/api/search")
def search_images(
    q: str = Query(default=""),
    offset: int = 0,
    limit: int = DEFAULT_PAGE_SIZE,
):
    clean_q = q.strip()
    if not clean_q:
        return list_images(offset=offset, limit=limit)

    matched_tags = match_taxonomy_tags(clean_q)
    ranked = rank_all_by_relevance(clean_q, matched_tags)

    total = len(ranked)
    page = ranked[offset : offset + limit]

    return {
        "items": [serialize_card(rec, score) for score, rec in page],
        "offset": offset,
        "limit": limit,
        "total": total,
        "has_more": offset + limit < total,
        "query": clean_q,
    }


@app.get("/api/image/{image_id}")
def get_image_detail(image_id: str):
    for rec in records:
        if rec["id"] == image_id:
            return serialize_detail(rec)
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/thumb/{file_name}")
def get_thumbnail(file_name: str):
    src_path = file_name_to_path.get(file_name)
    if not src_path:
        raise HTTPException(status_code=404, detail="Not found")

    cache_path = THUMB_CACHE_DIR / f"{file_name}.jpg"
    if not cache_path.exists():
        try:
            build_thumbnail(Path(src_path), cache_path)
        except Exception:
            raise HTTPException(status_code=500, detail="Could not build thumbnail")

    return FileResponse(cache_path, media_type="image/jpeg")


@app.get("/media/{file_name}")
def get_full_image(file_name: str):
    src_path = file_name_to_path.get(file_name)
    if not src_path:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(src_path)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_page():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8420, log_level="info")
