from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncpg
from src.config import Config
from src.models.scene import SceneModel
import os
from fastapi.responses import FileResponse
# Global variables
ml_models = {}
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    global db_pool
    print("Connecting to Database...")
    
    # UPDATED: Using your config's variable name 'DB_DSN'
    db_pool = await asyncpg.create_pool(Config.DB_DSN)
    
    print("Loading AI Models (Text Embedding only)...")
    ml_models["scene"] = SceneModel()
    
    yield
    
    # --- SHUTDOWN ---
    print("Shutting down...")
    if db_pool:
        await db_pool.close()
    if "scene" in ml_models:
        ml_models["scene"].unload()
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# Allow frontend to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER ---
def get_scene_model():
    if "scene" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ml_models["scene"]

# --- APIs ---

@app.get("/feed")
async def get_random_feed(limit: int = 50, offset: int = 0, seed: str = "default"):
    """
    Returns random-ish images but STABLE for a given seed.
    Pagination works via offset.
    """
    async with db_pool.acquire() as conn:
        # We use md5(id + seed) to create a random sort order that is consistent
        # for the same seed.
        rows = await conn.fetch("""
            SELECT id, image_path, caption, meta_data 
            FROM image_metadata 
            ORDER BY md5(id::text || $3) 
            LIMIT $1 OFFSET $2
        """, limit, offset, seed)
        return [dict(row) for row in rows]

@app.get("/search")
async def search_images(q: str, limit: int = 20):
    """Text -> Vector -> DB Search"""
    model = get_scene_model()
    
    # 1. Convert text to vector
    query_vector = model.get_embedding(q)
    
    # 2. Search DB using Cosine Similarity (<=>)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, image_path, caption, meta_data, 
                   1 - (scene_embedding <=> $1) as similarity
            FROM image_metadata
            ORDER BY scene_embedding <=> $1
            LIMIT $2
        """, str(query_vector), limit)
        return [dict(row) for row in rows]

@app.get("/similar/{image_id}")
async def get_similar(image_id: int, limit: int = 10):
    """Image-to-Image: Find neighbors of a specific image ID"""
    async with db_pool.acquire() as conn:
        # 1. Get the vector of the target image
        target_row = await conn.fetchrow(
            "SELECT scene_embedding FROM image_metadata WHERE id = $1", 
            image_id
        )
        if not target_row:
            raise HTTPException(status_code=404, detail="Image not found")
        
        vector = target_row['scene_embedding']

        # 2. Find closest vectors (excluding itself)
        rows = await conn.fetch("""
            SELECT id, image_path, caption, meta_data,
                   1 - (scene_embedding <=> $1) as similarity
            FROM image_metadata
            WHERE id != $2
            ORDER BY scene_embedding <=> $1
            LIMIT $3
        """, vector, image_id, limit)
        
        return [dict(row) for row in rows]
    
@app.get("/image/{image_id}")
async def get_image_file(image_id: int):
    """Serves the raw image file to the frontend"""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT image_path FROM image_metadata WHERE id = $1", image_id)
        if not row or not os.path.exists(row['image_path']):
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(row['image_path'])

@app.get("/image_details/{image_id}")
async def get_image_details(image_id: int):
    """Returns metadata for a specific image ID"""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, image_path, caption, meta_data FROM image_metadata WHERE id = $1",
            image_id
        )
        if not row:
            raise HTTPException(status_code=404, detail="Image details not found")
        return dict(row)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)