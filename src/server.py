from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncpg
from src.config import Config
from src.models.scene import SceneModel
import os
import json
import io
from PIL import Image  # Needed for cropping
from fastapi.responses import FileResponse, StreamingResponse # Added StreamingResponse
from pydantic import BaseModel

# Global variables
ml_models = {}
db_pool = None

# --- Pydantic Models ---
class PersonUpdate(BaseModel):
    name: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    global db_pool
    print("Connecting to Database...")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_scene_model():
    if "scene" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ml_models["scene"]

# --- APIs ---

@app.get("/feed")
async def get_random_feed(limit: int = 50, offset: int = 0, seed: str = "default"):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, image_path, caption, meta_data 
            FROM image_metadata 
            ORDER BY md5(id::text || $3) 
            LIMIT $1 OFFSET $2
        """, limit, offset, seed)
        return [dict(row) for row in rows]

@app.get("/search")
async def search_images(q: str, limit: int = 20):
    model = get_scene_model()
    query_vector = model.get_embedding(q)
    
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
    async with db_pool.acquire() as conn:
        target_row = await conn.fetchrow(
            "SELECT scene_embedding FROM image_metadata WHERE id = $1", 
            image_id
        )
        if not target_row:
            raise HTTPException(status_code=404, detail="Image not found")
        
        vector = target_row['scene_embedding']

        rows = await conn.fetch("""
            SELECT id, image_path, caption, meta_data,
                   1 - (scene_embedding <=> $1) as similarity
            FROM image_metadata
            WHERE id != $2
            ORDER BY scene_embedding <=> $1
            LIMIT $3
        """, vector, image_id, limit)
        
        return [dict(row) for row in rows]

# --- PEOPLE APIs ---

@app.get("/people")
async def get_people():
    """Returns list of people. We DO NOT return cover_image_id here anymore, logic is in thumbnail."""
    async with db_pool.acquire() as conn:
        query = """
            SELECT p.id, p.name, COUNT(fd.id) as face_count
            FROM people p
            LEFT JOIN face_detections fd ON p.id = fd.person_id
            GROUP BY p.id
            HAVING COUNT(fd.id) > 0
            ORDER BY face_count DESC
        """
        rows = await conn.fetch(query)
        return [dict(row) for row in rows]

@app.get("/people/{person_id}/thumbnail")
async def get_person_thumbnail(person_id: int):
    """
    Generates a smart crop of the person's best face.
    Adds padding to make it look like a nice portrait.
    """
    async with db_pool.acquire() as conn:
        # 1. Find the best face (highest confidence) for this person
        row = await conn.fetchrow("""
            SELECT im.image_path, fd.location_box 
            FROM face_detections fd
            JOIN image_metadata im ON fd.image_id = im.id
            WHERE fd.person_id = $1
            ORDER BY fd.confidence DESC
            LIMIT 1
        """, person_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Face not found")
        
        try:
            # 2. Open Image
            img = Image.open(row['image_path']).convert('RGB')
            
            # 3. Parse Box (Stored as JSON in DB)
            box = json.loads(row['location_box']) # {x, y, w, h}
            
            x, y, w, h = box['x'], box['y'], box['w'], box['h']

            # 4. Add Padding (50% expansion) for a "Portrait" look
            pad_x = w * 0.5
            pad_y = h * 0.5 
            
            # Clamp coordinates to image boundaries
            crop_x1 = max(0, x - pad_x)
            crop_y1 = max(0, y - pad_y)
            crop_x2 = min(img.width, x + w + pad_x)
            crop_y2 = min(img.height, y + h + pad_y)
            
            # 5. Crop
            face_crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            
            # 6. Save to Bytes
            img_byte_arr = io.BytesIO()
            face_crop.save(img_byte_arr, format='JPEG', quality=90)
            img_byte_arr.seek(0)
            
            # UPDATED: Added Cache-Control header
            return StreamingResponse(
                img_byte_arr, 
                media_type="image/jpeg", 
                headers={"Cache-Control": "public, max-age=31536000"}
            )
            
        except Exception as e:
            print(f"Thumbnail Error: {e}")
            raise HTTPException(status_code=500, detail="Could not process image")

@app.get("/people/{person_id}")
async def get_person_images(person_id: int, limit: int = 50):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT DISTINCT im.id, im.image_path, im.caption, im.meta_data
            FROM image_metadata im
            JOIN face_detections fd ON im.id = fd.image_id
            WHERE fd.person_id = $1
            LIMIT $2
        """, person_id, limit)
        
        return [dict(row) for row in rows]

@app.put("/people/{person_id}")
async def rename_person(person_id: int, person: PersonUpdate):
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE people SET name = $1 WHERE id = $2", 
            person.name, person_id
        )
        if result == "UPDATE 0":
            raise HTTPException(status_code=404, detail="Person not found")
        return {"status": "success", "name": person.name}

# --- ASSET SERVING ---

@app.get("/image/{image_id}")
async def get_image_file(image_id: int):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT image_path FROM image_metadata WHERE id = $1", image_id)
        if not row or not os.path.exists(row['image_path']):
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(row['image_path'])

@app.get("/image_details/{image_id}")
async def get_image_details(image_id: int):
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