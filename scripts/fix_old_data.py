import sys
import os

# Add the project root to Python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from src.database import Database
from src.pipeline.ingest import IngestionPipeline

def fix_metadata():
    print("Connecting to database...")
    db = Database()
    conn = db.connect()
    cur = conn.cursor()
    
    # Reuse the helper from your main pipeline
    # (This is safe, it doesn't load AI models in __init__)
    pipeline = IngestionPipeline()

    # 1. Find only the broken rows
    print("Fetching rows with empty metadata...")
    # Note: We cast to ::text to catch both JSONB '{}' and text '{}'
    cur.execute("SELECT id, image_path FROM image_metadata WHERE meta_data::text = '{}'")
    rows = cur.fetchall()

    if not rows:
        print("No empty metadata found! You are good to go.")
        return

    print(f"Found {len(rows)} images to fix.")

    # 2. Fix them
    for row_id, path in tqdm(rows):
        # Calculate fresh metadata
        meta_json = pipeline._get_metadata(path)
        
        # Update the row
        cur.execute(
            "UPDATE image_metadata SET meta_data = %s WHERE id = %s",
            (meta_json, row_id)
        )
    
    conn.commit()
    conn.close()
    print("✅ Fixed all old metadata.")

if __name__ == "__main__":
    fix_metadata()