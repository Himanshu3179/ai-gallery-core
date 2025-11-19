import sys
import os
import psycopg2

# Add project root to path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import Config

def recreate_tables():
    print("--- RE-CREATING DATABASE TABLES ---")
    
    try:
        conn = psycopg2.connect(Config.DB_DSN)
        cur = conn.cursor()

        # 1. Create PEOPLE Table
        print("Creating table: people...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # 2. Create FACE_DETECTIONS Table
        print("Creating table: face_detections...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_detections (
                id SERIAL PRIMARY KEY,
                image_id INT REFERENCES image_metadata(id) ON DELETE CASCADE,
                person_id INT REFERENCES people(id) ON DELETE SET NULL,
                face_embedding vector(512),
                location_box JSONB,
                confidence FLOAT
            );
        """)

        # 3. Create HNSW Index (Critical for speed)
        print("Creating index: idx_face_vec...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_face_vec 
            ON face_detections 
            USING hnsw (face_embedding vector_cosine_ops);
        """)

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Tables created successfully.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    recreate_tables()