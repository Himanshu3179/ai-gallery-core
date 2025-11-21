import psycopg2
from psycopg2.extras import execute_values
from src.config import Config

class Database:
    def __init__(self):
        self.conn = None


        if not self.conn:
            self.conn = psycopg2.connect(Config.DB_DSN)
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()

    def init_db(self):
        """Initialize the 3-Table Schema."""
        conn = self.connect()
        cur = conn.cursor()
        
        # Enable pgvector
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 1. PEOPLE Table (Identities)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # 2. IMAGE_METADATA Table (The Scene)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS image_metadata (
                id SERIAL PRIMARY KEY,
                image_path TEXT UNIQUE NOT NULL,
                caption TEXT,
                scene_embedding vector(384),
                meta_data JSONB DEFAULT '{}'::jsonb
            );
        """)
        # Index for JSON search
        cur.execute("CREATE INDEX IF NOT EXISTS idx_meta_data ON image_metadata USING gin (meta_data);")
        # Index for Vector search
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scene_vec ON image_metadata USING hnsw (scene_embedding vector_cosine_ops);")

        # 3. FACE_DETECTIONS Table (The Bridge)
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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_face_vec ON face_detections USING hnsw (face_embedding vector_cosine_ops);")

        conn.commit()
        print("Database schema initialized successfully.")