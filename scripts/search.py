import sys
import argparse
import psycopg2
from sentence_transformers import SentenceTransformer

# Hack to import from src from sibling folder
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

def search_text(query):
    print(f"Searching for concept: '{query}'...")
    model = SentenceTransformer(Config.EMBED_MODEL)
    vector = model.encode(query).tolist()
    
    conn = psycopg2.connect(Config.DB_DSN)
    cur = conn.cursor()
    
    sql = """
        SELECT image_path, caption, 1 - (scene_embedding <=> %s::vector) as similarity
        FROM image_metadata
        ORDER BY scene_embedding <=> %s::vector ASC
        LIMIT 5;
    """
    cur.execute(sql, (vector, vector))
    results = cur.fetchall()
    
    print("\n--- Results ---")
    for path, cap, score in results:
        print(f"[{score:.2f}] {os.path.basename(path)}\n      Caption: {cap[:100]}...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/search.py 'your query'")
    else:
        search_text(sys.argv[1])