import psycopg2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

def reset():
    confirm = input("Are you sure you want to DELETE ALL DATA? (y/n): ")
    if confirm.lower() != 'y': return

    conn = psycopg2.connect(Config.DB_DSN)
    cur = conn.cursor()
    
    cur.execute("DROP TABLE IF EXISTS face_detections;")
    cur.execute("DROP TABLE IF EXISTS people;")
    cur.execute("DROP TABLE IF EXISTS image_metadata;")
    
    conn.commit()
    print("Database wiped.")
    conn.close()

if __name__ == "__main__":
    reset()