import sys
import argparse
import psycopg2
from deepface import DeepFace

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

def label_person(reference_image, name):
    print(f"Analyzing reference image: {reference_image}...")
    
    # 1. Get embedding of the person
    try:
        results = DeepFace.represent(
            img_path=reference_image, 
            model_name=Config.FACE_MODEL,
            detector_backend="retinaface"
        )
        target_embedding = results[0]['embedding']
    except Exception as e:
        print(f"Error processing reference image: {e}")
        return

    conn = psycopg2.connect(Config.DB_DSN)
    cur = conn.cursor()

    # 2. Create or Get Person ID
    cur.execute("INSERT INTO people (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id", (name,))
    person_id = cur.fetchone()[0]
    print(f"Target Person ID: {person_id} ({name})")

    # 3. Find matches in DB (Threshold 0.4 is standard for ArcFace cosine)
    # We update the face_detections table directly
    sql = """
        UPDATE face_detections
        SET person_id = %s
        WHERE face_embedding <=> %s::vector < 0.4
    """
    cur.execute(sql, (person_id, target_embedding))
    updated_count = cur.rowcount
    conn.commit()
    
    print(f"Success! Labeled {updated_count} faces as '{name}'.")
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label faces in the DB based on a reference photo")
    parser.add_argument("image", help="Path to a photo of the person")
    parser.add_argument("name", help="Name of the person")
    
    args = parser.parse_args()
    label_person(args.image, args.name)