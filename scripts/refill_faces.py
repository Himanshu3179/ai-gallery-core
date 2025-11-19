import sys
import os
import json
import psycopg2
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import Config
from src.models.face import FaceModel

def refill_faces():
    print("--- STARTING FACE RE-SCAN ---")
    conn = psycopg2.connect(Config.DB_DSN)
    cur = conn.cursor()

    # Fetch existing images (so we don't lose captions)
    cur.execute("SELECT id, image_path FROM image_metadata")
    images = cur.fetchall()
    print(f"Scanning {len(images)} existing images...")

    face_model = FaceModel()
    count = 0

    for img_id, img_path in tqdm(images):
        if not os.path.exists(img_path): continue

        try:
            faces = face_model.detect_faces(img_path)
            for face in faces:
                cur.execute("""
                    INSERT INTO face_detections (image_id, face_embedding, location_box, confidence)
                    VALUES (%s, %s, %s, %s)
                """, (img_id, json.dumps(face['embedding']), json.dumps(face['facial_area']), face['confidence']))
                count += 1
            conn.commit()
        except Exception as e:
            print(f"Error {img_path}: {e}")

    conn.close()
    print(f"✅ Done! Detected {count} faces.")

if __name__ == "__main__":
    refill_faces()