import sys
import os
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import Database
from src.models.face import FaceModel

def process_faces():
    db = Database()
    conn = db.connect()
    cur = conn.cursor()

    print("--- Step 1: Backfilling Face Detections ---")
    
    # 1. Find images that are in metadata BUT NOT in face_detections
    cur.execute("""
        SELECT m.id, m.image_path 
        FROM image_metadata m
        LEFT JOIN face_detections f ON m.id = f.image_id
        WHERE f.id IS NULL
    """)
    missing_faces = cur.fetchall()
    
    print(f"Found {len(missing_faces)} images that need face scanning.")

    if missing_faces:
        face_model = FaceModel()
        
        for image_id, image_path in tqdm(missing_faces):
            try:
                faces = face_model.detect_faces(image_path)
                
                # If no faces found, we should probably mark it scanned to avoid rescanning
                # For now, we just loop.
                
                for face in faces:
                    embedding = face.get('embedding')
                    area = face.get('facial_area')
                    conf = face.get('face_confidence', 0.0)

                    if embedding:
                        cur.execute("""
                            INSERT INTO face_detections (image_id, face_embedding, location_box, confidence)
                            VALUES (%s, %s, %s, %s)
                        """, (image_id, embedding, json.dumps(area), conf))
                
                conn.commit()
            except Exception as e:
                print(f"Error scanning {image_path}: {e}")
        
        print("Face scanning complete.")
    else:
        print("All images have been scanned for faces.")

    print("\n--- Step 2: Clustering Identities (The 'People' Table) ---")
    
    # 2. Load all face embeddings
    cur.execute("SELECT id, face_embedding FROM face_detections")
    rows = cur.fetchall()
    
    if not rows:
        print("No faces found in database to cluster.")
        return

    ids = [row[0] for row in rows]
    raw_embeddings = [row[1] for row in rows]
    
    print(f"Clustering {len(raw_embeddings)} faces...")
    
    # --- FIX: Parse Embeddings correctly ---
    clean_embeddings = []
    for emb in raw_embeddings:
        # If it comes as a string "[0.1, 0.2]", parse it
        if isinstance(emb, str):
            try:
                # Remove brackets if present and split
                clean_vals = json.loads(emb) 
            except:
                # Fallback for simple string splitting if json fails
                clean_vals = [float(x) for x in emb.strip('[]').split(',')]
            clean_embeddings.append(clean_vals)
        # If it's already a list (unlikely with psycopg2+pgvector sometimes), use as is
        elif isinstance(emb, list):
            clean_embeddings.append(emb)
        # If it's numpy array
        elif isinstance(emb, np.ndarray):
            clean_embeddings.append(emb.tolist())

    # Convert to 2D Numpy Array for Scikit-Learn
    X = np.array(clean_embeddings)

    # Check shape
    if len(X.shape) != 2:
        print(f"Error: Embeddings shape is wrong: {X.shape}. Expected (n_samples, n_features)")
        return

    # metric="cosine" is required for InsightFace vectors
    # min_samples=1 ensures people with only 1 photo are not deleted
    clt = DBSCAN(metric="cosine", n_jobs=-1, eps=0.5, min_samples=1)
    clt.fit(X)
    
    labels = clt.labels_
    
    unique_labels = set(labels)
    print(f"Found {len(unique_labels) - (1 if -1 in unique_labels else 0)} unique people.")

    # Clear old people mapping
    cur.execute("UPDATE face_detections SET person_id = NULL")
    cur.execute("DELETE FROM people")  # Use DELETE instead of TRUNCATE CASCADE 
    
    for label in tqdm(unique_labels):
        if label == -1: continue # Noise

        # Create Person
        cur.execute("INSERT INTO people (name) VALUES (%s) RETURNING id", (f"Person {label}",))
        person_id = cur.fetchone()[0]
        
        # Update Faces
        indices = [i for i, x in enumerate(labels) if x == label]
        face_ids = [ids[i] for i in indices]
        
        if face_ids:
            cur.execute("UPDATE face_detections SET person_id = %s WHERE id = ANY(%s)", (person_id, face_ids))
    
    conn.commit()
    conn.close()
    print("✅ Clustering Complete! 'people' table populated.")

if __name__ == "__main__":
    process_faces()