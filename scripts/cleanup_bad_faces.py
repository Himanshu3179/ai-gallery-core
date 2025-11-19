import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import Database

def cleanup_bad_faces():
    """Remove face detections that are clearly false positives"""
    db = Database()
    conn = db.connect()
    cur = conn.cursor()
    
    print("Analyzing face detections...")
    
    # Get total count before cleanup
    cur.execute("SELECT COUNT(*) FROM face_detections")
    total_before = cur.fetchone()[0]
    
    # Delete faces with:
    # 1. Confidence = 0 (failed detection)
    # 2. Location box covering entire image (x=0, y=0, and large w/h)
    # 3. No proper facial landmarks
    
    print("\nRemoving bad detections...")
    cur.execute("""
        DELETE FROM face_detections
        WHERE confidence < 0.5
           OR (location_box->>'x')::int = 0 
           AND (location_box->>'y')::int = 0
           AND (location_box->>'w')::int > 500
    """)
    
    deleted_count = cur.rowcount
    conn.commit()
    
    # Get count after cleanup
    cur.execute("SELECT COUNT(*) FROM face_detections")
    total_after = cur.fetchone()[0]
    
    print(f"\n✅ Cleanup Complete!")
    print(f"   Before: {total_before} face detections")
    print(f"   Deleted: {deleted_count} bad detections")
    print(f"   After: {total_after} valid face detections")
    
    # Now re-run clustering on clean data
    print("\n🔄 Re-clustering with clean data...")
    
    import numpy as np
    import json
    from sklearn.cluster import DBSCAN
    from tqdm import tqdm
    
    cur.execute("SELECT id, face_embedding FROM face_detections")
    rows = cur.fetchall()
    
    if not rows:
        print("No faces found to cluster.")
        conn.close()
        return
    
    ids = [row[0] for row in rows]
    raw_embeddings = [row[1] for row in rows]
    
    print(f"Clustering {len(raw_embeddings)} faces...")
    
    # Parse embeddings
    clean_embeddings = []
    for emb in raw_embeddings:
        if isinstance(emb, str):
            try:
                clean_vals = json.loads(emb)
            except:
                clean_vals = [float(x) for x in emb.strip('[]').split(',')]
            clean_embeddings.append(clean_vals)
        elif isinstance(emb, list):
            clean_embeddings.append(emb)
        elif isinstance(emb, np.ndarray):
            clean_embeddings.append(emb.tolist())
    
    X = np.array(clean_embeddings)
    
    # DBSCAN Clustering
    clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=0.5, min_samples=3)
    clt.fit(X)
    
    labels = clt.labels_
    unique_labels = set(labels)
    print(f"Found {len(unique_labels) - (1 if -1 in unique_labels else 0)} unique people.")
    
    # Clear old people mapping
    cur.execute("UPDATE face_detections SET person_id = NULL")
    cur.execute("DELETE FROM people")
    
    for label in tqdm(unique_labels, desc="Creating identities"):
        if label == -1:
            continue
        
        cur.execute("INSERT INTO people (name) VALUES (%s) RETURNING id", (f"Person {label}",))
        person_id = cur.fetchone()[0]
        
        indices = [i for i, x in enumerate(labels) if x == label]
        face_ids = [ids[i] for i in indices]
        
        if face_ids:
            cur.execute("UPDATE face_detections SET person_id = %s WHERE id = ANY(%s)", (person_id, face_ids))
    
    conn.commit()
    conn.close()
    print("✅ Re-clustering Complete!")

if __name__ == "__main__":
    cleanup_bad_faces()
