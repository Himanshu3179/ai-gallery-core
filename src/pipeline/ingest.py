import os
import json
import numpy as np
from datetime import datetime
from PIL import Image, ExifTags
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from src.config import Config
from src.database import Database
from src.models.scene import SceneModel
from src.models.face import FaceModel

class IngestionPipeline:
    def __init__(self):
        self.db = Database()
        self.db.init_db()
        self.valid_exts = ('.jpg', '.jpeg', '.png', '.webp')

    def get_all_files(self):
        files = []
        for root, _, filenames in os.walk(Config.IMAGE_SOURCE_DIR):
            for f in filenames:
                if f.lower().endswith(self.valid_exts):
                    files.append(os.path.abspath(os.path.join(root, f)))
        return files

    def get_processed_files(self):
        conn = self.db.connect()
        cur = conn.cursor()
        cur.execute("SELECT image_path FROM image_metadata")
        processed = {row[0] for row in cur.fetchall()}
        return processed

    def _get_metadata(self, image_path):
        """Extracts basic metadata and EXIF date"""
        stats = os.stat(image_path)
        meta = {
            "size_bytes": stats.st_size,
            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "width": 0,
            "height": 0,
            "date_taken": None
        }

        try:
            with Image.open(image_path) as img:
                meta["width"], meta["height"] = img.size
                
                # Extract EXIF Date Taken if available
                exif = img._getexif()
                if exif:
                    for tag, value in exif.items():
                        tag_name = ExifTags.TAGS.get(tag, tag)
                        if tag_name == "DateTimeOriginal":
                            meta["date_taken"] = value
                            break
        except Exception:
            pass
        
        return json.dumps(meta)

    def run_clustering(self):
        """Phase 3: Groups faces into People (identities)"""
        print("\n--- Phase 3: Clustering Identities ---")
        conn = self.db.connect()
        cur = conn.cursor()

        # 1. Load all face embeddings
        cur.execute("SELECT id, face_embedding FROM face_detections")
        rows = cur.fetchall()
        
        if not rows:
            print("No faces found to cluster.")
            return

        ids = [row[0] for row in rows]
        raw_embeddings = [row[1] for row in rows]
        
        print(f"Clustering {len(raw_embeddings)} faces...")

        # --- CRITICAL FIX: Parse Embeddings (String -> Float List) ---
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

        # Convert to Numpy
        X = np.array(clean_embeddings)
        
        # --- UPDATED CLUSTERING LOGIC ---
        # metric="cosine": Correct for InsightFace (normalized vectors)
        # eps=0.5: Good starting point (Tune with scripts/tune_clustering.py if needed)
        # min_samples=3: Keeps gallery clean (needs 3 similar faces to make a 'Person')
        clt = DBSCAN(metric="cosine", n_jobs=-1, eps=0.45, min_samples=3)
        clt.fit(X)
        labels = clt.labels_
        
        unique_labels = set(labels)
        print(f"Found {len(unique_labels) - (1 if -1 in unique_labels else 0)} unique people.")

        # Reset people mapping to rebuild it cleanly
        cur.execute("UPDATE face_detections SET person_id = NULL")
        cur.execute("DELETE FROM people") 
        
        for label in tqdm(unique_labels):
            if label == -1: continue # Unknown/Noise faces

            # Create Person Entry
            cur.execute("INSERT INTO people (name) VALUES (%s) RETURNING id", (f"Person {label}",))
            person_id = cur.fetchone()[0]
            
            # Update Faces
            indices = [i for i, x in enumerate(labels) if x == label]
            face_ids = [ids[i] for i in indices]
            
            if face_ids:
                cur.execute("UPDATE face_detections SET person_id = %s WHERE id = ANY(%s)", (person_id, face_ids))
        
        conn.commit()
        print("✅ Clustering Complete.")

    def run(self):
        all_files = self.get_all_files()
        processed = self.get_processed_files()
        to_process = [f for f in all_files if f not in processed]
        
        print(f"Found {len(all_files)} images. Processing {len(to_process)} new ones.")

        if to_process:
            # --- Phase 1: Scene Understanding ---
            scene_model = SceneModel()
            conn = self.db.connect()
            cur = conn.cursor()

            print("Phase 1: Analyzing Scenes...")
            for img_path in tqdm(to_process):
                caption = scene_model.generate_caption(img_path)
                metadata_json = self._get_metadata(img_path)
                
                if caption:
                    vector = scene_model.get_embedding(caption)
                    
                    cur.execute("""
                        INSERT INTO image_metadata (image_path, caption, scene_embedding, meta_data)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (image_path) DO NOTHING
                    """, (img_path, caption, vector, metadata_json))
                    conn.commit()
            
            scene_model.unload()
            print("Scene analysis complete. Models unloaded.")

            # --- Phase 2: Face Recognition ---
            face_model = FaceModel()
            
            print("Phase 2: Scanning Faces...")
            for img_path in tqdm(to_process):
                cur.execute("SELECT id FROM image_metadata WHERE image_path = %s", (img_path,))
                res = cur.fetchone()
                if not res: continue
                image_id = res[0]

                faces = face_model.detect_faces(img_path)
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
            self.db.close()
        
        else:
            print("Skipping Phase 1 & 2 (No new images).")

        # --- Phase 3: Clustering (ALWAYS RUNS) ---
        self.run_clustering()
        
        print("Ingestion Complete!")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()