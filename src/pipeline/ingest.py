import os
import json
from datetime import datetime
from PIL import Image, ExifTags
from tqdm import tqdm
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

    def run(self):
        all_files = self.get_all_files()
        processed = self.get_processed_files()
        to_process = [f for f in all_files if f not in processed]
        
        print(f"Found {len(all_files)} images. Processing {len(to_process)} new ones.")

        if not to_process:
            return

        # --- Phase 1: Scene Understanding ---
        scene_model = SceneModel()
        conn = self.db.connect()
        cur = conn.cursor()

        print("Phase 1: Analyzing Scenes...")
        for img_path in tqdm(to_process):
            caption = scene_model.generate_caption(img_path)
            metadata_json = self._get_metadata(img_path) # Extract metadata
            
            if caption:
                vector = scene_model.get_embedding(caption)
                
                # Insert with metadata
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
        print("Ingestion Complete!")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()