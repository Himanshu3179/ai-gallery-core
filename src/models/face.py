import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceModel:
    def __init__(self):
        print("Loading InsightFace (Buffalo_L) [M4 Optimized]...")
        # CoreMLExecutionProvider triggers the Apple Neural Engine
        self.app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error reading {image_path}")
            return []

        faces = self.app.get(img)
        results = []

        for face in faces:
            if face.det_score < 0.60: continue

            # Convert numpy types to python types
            bbox = face.bbox.astype(int)
            results.append({
                "embedding": face.normed_embedding.tolist(),
                "confidence": float(face.det_score),
                "facial_area": {
                    "x": int(bbox[0]), "y": int(bbox[1]),
                    "w": int(bbox[2] - bbox[0]), "h": int(bbox[3] - bbox[1])
                }
            })
        
        # This return must be aligned with 'img = ...' (4 spaces indentation)
        return results