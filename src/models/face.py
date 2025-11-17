from deepface import DeepFace
from src.config import Config

class FaceModel:
    def __init__(self):
        self.model_name = Config.FACE_MODEL

    def detect_faces(self, image_path):
        """Returns list of dicts with 'embedding', 'facial_area', 'confidence'."""
        try:
            results = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend="retinaface",
                enforce_detection=False
            )
            return results
        except Exception as e:
            # Often raised if no face found
            return []