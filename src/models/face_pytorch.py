import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from src.config import Config

class FaceModelPyTorch:
    """PyTorch-based face detection and recognition with MPS/GPU support"""
    
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        print(f"Loading Face Models on {self.device}...")
        
        # MTCNN for face detection (runs on GPU/MPS)
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=True  # Detect multiple faces
        )
        
        # InceptionResnetV1 for face recognition (512-dim embeddings, same as ArcFace)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
    
    def detect_faces(self, image_path):
        """
        Returns list of dicts with 'embedding', 'facial_area', 'confidence'
        Compatible with DeepFace format for drop-in replacement
        """
        try:
            img = Image.open(image_path).convert('RGB')
            
            # Detect faces and get bounding boxes
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None:
                return []
            
            results = []
            
            # Extract face embeddings
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().to(self.device)
            
            for box, prob in zip(boxes, probs):
                if prob < 0.9:  # Confidence threshold
                    continue
                
                # Extract face region
                x1, y1, x2, y2 = [int(b) for b in box]
                face = img.crop((x1, y1, x2, y2))
                
                # Get embedding
                face_tensor = self.mtcnn(face)
                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.resnet(face_tensor).cpu().numpy()[0]
                    
                    # Format to match DeepFace output
                    results.append({
                        'embedding': embedding.tolist(),
                        'facial_area': {
                            'x': x1,
                            'y': y1,
                            'w': x2 - x1,
                            'h': y2 - y1
                        },
                        'face_confidence': float(prob)
                    })
            
            return results
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []
    
    def get_embedding(self, image_path):
        """Get embedding for a single face image"""
        try:
            img = Image.open(image_path).convert('RGB')
            face_tensor = self.mtcnn(img)
            
            if face_tensor is not None:
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.resnet(face_tensor).cpu().numpy()[0]
                return embedding.tolist()
            return None
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
