import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from src.config import Config

class SceneModel:
    def __init__(self):
        self.device = Config.DEVICE
        print(f"Loading Scene Models on {self.device} (float16)...")
        
        # Load Moondream (Vision Language) with float16
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MOONDREAM_ID, revision=Config.MOONDREAM_REV)
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MOONDREAM_ID, 
            trust_remote_code=True, 
            revision=Config.MOONDREAM_REV,
            torch_dtype=torch.float16  # <--- This was the key setting
        ).to(self.device)
        
        # Load Embedding Model (Text -> Vector)
        self.embed_model = SentenceTransformer(Config.EMBED_MODEL, device=self.device)

    def generate_caption(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Resize image to optimization sweet spot
            image.thumbnail((768, 768)) 
            
            # Run inference (Default mode, no explicit no_grad)
            enc_image = self.model.encode_image(image)
            prompt = "Describe this image in detail, focusing on the setting, objects, colors, and lighting."
            return self.model.answer_question(enc_image, prompt, self.tokenizer)
        except Exception as e:
            print(f"Error captioning {image_path}: {e}")
            return None

    def get_embedding(self, text):
        return self.embed_model.encode(text).tolist()
    
    def unload(self):
        """Basic unload"""
        del self.model
        del self.tokenizer
        del self.embed_model
        if self.device == "mps":
            torch.mps.empty_cache()