import os
import torch
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_USER = os.getenv("DB_USER", "admin")
    DB_PASS = os.getenv("DB_PASS", "password")
    DB_NAME = os.getenv("DB_NAME", "ai_gallery")
    
    DB_DSN = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    IMAGE_SOURCE_DIR = os.getenv("IMAGE_SOURCE_DIR", "/Users/apple/Downloads/photos_phone")
    
    # Intelligent Device Selection
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    # Model Configs
    MOONDREAM_ID = "vikhyatk/moondream2"
    MOONDREAM_REV = "2024-08-26"
    EMBED_MODEL = "all-MiniLM-L6-v2"
    FACE_MODEL = "ArcFace"