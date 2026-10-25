import os
from PIL import Image
from sentence_transformers import SentenceTransformer

# Load model lazily
_model = None

def get_clip_model():
    global _model
    
    # Safety toggle: If running the massive 3GB Qwen model locally, loading CLIP
    # on top of it causes a Windows Pagefile OOM crash. We bypass it and rely 
    # exclusively on the Text-Guided Figure Index fallback instead.
    if os.getenv("LLM_BACKEND", "").lower() == "qwen":
        print("⚠️ Bypassing CLIP visual model to save RAM (Qwen is active). Using Figure Index fallback.")
        return None
        
    if _model is None:
        print("Loading CLIP ViT-B-32 model for multimodal embeddings...")
        _model = SentenceTransformer('clip-ViT-B-32')
    return _model

def embed_image(image_path: str):
    """Generates a 512-dim embedding for a local image file."""
    model = get_clip_model()
    if model is None:
        return [0.0] * 512
        
    # SentenceTransformer can automatically encode PIL Images
    img = Image.open(image_path)
    embedding = model.encode(img)
    return embedding.tolist()

def embed_text(text: str):
    """Generates a 512-dim embedding for a text query."""
    model = get_clip_model()
    if model is None:
        return [0.0] * 512
        
    embedding = model.encode(text)
    return embedding.tolist()
