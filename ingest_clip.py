import os
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

# Load env variables (HF_HOME, etc)
load_dotenv()

from rag.clip_embeddings import embed_image

PDF_PATH = "ostxtbook.pdf"
IMAGES_DIR = "images/ostxtbook_pages"
COLLECTION_NAME = "os_images_clip"

def init_qdrant():
    client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333)))
    
    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Creating Qdrant collection '{COLLECTION_NAME}' (dim=512, COSINE)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        
    return client

def ingest_pdf_images():
    print(f"Starting CLIP ingestion for {PDF_PATH}...")
    client = init_qdrant()
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    
    print(f"Total pages to process: {total_pages}")
    
    # Process page by page
    for page_num in range(total_pages):
        page = doc[page_num]
        
        # 1. Render page to image
        # Using a moderate DPI to balance quality and embedding speed
        pix = page.get_pixmap(dpi=100)
        image_path = os.path.join(IMAGES_DIR, f"page_{page_num}.png")
        pix.save(image_path)
        
        # 2. Embed image using CLIP
        try:
            vector = embed_image(image_path)
        except Exception as e:
            print(f"Failed to embed page {page_num}: {e}")
            continue
            
        # 3. Save to Qdrant
        payload = {
            "page": page_num,
            "page_human": page_num + 1,
            "image_path": image_path,
            "abs_path": os.path.abspath(image_path),
            "pdf": PDF_PATH
        }
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=page_num,  # We can just use the page_num as the ID
                    vector=vector,
                    payload=payload
                )
            ]
        )
        
        if (page_num + 1) % 10 == 0:
            print(f"Processed {page_num + 1}/{total_pages} pages...")

    print("CLIP ingestion complete!")

if __name__ == "__main__":
    ingest_pdf_images()
