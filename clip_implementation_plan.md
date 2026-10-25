# True Multimodal RAG with CLIP

## The Goal
Instead of just relying on text search and exact keyword matching, we will implement true mathematical vector embeddings for textbook images using OpenAI's `clip-ViT-B-32`. This allows the system to semantically match your text query to the visual content of the textbook pages.

## Architecture & Pipeline

### 1. New CLIP Embedding Pipeline
We will use the `sentence-transformers` library to load `clip-ViT-B-32`. This model has a unique property: it can encode both text and images into the *exact same* 512-dimensional vector space.

### 2. PDF Ingestion Phase (The "Slow" Part)
- We will modify the extraction logic to export every page of `ostxtbook.pdf` as a `.png` file.
- We will pass every single `.png` through the CLIP model to generate a 512-dimensional image vector.
- These vectors will be stored in a **new Qdrant collection** named `os_images_clip`.

### 3. Query & Retrieval Phase
- When you ask a question like *"Show me the inverted page table"*, the backend will detect the visual intent.
- It will pass your query text through the same CLIP model to get a 512-dimensional text vector.
- Qdrant will perform a cosine similarity search against `os_images_clip` to find the image whose mathematical vector is closest to your text vector.
- Streamlit will display the winning image screenshot inline!

---

## Proposed Changes

### [NEW] `rag/clip_embeddings.py`
- A utility script to load the CLIP model and provide two functions: `embed_image(image_path)` and `embed_text(query)`.

### [MODIFY] `rag/document_processor.py`
- Update `ingest_pdf()` to also call the new CLIP ingestion pipeline. It will:
  1. Render each page as a `.png`.
  2. Embed it using CLIP.
  3. Upload the vector and payload (page number, image path) to Qdrant.

### [MODIFY] `main.py` & `api.py`
- Initialize the `os_images_clip` collection in Qdrant (dimension 512, distance COSINE).
- Update the `POST /images/search` endpoint to use CLIP vector search instead of keyword matching.

### [MODIFY] `app.py`
- Ensure the UI correctly displays the full-page screenshot returned by the new search endpoint.

---

## Verification Plan
1. Delete the old Qdrant volume if necessary, or just run the new ingestion script.
2. Wait for the CLIP model to download and process the 1,200 pages (this may take ~10-30 minutes on CPU).
3. Ask the frontend: *"show me the inverted page table diagram"*.
4. Verify that the correct textbook page screenshot is returned based on mathematical similarity rather than keyword matching.

## User Review Required
> [!WARNING]  
> This approach requires generating embeddings for all 1,200+ pages of the textbook. On a standard CPU, this ingestion process will take a significant amount of time (potentially 15-45 minutes depending on your exact CPU speed). **You only have to do this once**, but please be prepared to let your laptop run for a while during the ingestion phase. 

Are you ready to proceed with this architecture?
