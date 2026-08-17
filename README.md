# 🎓 OS Tutor AI - RAG MVP

A full-stack Retrieval-Augmented Generation (RAG) educational chatbot specifically designed to teach Operating Systems based on the Galvin OS Textbook. 

## ✨ Key Features
- **Modern UI**: A premium, ChatGPT-like Streamlit frontend with a dynamic streaming typewriter effect and clean aesthetics.
- **Robust Backend**: FastAPI application managing state, endpoints, and the RAG pipeline.
- **Local Embeddings**: Fully offline, free embeddings using `BAAI/bge-small-en-v1.5` via Hugging Face.
- **Vector Storage**: High-performance retrieval using Qdrant.
- **Automated Evaluation**: Built-in Ragas evaluation script using a local `Qwen2.5-7B-Instruct` judge to benchmark Faithfulness, Answer Relevancy, and Context Precision against a Baseline LLM.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create or edit your `.env` file in the root directory:
```ini
# Core AI & Database
GOOGLE_API_KEY=your_gemini_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION=major_project

# Advanced PDF Parsing (Optional)
LLAMA_CLOUD_API_KEY=your_llama_parse_key

# Evaluation / Local Models
HF_TOKEN=your_huggingface_token
HF_HOME=D:\Projects\Major_Project\hf_cache  # Redirect heavy model weights away from C: drive
```

### 3. Build the Knowledge Base
Ingest the Galvin OS Textbook into your Qdrant Vector Store (one-time setup):
```bash
python setup_db.py
```

### 4. Run the Application
You will need two separate terminal windows.

**Terminal 1 (Backend):**
```bash
python api.py
```
*Wait for the `Uvicorn running on http://0.0.0.0:8000` message.*

**Terminal 2 (Frontend):**
```bash
python -m streamlit run app.py
```
*Your browser will automatically open the OS Tutor at `http://localhost:8501`.*

---

## 📊 Batch Evaluation System

We utilize an automated benchmarking pipeline using the [Ragas framework](https://docs.ragas.io/) to mathematically prove the effectiveness of our RAG architecture compared to a standard LLM.

### Running the Evaluator:
```bash
python eval/run_eval.py
```

### How it works:
1. **Dataset**: Iterates through 20 "Gold Standard" deadlock questions based on Galvin Chapter 8.
2. **Local Judge**: Uses `Qwen/Qwen2.5-7B-Instruct` running locally (cached on your D: drive) to judge responses neutrally.
3. **Metrics Computed**:
   - **Faithfulness**: Is the answer derived *only* from the textbook context? (No hallucinations).
   - **Answer Relevancy**: Does the answer directly address the user's question?
   - **Context Precision**: Did the vector database fetch the correct paragraphs?
4. **Output**: Progress is safely checkpointed to `eval/results.csv` every 5 questions, and a final `evaluation_report.md` is generated for research papers.

---

## 📂 Project Structure

```text
Project-major/
├── app.py                   ← Streamlit Frontend (ChatGPT UI)
├── api.py                   ← FastAPI Backend Server
├── setup_db.py              ← Initial Vector DB Ingestion Script
├── .env                     ← Environment keys & HF_HOME configs
├── requirements.txt         ← Project Dependencies
├── eval/
│   ├── run_eval.py          ← Ragas Evaluation Pipeline
│   ├── results.csv          ← Raw evaluation metrics (generated)
│   └── evaluation_report.md ← Generated markdown table
└── rag/
    ├── document_processor.py  ← PDF loading & chunking
    ├── embeddings.py          ← HuggingFace embeddings config
    ├── vector_store.py        ← Qdrant operations
    └── qa_chain.py            ← LangChain QA Retrieval logic
```
